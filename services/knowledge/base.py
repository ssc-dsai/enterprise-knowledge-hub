"""Base class for knowledge services."""
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
import logging
import threading
import time
from services.knowledge.models import KnowledgeItem, DatabaseWikipediaItem
from services.queue.queue_service import QueueService
from services.stats.knowledge_service_stats import KnowledgeServiceStats
from repository.postgrespg import WikipediaDbRecord

QUEUE_BATCH_NAME = "wikipedia_embeddings_sink"

@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str
    _producer_done: threading.Event = field(default_factory=threading.Event, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _poll_interval: float = 0.5  # seconds to wait before retrying empty queue
    _stats: KnowledgeServiceStats = field(default_factory=KnowledgeServiceStats, init=False)
    _is_ingestion_queue_complete = False

    @property
    def stats(self) -> KnowledgeServiceStats:
        """Get the statistics tracker for this service."""
        return self._stats

    def run(self) -> None:
        """Run the knowledge ingestion/processing in parallel threads."""
        self.logger.info("Running knowledge ingestion for %s", self.service_name)
        self._producer_done.clear()
        self._stop_event.clear()
        self._stats.reset()  # Reset stats at the start of each run
        with ThreadPoolExecutor(max_workers=3) as executor:
            queue_future = executor.submit(self.queue_for_processing)
            process_future = executor.submit(self.process)
            insert_future = executor.submit(self.process_wikipedia_sink)
            # Wait for both to complete and propagate any exceptions
            queue_future.result()
            process_future.result()
            insert_future.result()

    @abstractmethod
    def fetch_from_source(self) -> Iterator[KnowledgeItem]:
        """Read data from a source that can be anything and will pass the message to the ingest queue."""
        raise NotImplementedError("Subclasses must implement the read method.")

    @abstractmethod
    def process_queue(self, knowledge_item: dict[str, object]):
        """Process ingested data from the queue. May return a single item or a list of items."""
        raise NotImplementedError("Subclasses must implement the process method.")

    @abstractmethod
    def store_item(self, item: KnowledgeItem) -> None:
        """Store the processed knowledge item into the knowledge base."""
        raise NotImplementedError("Subclasses must implement the store_item method.")

    @abstractmethod
    def insert_item(self, item: dict[str, object]) -> None:
        """Insert the object into repository"""
        raise NotImplementedError("Subclasses must implement the insert_item method.")

    def queue_for_processing(self) -> None:
        """Ingest data into the knowledge base."""
        self.logger.info("Ingesting data into the knowledge base. (%s)", self.service_name)
        try:
            for item in self.fetch_from_source():
                if self._stop_event.is_set():
                    break
                self.queue_service.write(self.service_name + ".ingest", item.to_dict())
                self._stats.record_added()
        except Exception as e:
            self.logger.exception("Error during ingestion for %s: %s", self.service_name, e)
        finally:
            self._producer_done.set()  # Signal that producer is finished
            self.logger.info("Done ingestion for %s", self.service_name)

    def process(self) -> None:
        """Process ingested data. Keeps polling until producer is done and queue is empty."""
        self.logger.info("Processing ingested data. (%s)", self.service_name)
        queue_name = self.service_name + ".ingest"
        try:
            while not self._stop_event.is_set():
                # Drain all available messages
                for item, delivery_tag in self.queue_service.read(queue_name):
                    try:
                        if self._stop_event.is_set():
                            self.logger.info("Stop event is true.  Stopping process loop")
                            self._ack_message(delivery_tag, successful=False)
                            break
                        processed = self.process_queue(item) # GPU work happens here
                        items = processed if isinstance(processed, list) else [processed]
                        for item_with_embedding in items:
                            self.store_item(item_with_embedding)
                        self._stats.record_processed()
                        self._ack_message(delivery_tag, successful=True)
                    except Exception as e:
                        self.logger.exception("Error processing item in %s: %s", self.service_name, e)
                        self._ack_message(delivery_tag, successful=False)
                # Queue is empty - check if we should exit or wait
                if self._producer_done.is_set():
                    break  # Producer done and ingestion queue empty
                time.sleep(self._poll_interval)
        except Exception as e:
            self.logger.exception("Error during processing for %s: %s", self.service_name, e)
        finally:
            try:
                self.finalize_processing()
            except Exception as e:
                self.logger.exception("Error during finalize_processing for %s: %s", self.service_name, e)
            self.logger.info("Done processing ingested data. (%s)", self.service_name)

    def process_wikipedia_sink(self) -> None:
        """
            Process wikipedia embedding sink queue
            Inserts into database essentially
        """
        self.logger.info("Processing wikipedia embedding sink data. (%s)", self.service_name)
        try:
            while not self._stop_event.is_set():
                for item, delivery_tag in self.queue_service.read(QUEUE_BATCH_NAME):
                    try:
                        if self._stop_event.is_set():
                            self.logger.info("Stop event is true.  Stopping wiki sink loop")
                            self._ack_message(delivery_tag, successful=False)
                            break
                        #lol we can fix this afterwards.  So much conversion
                        wiki_item = DatabaseWikipediaItem.from_rabbitqueue_dict(item)
                        record_to_insert = WikipediaDbRecord.from_item(wiki_item)
                        if os.getenv("DB_SKIP_STORE", "false").lower() not in ("1", "true", "yes"):
                            self.insert_item(record_to_insert.as_mapping())
                        self._ack_message(delivery_tag, successful=True)
                    except Exception as e:
                        self.logger.exception("Error processing item in %s: %s", self.service_name, e)
                        self._ack_message(delivery_tag, successful=False)
                if self._producer_done.is_set() and self._get_is_ingestion_queue_complete():
                    break  # Producer done and ingestion queue and sink queue empty
                time.sleep(self._poll_interval)
        except Exception as e:
            self.logger.exception("Error during processing for wikipedia embedding sink %s: %s", self.service_name, e)
        finally:
            try:
                self.finalize_processing()
            except Exception as e:
                self.logger.exception("Error during finalize_processing for %s: %s", self.service_name, e)
            self.logger.info("Done processing wiki sink data. (%s)", self.service_name)

    def finalize_processing(self) -> None:
        """Optional hook called after processing loop ends."""
        self._is_ingestion_queue_complete = True

    def _ack_message(self, delivery_tag, successful: bool):
        """Acknoledge or unack message back to queue"""
        if delivery_tag is not None:
            self.queue_service.read_ack(delivery_tag, successful=successful)

    def _get_is_ingestion_queue_complete(self) -> bool:
        """Getter for _is_ingestion_queue_complete"""
        if self._is_ingestion_queue_complete is not None and self._is_ingestion_queue_complete:
            return True
        return False

    def request_stop(self) -> None:
        """Stop event for knowledge process"""
        self._stop_event.set()

    def should_stop(self) -> bool:
        """Return true if and only if the internal flag is true."""
        return self._stop_event.is_set()
