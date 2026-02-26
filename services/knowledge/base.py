"""Base class for knowledge services."""
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor
from typing import Any
import logging
import threading
from services.knowledge.models import KnowledgeItem
from services.knowledge.batch_handler import BatchHandler
from services.knowledge.wikipedia.models import WikipediaItemProcessed
from services.queue.queue_worker import QueueWorker
from services.queue.queue_service import QueueService
from services.stats.knowledge_service_stats import KnowledgeServiceStats

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
            queue_future = executor.submit(self.ingest)
            process_future = executor.submit(self.process)
            insert_future = executor.submit(self.store)
            # Wait for both to complete and propagate any exceptions
            queue_future.result()
            process_future.result()
            insert_future.result()

    @abstractmethod
    def fetch_from_source(self) -> Iterator[KnowledgeItem]:
        """Read data from a source that can be anything and transform into KnowledgeItem object"""
        raise NotImplementedError("Subclasses must implement the read method.")

    @abstractmethod
    def emit_fetched_item(self, item: KnowledgeItem) -> None:
        """Take fetched and transformed item, and pass to the .raw queue"""
        raise NotImplementedError("Subclasses must implement the emit_fetched_item method.")

    @abstractmethod
    def process_item(self, knowledge_item: Any):
        """Process ingested data from the queue. May return a single item or a list of items."""
        raise NotImplementedError("Subclasses must implement the process_item method.")

    @abstractmethod
    def emit_processed_item(self, item: KnowledgeItem) -> None:
        """Take processed item, and pass to the .processed queue"""
        raise NotImplementedError("Subclasses must implement the emit_processed_item method.")

    @abstractmethod
    def store_item(self, item: WikipediaItemProcessed) -> None:
        """Insert the object into repository"""
        raise NotImplementedError("Subclasses must implement the store_item method.")

    @abstractmethod
    def get_batch_size(self) -> int:
        """Get the set batch size"""
        raise NotImplementedError("Subclasses must implement the get_batch_size method.")

    def _ingest_queue_name(self) -> str:
        """Return ingestion queue name.  Ingest raw source into embedding ready units"""
        return self.service_name + ".raw"

    def _processed_queue_name(self) -> str:
        """Return indexing queue name. Post-embedding, pre-storage ready units"""
        return self.service_name + ".processed"

    def ingest(self) -> None:
        """Ingest data into the knowledge base."""
        self.logger.info("Ingesting data into the knowledge base. (%s)", self.service_name)
        try:
            for item in self.fetch_from_source():
                if self._stop_event.is_set():
                    break
                self.emit_fetched_item(item)
                self._stats.record_added()
        except Exception:
            self.logger.exception("Error during ingestion for %s", self.service_name)
        finally:
            self._producer_done.set()  # Signal that producer is finished
            self.logger.info("Done ingestion for %s", self.service_name)

    def process(self) -> None:
        """Process ingested data. Keeps polling until producer is done and queue is empty."""
        self.logger.info("Processing ingested data from queue: %s. (%s)", self._ingest_queue_name(), self.service_name)

        batch_size = self.get_batch_size()
        worker = QueueWorker(
            queue_service=self.queue_service,
            logger=self.logger,
            stop_event=self._stop_event,
            poll_interval=self._poll_interval
        )

        def acknowledge(delivery_tag: int, successful: bool):
            self.queue_service.read_ack(delivery_tag, successful)

        handler = BatchHandler(self.process_item, acknowledge, batch_size, self.logger)

        def should_exit(drained_any: bool) -> bool:
            #Producer done and ingestion queue empty AND queue was empty this iteration
            return self._producer_done.is_set() and not drained_any

        try:
            worker.run(
                queue_name=self._ingest_queue_name(),
                service_name=self.service_name,
                handler=handler,
                should_exit=should_exit
            )
        except Exception:
            self.logger.exception("Error during processing for queue: %s. (%s)",
                            self._ingest_queue_name(), self.service_name)
        finally:
            try:
                self.finalize_processing()
            except Exception:
                self.logger.exception("Error during finalize_processing for queue: %s. (%s)",
                                self._ingest_queue_name(), self.service_name)
            self.logger.info("Done processing ingested data from queue: %s. (%s)", self._ingest_queue_name(),
                                                                                self.service_name)

    def store(self) -> None:
        """
            Process {service_name}.processed queue
            Inserts into database
        """
        self.logger.info("Processing processed data from queue: %s. (%s)", self._processed_queue_name(),
                                                                        self.service_name)

        worker = QueueWorker(
            queue_service=self.queue_service,
            logger=self.logger,
            stop_event=self._stop_event,
            poll_interval=self._poll_interval
        )

        def handler(item: WikipediaItemProcessed, delivery_tag: str) -> bool:
            if os.getenv("DB_SKIP_STORE", "false").lower() not in ("1", "true", "yes"):
                self.store_item(WikipediaItemProcessed.model_validate(item))
            self.logger.debug("DeliveryTag: %s", delivery_tag)
            # this is to tell queueworker to handle ack
            return False

        def should_exit(drained_any: bool) -> bool:
            #Producer done and ingestion queue empty AND queue was empty this iteration
            return self._producer_done.is_set() and not drained_any

        try:
            worker.run(
                queue_name=self._processed_queue_name(),
                service_name=self.service_name,
                handler=handler,
                should_exit=should_exit
            )
        except Exception:
            self.logger.exception("Error during processing for queue: %s. (%s)", self._processed_queue_name(),
                                                                            self.service_name)
        finally:
            try:
                self.finalize_processing()
            except Exception:
                self.logger.exception("Error during finalize_processing for queue: %s. (%s)",
                                                                        self._processed_queue_name(), self.service_name)
            self.logger.info("Done processing processed data from queue: %s. (%s)", self._processed_queue_name(),
                                                                            self.service_name)

    def finalize_processing(self) -> None:
        """Optional hook called after processing loop ends."""
        self._is_ingestion_queue_complete = True

    def _ack_message(self, delivery_tag, successful: bool):
        """Acknoledge or unack message back to queue"""
        if delivery_tag is not None:
            self.queue_service.read_ack(delivery_tag, successful=successful)

    def request_stop(self) -> None:
        """Stop event for knowledge process"""
        self._stop_event.set()

    def should_stop(self) -> bool:
        """Return true if and only if the internal flag is true."""
        return self._stop_event.is_set()
