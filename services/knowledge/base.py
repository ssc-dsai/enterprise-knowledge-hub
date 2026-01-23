"""Base class for knowledge services."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
import logging
import threading
import time
from services.knowledge.models import KnowledgeItem
from services.queue.queue_service import QueueService
from services.stats.knowledge_service_stats import KnowledgeServiceStats

@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str
    _producer_done: threading.Event = field(default_factory=threading.Event, init=False)
    _poll_interval: float = 0.5  # seconds to wait before retrying empty queue
    _stats: KnowledgeServiceStats = field(default_factory=KnowledgeServiceStats, init=False)

    @property
    def stats(self) -> KnowledgeServiceStats:
        """Get the statistics tracker for this service."""
        return self._stats

    @abstractmethod
    def run(self) -> None:
        """Run the knowledge ingestion/processing in parallel threads."""
        raise NotImplementedError("Subclasses must implement the run method.")

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

    def queue_for_processing(self) -> None:
        """Ingest data into the knowledge base."""
        self.logger.info("Ingesting data into the knowledge base. (%s)", self.service_name)
        try:
            for item in self.fetch_from_source():
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
            while True:
                # Drain all available messages
                for item, delivery_tag in self.queue_service.read(queue_name):
                    try:
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
                    break  # Producer done and queue empty
                time.sleep(self._poll_interval)
        except Exception as e:
            self.logger.exception("Error during processing for %s: %s", self.service_name, e)
        finally:
            try:
                self.finalize_processing()
            except Exception as e:
                self.logger.exception("Error during finalize_processing for %s: %s", self.service_name, e)
            self.logger.info("Done processing ingested data. (%s)", self.service_name)

    def finalize_processing(self) -> None:
        """Optional hook called after processing loop ends."""
        return

    def _ack_message(self, delivery_tag, successful: bool):
        if delivery_tag is not None:
            self.queue_service.read_ack(delivery_tag, successful=successful)
