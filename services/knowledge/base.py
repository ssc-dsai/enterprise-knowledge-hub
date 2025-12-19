"""Base class for knowledge services."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
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

    def run(self) -> None:
        """Run the knowledge ingestion/processing in parallel threads."""
        self.logger.info("Running knowledge ingestion for %s", self.service_name)
        self._producer_done.clear()
        self._stats.reset()  # Reset stats at the start of each run
        with ThreadPoolExecutor(max_workers=2) as executor:
            queue_future = executor.submit(self.queue_for_processing)
            process_future = executor.submit(self.process)
            # Wait for both to complete and propagate any exceptions
            queue_future.result()
            process_future.result()

    @abstractmethod
    def fetch_from_source(self) -> Iterator[KnowledgeItem]:
        """Read data from a source that can be anything and will pass the message to the ingest queue."""
        raise NotImplementedError("Subclasses must implement the read method.")

    @abstractmethod
    def process_queue(self, knowledge_item: dict[str, object]) -> None:
        """Process ingested data from the queue."""
        raise NotImplementedError("Subclasses must implement the process method.")

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
            self.logger.info("Done processing with ingestion for %s", self.service_name)

    def process(self) -> None:
        """Process ingested data. Keeps polling until producer is done and queue is empty."""
        self.logger.info("Processing ingested data. (%s)", self.service_name)
        queue_name = self.service_name + ".ingest"
        try:
            while True:
                # Drain all available messages
                for item in self.queue_service.read(queue_name):
                    self.process_queue(item)
                    self._stats.record_processed()
                # Queue is empty - check if we should exit or wait
                if self._producer_done.is_set():
                    break  # Producer done and queue empty
                time.sleep(self._poll_interval)
        except Exception as e:
            self.logger.exception("Error during processing for %s: %s", self.service_name, e)
        finally:
            self.logger.info("Done processing ingested data. (%s)", self.service_name)
