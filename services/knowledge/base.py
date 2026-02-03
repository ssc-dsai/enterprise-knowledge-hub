"""Base class for knowledge services."""
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
import logging
import threading
from services.knowledge.models import KnowledgeItem
from services.queue.queue_service import QueueService
from services.stats.knowledge_service_stats import KnowledgeServiceStats

@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str
    _stop_event: threading.Event
    _producer_done: threading.Event = field(default_factory=threading.Event, init=False)
    _poll_interval: float = 0.5  # seconds to wait before retrying empty queue
    _stats: KnowledgeServiceStats = field(default_factory=KnowledgeServiceStats, init=False)
    _is_ingestion_queue_complete = False


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

    @abstractmethod
    def insert_item(self, item: dict[str, object]) -> None:
        """Insert the object into repository"""
        raise NotImplementedError("Subclasses must implement the insert_item method.")

    @abstractmethod
    def ingest(self) -> None:
        """Ingest data into the knowledge base."""
        raise NotImplementedError("Subclasses must implement the ingest method.")

    @abstractmethod
    def process(self) -> None:
        """Process ingested data. Keeps polling until producer is done and queue is empty."""
        raise NotImplementedError("Subclasses must implement the process method.")

    @abstractmethod
    def store(self) -> None:
        """
            Process embedding sink queue
            Inserts into database essentially
        """
        raise NotImplementedError("Subclasses must implement the store method.")

    def _ingest_queue_name(self) -> str:
        """Return ingestion queue name.  Ingest raw source into embedding ready units"""
        return self.service_name + ".raw"

    def _process_queue_name(self) -> str:
        """Return indexing queue name. Post-embedding, pre-storage ready units"""
        return self.service_name + ".processed"

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

