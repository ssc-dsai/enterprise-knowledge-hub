from abc import ABC, abstractmethod
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import logging
from services.knowledge.models import KnowledgeItem
from services.queue.queue_service import QueueService


@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str

    def run(self) -> None:
        """Run the knowledge ingestion/processing in parallel threads."""
        self.logger.info("Running knowledge ingestion for %s", self.service_name)
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
        except Exception as e:
            self.logger.exception("Error during ingestion for %s: %s", self.service_name, e)
        finally:
            self.logger.info("Done processing with ingestion for %s", self.service_name)

    def process(self) -> None:
        """Process ingested data."""
        self.logger.info("Processing ingested data. (%s)", self.service_name)
        # Placeholder for processing logic
        try:
            for item in self.queue_service.read(self.service_name + ".ingest"):
                self.process_queue(item)
        except Exception as e:
            self.logger.exception("Error during processing for %s: %s", self.service_name, e)
        finally:
            self.logger.info("Done processing ingested data. (%s)", self.service_name)
