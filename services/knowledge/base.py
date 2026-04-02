"""Base class for knowledge services."""
import os
from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, Future
from typing import Any, Optional
import logging
from random import random
import threading
from datetime import datetime

from repository.postgrespg import WikipediaPgRepository
from services.knowledge.models import KnowledgeItem
from services.knowledge.batch_handler import BatchHandler
from services.knowledge.wikipedia.models import WikipediaItemProcessed
from services.knowledge.models import RunStatus
from services.queue.queue_worker import QueueWorker
from services.queue.queue_service import QueueService

@dataclass
class KnowledgeService(ABC):
    """Abstract base class for knowledge services."""
    queue_service: QueueService
    logger: logging.Logger
    service_name: str
    _repository: WikipediaPgRepository | None = None  # assigned in subclass init after super() call
    _run_id = None  # Assigned at runtime for tracking in logs and stats
    _ingest_done: threading.Event = field(default_factory=threading.Event, init=False)
    _process_done: threading.Event = field(default_factory=threading.Event, init=False)
    _stop_event: threading.Event = field(default_factory=threading.Event, init=False)
    _poll_interval: float = 0.5  # seconds to wait before retrying empty queue
    _executor: ThreadPoolExecutor = ThreadPoolExecutor(max_workers=3)
    _futures: list[Future] = field(default_factory=list)

    def run(self) -> None:
        """Run the knowledge ingestion/processing in parallel threads."""
        self.logger.info("Running knowledge ingestion for %s", self.service_name)
        self._ingest_done.clear()
        self._process_done.clear()
        self._stop_event.clear()
        self._run_id = int(random() * 1e6)  # Assign a random run ID for tracking in logs and stats
        # Record the start of this run in the run_history table for observability
        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                          RunStatus.RUN_STARTED, None, datetime.now())

        ingestion_enabled = os.getenv("SVC_KB_ENABLE_INGESTION", "true").lower() in ("1", "true", "yes")
        processing_enabled = os.getenv("SVC_KB_ENABLE_PROCESSING", "true").lower() in ("1", "true", "yes")
        storing_enabled = os.getenv("SVC_KB_ENABLE_STORING", "true").lower() in ("1", "true", "yes")

        if self._executor is None:
            self._executor = ThreadPoolExecutor(max_workers=3)

        self._futures = []

        if ingestion_enabled:
            self._futures.append(self._executor.submit(self.ingest))
        else:
            self.logger.info("Ingestion disabled via SVC_KB_ENABLE_INGESTION (%s)", self.service_name)
            self._ingest_done.set()

        if processing_enabled:
            self._futures.append(self._executor.submit(self.process))
        else:
            self.logger.info("Processing disabled via SVC_KB_ENABLE_PROCESSING (%s)", self.service_name)
            self._process_done.set()

        if storing_enabled:
            self._futures.append(self._executor.submit(self.store))
        else:
            self.logger.info("Storing disabled via SVC_KB_ENABLE_STORING (%s)", self.service_name)

        # Wait for completion and propagate any exceptions
        for future in self._futures:
            future.result()

        # Record the end of this run
        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                          RunStatus.RUN_ENDED, None, datetime.now())

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

        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                          RunStatus.INGESTION_STARTED, None, datetime.now())
        count = 0
        try:
            for item in self.fetch_from_source():
                if self._stop_event.is_set():
                    break
                self.emit_fetched_item(item)
                count += 1
        except Exception:
            self.logger.exception("Error during ingestion for %s", self.service_name)

        try:
            self.finalize_ingest()
            self._ingest_done.set()  # Signal that _ingest_done is finished
        except Exception:
            self.logger.exception("Error during finalize_ingest for: %s",
                                self.service_name)
        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                              RunStatus.INGESTION_COMPLETED,
                                                              {"count": count,
                                                                "msg": "Records Ingested"}, datetime.now())
        self.logger.info("Done ingestion for %s", self.service_name)

    def process(self) -> None:
        """Process ingested data. Keeps polling until producer is done and queue is empty."""

        self.logger.info("Processing ingested data from queue: %s. (%s)", self._ingest_queue_name(), self.service_name)

        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                              RunStatus.PROCESSING_STARTED, None, datetime.now())
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
            #Ingest done, AND check ingestion queue was empty this iteration
            return self._ingest_done.is_set() and not drained_any

        count = 0

        try:
            worker.run(
                queue_name=self._ingest_queue_name(),
                service_name=self.service_name,
                handler=handler,
                should_exit=should_exit
            )

            # flushes last batch that hangs in memory, due to not reaching batch size.
            if handler.item_list:
                handler.flush()

            count = worker.message_count
        except Exception:
            self.logger.exception("Error during processing for queue: %s. (%s)",
                            self._ingest_queue_name(), self.service_name)

        try:
            self.finalize_process()
            self._process_done.set() # Signal that _process_done is finished
        except Exception:
            self.logger.exception("Error during finalize_process for queue: %s. (%s)",
                                self._ingest_queue_name(), self.service_name)


        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                              RunStatus.PROCESSING_COMPLETED,
                                                              {"count": count,
                                                               "msg": "Messages Processed"}, datetime.now())
        self.logger.info("Done processing ingested data from queue: %s. (%s)", self._ingest_queue_name(),
                                                                                self.service_name)

    def store(self) -> None:
        """
            Process {service_name}.processed queue
            Inserts into database
        """
        self.logger.info("Storing processed data from queue: %s. (%s)", self._processed_queue_name(),
                                                                        self.service_name)
        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                          RunStatus.STORING_STARTED, None, datetime.now())

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
            # process is done, AND check processed queue was empty this iteration
            return self._ingest_done.is_set() and self._process_done.is_set() and not drained_any

        count = 0

        try:
            worker.run(
                queue_name=self._processed_queue_name(),
                service_name=self.service_name,
                handler=handler,
                should_exit=should_exit
            )
            count = worker.message_count
        except Exception:
            self.logger.exception("Error during storing for queue: %s. (%s)", self._processed_queue_name(),
                                                                            self.service_name)

        try:
            self.finalize_store()
        except Exception:
            self.logger.exception("Error during finalize_store for queue: %s. (%s)",
                                                                    self._processed_queue_name(), self.service_name)

        self._repository.insert_history_table_log(self._run_id, self.service_name,
                                                              RunStatus.STORING_COMPLETED,
                                                              {"count": count,
                                                               "msg": "Messages Stored"}, datetime.now())
        self.logger.info("Done storing processed data from queue: %s. (%s)", self._processed_queue_name(),
                                                                        self.service_name)

    def finalize_process(self) -> None:
        """Optional hook called after processing loop ends."""

    def finalize_ingest(self) -> None:
        """Optional hook called after ingest loop ends."""

    def finalize_store(self) -> None:
        """Optional hook called after store loop ends."""

    def _ack_message(self, delivery_tag, successful: bool):
        """Acknowledge or unack message back to queue"""
        if delivery_tag is not None:
            self.queue_service.read_ack(delivery_tag, successful=successful)

    def request_stop(self) -> None:
        """Stop event for knowledge process"""
        self.logger.info("Stop event requested")

        self._stop_event.set()

        # Cleanup workers and connections
        if self._executor:
            self.logger.info("Shutting down executor.")
            self._executor.shutdown(wait=True)

        self.logger.info("Propogating any exceptions during shutdown")
        for f in self._futures:
            try:
                f.result()
            except Exception as e:
                self.logger.exception("Worker error during shutdown: %s", e)

        self._executor = None
        self._futures = []
        self.logger.info("Cleaning up thread local queue connections and channels")
        self.queue_service.cleanup()

    def should_stop(self) -> bool:
        """Return true if and only if the internal flag is true."""
        return self._stop_event.is_set()
