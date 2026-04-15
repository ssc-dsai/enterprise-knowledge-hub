"""Batch time tracker.  THink of better name"""
from datetime import datetime
import time

from services.database.run_history_service import RunHistoryService
from services.knowledge.models import RunStatus

class BatchTimeTracker:
    """Class to track and log average time per set amount of batch intervals/count"""
    start: float = None
    batch_start_time: float = None
    count: int
    interval: int
    history_service: RunHistoryService

    def __init__(self, interval: int, run_id: int, service_name: str, logger, history_service: RunHistoryService):
        self.interval = interval
        self.logger = logger
        self.count = 0
        self.history_service = history_service
        self.run_id = run_id
        self.service_name = service_name

    def tick(self):
        """
        tick to increment counter
        once we hit wanted interval.  logs average
        """

        self.count += 1
        if self.count % self.interval == 0:
            now = time.perf_counter()
            elapsed = now - self.start

            average_time_per_processing_batch = elapsed / self.interval

            meta_data: dict = {"average_time_per_processing_batch": average_time_per_processing_batch}
            self.history_service.insert_history_table_log(self.run_id, self.service_name, RunStatus.BATCH_AVERAGE_TIME, meta_data, datetime.now())
            self.start = now

    def start_timer(self) -> None:
        """start the timer"""
        self.start = time.perf_counter()

    def batch_start(self) -> None:
        """This is to reset start time per batch if we wanted to print it out per processing batch."""
        self.batch_start_time = time.perf_counter()

    def print_current_batch_time(self, knowledge_item_len: int, gpu_batch_size: int) -> None:
        """Print out time per processing batch"""
        batch_time = time.perf_counter() - self.batch_start_time
        self.logger.debug("Generated embeddings for %s items in %.2f seconds per batch (GPU batch size: %s)",
                             knowledge_item_len, (batch_time)/gpu_batch_size, gpu_batch_size)
