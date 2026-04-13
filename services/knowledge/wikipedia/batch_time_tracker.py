"""Batch time tracker.  THink of better name"""
from datetime import datetime
import time

from services.database.run_history_service import RunHistoryService

class BatchTimeTracker:
    """Class to track and log average time per set amount of batch intervals/count"""
    def __init__(self, interval: int, run_id: int, service_name: str, logger, history_service: RunHistoryService):
        self.interval = interval
        self.logger = logger
        self.count = 0
        self.history_service = history_service
        self.start: float = 0
        self.run_id = run_id
        self.service_name = service_name
    
    def tick(self):
        """
        tick to increment counter
        once we hit wanted interval.  logs average
        """
        if self.start is None:
            # place holder, check if raise an exception, it wont stop run.
            self.logger.info("Start time not set") 
            return
        
        self.count += 1
        if self.count % self.interval == 0:
            now = time.perf_counter()
            elapsed = now - self.start
            
            average_time_per_batch = elapsed / self.interval
            
            meta_data: dict = {"average_time_per_batch": average_time_per_batch}
            self.history_service.insert_history_table_log(self.run_id, self.service_name, "Logging average", meta_data, datetime.now())
            self.start = now
            
    def start_timer(self) -> None:
        """start the timer"""
        self.start = time.perf_counter()