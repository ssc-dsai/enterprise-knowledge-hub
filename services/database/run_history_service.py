"""Run history service class for run_history table"""
from dataclasses import dataclass
from datetime import datetime
import logging

from psycopg.rows import DictRow

from repository.pool_provider import PoolProvider
from repository.run_history import RunHistoryRepository


@dataclass
class RunHistoryService():
    """Run history service class for run_history table"""

    logger: logging.Logger
    _repository: RunHistoryRepository

    def __init__(self, logger):
        self._logger = logger
        pool = PoolProvider.get_pool()
        self._repository = RunHistoryRepository(pool)

    def insert_history_table_log(self, run_id: int, service_name: str, status: str, metadata: dict | None,
                                 timestamp: datetime) -> None:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Insert a log entry into the history table"""
        self._repository.insert_history_table_log(run_id, service_name, status, metadata, timestamp)

    def run_history_table_rows(self) -> list[DictRow]:
        """Get all history table rows"""
        return self._repository.run_history_table_rows()

    def cronjob_insert_new_log(self, service_name: str, status: str,
                               metadata: dict | None, timestamp: datetime) -> None:
        """Insert a log entry into the history table for cronjobs"""
        self._repository.cronjob_insert_new_log(service_name, status, metadata, timestamp)

    def cronjob_get_most_recent_dump_date(self, source: str) -> str | None:
        """Get the most recent dump date for a given source"""
        return self._repository.cronjob_get_most_recent_dump_date(source)

    def select_first_instance_of_run_id(self, run_id: int) -> DictRow:
        """Get the first record with run_id"""
        return self._repository.select_first_instance_of_run_id(run_id)
