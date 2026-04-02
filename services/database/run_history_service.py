"""Run history module"""
from dataclasses import dataclass
from datetime import datetime
import logging

from psycopg.rows import DictRow

from repository.pool_provider import PoolProvider
from repository.run_history import RunHistoryRepository


@dataclass
class RunHistoryService():
    """Run history service class for run_history table"""
    """Service to query wiki embeddings"""

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
