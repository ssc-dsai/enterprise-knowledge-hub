"""Run history service class for run_history table"""
from dataclasses import dataclass
from datetime import datetime
import logging

from repository.run_history import RunHistoryRepository
from repository.run_history_model import RunHistory

@dataclass
class RunHistoryService():
    """Run history service class for run_history table"""

    logger: logging.Logger
    _repository: RunHistoryRepository

    def __init__(self, logger):
        self._logger = logger
        self._repository = RunHistoryRepository()

    def insert_history_table_log(self, run_id: int, service_name: str, status: str, metadata: dict | None,
                                 timestamp: datetime) -> RunHistory:
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Insert a log entry into the history table"""
        return self._repository.create(
                                run_id=run_id,
                                status=status,
                                service_name=service_name,
                                metadata=metadata,
                                timestamp=timestamp
                            )

    def cronjob_insert_new_log(self, service_name: str, status: str,
                               metadata: dict | None, timestamp: datetime) -> RunHistory:
        """Insert a log entry into the history table for cronjobs"""
        return self._repository.create(
                                run_id=None,
                                status=status,
                                service_name=service_name,
                                metadata=metadata,
                                timestamp=timestamp
                            )

    def run_history_table_rows(self) -> list[RunHistory]:
        """Get all history table rows"""
        return self._repository.list_all()

    def cronjob_get_most_recent_dump_date(self, source: str) -> RunHistory | None:
        """Get the most recent dump date for a given source"""
        status = "New Dump Link Detected and Downloaded"
        result = self._repository.get_by_source_and_status(source, status)
        return result.metadata.get('dump_date')
