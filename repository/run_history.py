"""Run history table repository"""

from repository.base import BaseRepository
from repository.run_history_model import RunHistory

class RunHistoryRepository(BaseRepository):
    """Repository for run_history table."""

    def __init__(self):
        super().__init__(RunHistory)

    def get_by_source_and_status(self, source: str, status: str) -> RunHistory | None:
        """
            Return first based on source and status
        """
        query = (self.model
                .select()
                .where(
                    (self.model.service_name == source) &
                    (self.model.status == status)
                )
                .order_by(self.model.timestamp.desc())
                .get_or_none())
        return query

    def select_first_instance_of_run_id(self, run_id: int) -> RunHistory | None:
        """
        Returns the first record that contains run_id
        """
        query = (self.model
                .select()
                .where(
                    self.model.run_id == run_id
                )
                .get_or_none())

        return query
