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

    def select_first_instance_of_run_id(self, run_id: int) -> DictRow | None:
        """
        Returns the first record that contains run_id
        """
        query_sql=sql.SQL(
            """
            SELECT run_id FROM {table}
            WHERE run_id = %s
            limit 1
            """
        ).format(table=sql.Identifier(RUN_HISTORY_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, [run_id])
            row = cur.fetchone()

        return row
