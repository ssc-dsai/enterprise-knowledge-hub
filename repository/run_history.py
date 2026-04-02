from datetime import datetime
from psycopg import sql
from psycopg.types.json import Json
from psycopg.rows import DictRow, dict_row

from repository.base import BaseRepository

RUN_HISTORY_TABLE_NAME = "run_history"

class RunHistoryRepository(BaseRepository):
    """Repository for run_history table."""
        
    def run_history_table_rows(self) -> list[DictRow]:
        """Query all rows from the run_history table for debugging/observability purposes."""

        query_sql = sql.SQL(
                """
                SELECT * FROM {table} ORDER BY id DESC;
                """
            ).format(table=sql.Identifier(RUN_HISTORY_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_sql)
            rows = cur.fetchall()
        return rows

    def insert_history_table_log(self, run_id: int, service_name: str, status: str, metadata: dict | None,
                                 timestamp: datetime):
        # pylint: disable=too-many-arguments
        # pylint: disable=too-many-positional-arguments
        """Insert a log entry into the history table"""

        query_sql = sql.SQL(
            """
            INSERT INTO {table} (run_id, service_name, status, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            """
        ).format(table=sql.Identifier(RUN_HISTORY_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (run_id, service_name, status, Json(metadata), timestamp))
            conn.commit()