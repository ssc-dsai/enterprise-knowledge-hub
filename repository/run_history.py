"""Run history table repository"""
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

    def cronjob_insert_new_log(self, service_name: str, status: str,
                               metadata: dict | None, timestamp: datetime) -> datetime | None:
        """
        Queries the database for the most recent last_modified_date for a given source (e.g. wikipedia dump)

        Returns the most recent last_modified_date for the given source, or None if no records are found
        """
        query_sql = sql.SQL(
            """
            INSERT INTO {table} (service_name, status, metadata, timestamp)
            VALUES (%s, %s, %s, %s)
            """
        ).format(table=sql.Identifier(RUN_HISTORY_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (service_name, status, Json(metadata), timestamp))

    def cronjob_get_most_recent_dump_date(self, source: str) -> str | None:
        """
        Queries the database for the most recent last_modified_date for a given source (e.g. wikipedia dump)

        Returns the most recent last_modified_date for the given source, or none if no records are found
        """
        query_sql = sql.SQL(
            """
            SELECT metadata->>'dump_date' AS dump_date FROM {table}
            WHERE service_name = %s AND status = %s
            ORDER BY timestamp DESC
            LIMIT 1
            """
        ).format(table=sql.Identifier(RUN_HISTORY_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (source, "New Dump Link Detected and Downloaded"))
            row = cur.fetchone()

        if row and row[0]:
            return row[0]
        return None

    # change dictrow to a model
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
            cur.execute(query_sql, (run_id))
            row = cur.fetchone()

        return row
