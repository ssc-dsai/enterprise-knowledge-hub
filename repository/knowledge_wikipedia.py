"""Postgres/pgvector helper utilities."""
from __future__ import annotations

from datetime import datetime
from typing import Sequence

import numpy as np
from psycopg.rows import dict_row
from psycopg import sql

from repository.connection_pool import ConnectionPoolPG
from repository.model import DocumentRecord, WikipediaDbRecord


VECTOR_TABLE_NAME = "documents"

class KnowledgeWikipedia:
    """Repository to write Wikipedia records into Postgres/pgvector."""

    _pool: ConnectionPoolPG
    
    def __init__(
        self,
        pool: ConnectionPoolPG
    ) -> None:
        self._pool = pool


    def insert(self, row: WikipediaDbRecord) -> None:
        """Insert row"""
        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (pid, chunk_index, name, title, content, last_modified_date, embedding, source)
            VALUES (%(pid)s, %(chunk_index)s, %(name)s, %(title)s, %(content)s,
                %(last_modified_date)s, %(embedding)s, %(source)s)
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))
        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(insert_sql, (row))
            conn.commit()

    def insert_many(self, rows: Sequence[WikipediaDbRecord]) -> None:
        """Insert rows using executemany batching (no psycopg2 extras required)."""
        if not rows:
            return

        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (pid, chunk_index, name, title, content, last_modified_date, embedding, source)
            VALUES (%(pid)s, %(chunk_index)s, %(name)s, %(title)s, %(content)s, %(last_modified_date)s, %(embedding)s, %(source)s) #pylint: disable=line-too-long
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))

        params = [row.as_mapping() for row in rows]
        with self._pool.connection() as conn, conn.cursor() as cur:
            for i in range(0, len(params), self._batch_size):
                batch = params[i : i + self._batch_size]
                cur.executemany(insert_sql.as_string(conn), batch)
            conn.commit()

    def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 100,
        probes: int = 100,
    ) -> list[dict]:
        """Search for similar embeddings using pgvector's <=> operator.

        Args:
            embedding: The query embedding vector.
            limit: Maximum number of results to return (acts as a safety cap).
            probes: Number of IVFFlat lists to search. Higher = better recall but slower.
                    With 3464 lists, recommended range is 60-350 (sqrt(lists) to lists/10).
        """
        embedding_vector = embedding[0] if isinstance(embedding[0], (list, tuple, np.ndarray)) else embedding

        # SET doesn't support parameterized values, so format directly (int is safe)
        set_probes_sql = sql.SQL("SET LOCAL ivfflat.probes = {}").format(sql.Literal(probes))

        # Use a larger limit for index scan, then filter by similarity threshold
        # The WHERE clause filters after the index scan finds candidates
        query_sql = sql.SQL(
            """
            SELECT name, content, chunk_index, 1 - (embedding <=> %s::vector) AS similarity
            FROM {table}
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))

        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(set_probes_sql)
                    cur.execute(query_sql, (embedding_vector, embedding_vector, limit))
                    rows = cur.fetchall()
        return rows

    def get_pid_by_title(self, title: str, source: str) -> int | None:
        """Query for pid based on title"""

        query_sql = sql.SQL(
            """
            SELECT pid FROM {table}
            WHERE name = %s
            AND source = %s
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (title, source))
            row = cur.fetchone()

        if row:
            return row[0]
        return None

    def get_record_full_chunks_content(self, pid: int, source: str) -> list[DocumentRecord]:
        """Query for entire record chunks based on title"""

        query_sql = sql.SQL(
            """
            SELECT title, content FROM {table}
            WHERE pid = %s
            AND source = %s
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))

        pattern=pid

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_sql, (pattern, source))
            rows = cur.fetchall()
        return rows

    def record_is_up_to_date(self, pid: int, source: str, last_date_modified: datetime):
        """
        Queries the database for the documents with pid and checks if the date is currently
        more recent than the one in the database

        --- Query needs to return a record (it needs to exists) AND make sure current db date is 
            greater or equal to current passed date

        Returns True if the record EXISTS AND is UP TO DATE, False otherwise
        """
        query_sql = sql.SQL(
            """
            SELECT pid FROM {table}
            WHERE pid = %s and source LIKE %s and last_modified_date >= %s
            LIMIT 1
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (pid, source, last_date_modified))
            row = cur.fetchone()

        if row:
            return True # exists and is up to date
        return False # either doesn't exist or is outdated

    def delete_by_pid_source(self, pid: int, source: str) -> None:
        """Delete chunks for a given pid and source"""
        query_sql = sql.SQL(
            """
            DELETE FROM {table}
            WHERE pid = %s 
            AND source = %s
            """
        ).format(table=sql.Identifier(VECTOR_TABLE_NAME))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (pid, source))
            conn.commit()
