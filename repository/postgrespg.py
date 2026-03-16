"""Postgres/pgvector helper utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg.rows import dict_row
from psycopg import sql
from psycopg.types.json import Json
from psycopg_pool import ConnectionPool
from torch import Tensor

from repository.model import DocumentRecord
from services.knowledge.wikipedia.models import WikipediaItemProcessed

load_dotenv()

@dataclass(slots=True)
class WikipediaDbRecord: #pylint: disable=too-many-instance-attributes
    """Serializable record for Postgres storage."""
    pid: int
    chunk_index: int
    name: str
    title: str
    content: str
    last_modified_date: datetime | None
    embedding: list[float]
    source: str | None = None

    @classmethod
    def from_item(cls, item: WikipediaItemProcessed) -> "WikipediaDbRecord":
        """Build a record from a domain object, coercing embeddings to floats."""
        embedding = cls._to_floats(item.embeddings)
        return cls(
            pid=item.pid,
            chunk_index=item.chunk_index,
            name=item.name,
            title=item.title,
            content=item.content,
            last_modified_date=item.last_modified_date,
            embedding=embedding,
            source=item.source,
        )

    def as_mapping(self) -> dict[str, object]:
        """Return a mapping compatible with psycopg executemany parameters."""
        return {
            "pid": self.pid,
            "chunk_index": self.chunk_index,
            "name": self.name,
            "title": self.title,
            "content": self.content,
            "last_modified_date": self.last_modified_date,
            "embedding": self.embedding,
            "source": self.source,
        }

    @staticmethod
    def _to_floats(raw_embedding: object) -> list[float]:
        if raw_embedding is None:
            raise ValueError("Embeddings are required for storage.")
        if isinstance(raw_embedding, Tensor):
            return raw_embedding.detach().cpu().flatten().tolist()
        if isinstance(raw_embedding, np.ndarray):
            return raw_embedding.flatten().tolist()
        if isinstance(raw_embedding, (list, tuple)):
            return [float(x) for x in raw_embedding]
        raise TypeError(f"Unsupported embedding type: {type(raw_embedding)!r}")


class WikipediaPgRepository:
    """Lightweight repository to write Wikipedia records into Postgres/pgvector."""

    def __init__(
        self,
        conninfo: str,
        table_name: str = "wikipedia_pages",
        pool_size: int = 5,
        batch_size: int = 500,
    ) -> None:
        """Initialize the repository and open a connection pool."""
        self._table_name = table_name
        self._batch_size = batch_size
        self._pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=pool_size,
            open=False,
            configure=register_vector,
        )
        self._pool.open()

    @classmethod
    def from_env(cls) -> "WikipediaPgRepository":
        """
        Docstring for from_env


        :param cls: Description
        :return: Description
        :rtype: WikipediaPgRepository
        """

        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        dbname = os.getenv("POSTGRES_DB", "postgres")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        table_name = os.getenv("WIKIPEDIA_TABLE", "documents")
        pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        batch_size = int(os.getenv("POSTGRES_BATCH_SIZE", "500"))
        conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        return cls(conninfo=conninfo, table_name=table_name, pool_size=pool_size, batch_size=batch_size)

    def insert(self, row: WikipediaDbRecord) -> None:
        """Insert row"""
        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (pid, chunk_index, name, title, content, last_modified_date, embedding, source)
            VALUES (%(pid)s, %(chunk_index)s, %(name)s, %(title)s, %(content)s,
                %(last_modified_date)s, %(embedding)s, %(source)s)
            ON CONFLICT (pid, chunk_index) DO UPDATE SET
                name = EXCLUDED.name,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                last_modified_date = EXCLUDED.last_modified_date,
                embedding = EXCLUDED.embedding,
                source = EXCLUDED.source
            """
        ).format(table=sql.Identifier(self._table_name))
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
            ON CONFLICT (pid, chunk_index) DO UPDATE SET
                name = EXCLUDED.name,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                last_modified_date = EXCLUDED.last_modified_date,
                embedding = EXCLUDED.embedding,
                source = EXCLUDED.source
            """
        ).format(table=sql.Identifier(self._table_name))

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
        ).format(table=sql.Identifier(self._table_name))

        with self._pool.connection() as conn:
            with conn.transaction():
                with conn.cursor() as cur:
                    cur.execute(set_probes_sql)
                    cur.execute(query_sql, (embedding_vector, embedding_vector, limit))
                    rows = cur.fetchall()
        return rows

    def get_pid_by_title(self, title: str) -> int | None:
        """Query for pid based on title"""

        query_sql = sql.SQL(
            """
            SELECT pid FROM {table}
            WHERE name = %s
            """
        ).format(table=sql.Identifier(self._table_name))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (title,))
            row = cur.fetchone()

        if row:
            return row[0]
        return None

    def get_record_full_chunks_content(self, pid: int) -> list[DocumentRecord]:
        """Query for entire record chunks based on title"""

        query_sql = sql.SQL(
            """
            SELECT title, content FROM {table}
            WHERE pid = %s
            """
        ).format(table=sql.Identifier(self._table_name))

        pattern=pid

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_sql, (pattern,))
            rows = cur.fetchall()
        return rows

    def run_history_table_rows(self):
        """Query all rows from the run_history table for debugging/observability purposes."""

        query_sql = sql.SQL(
                """
                SELECT * FROM run_history ORDER BY id DESC;
                """
            )

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
            INSERT INTO run_history (run_id, service_name, status, metadata, timestamp)
            VALUES (%s, %s, %s, %s, %s)
            """
        )

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (run_id, service_name, status, Json(metadata), timestamp))
            conn.commit()

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
        ).format(table=sql.Identifier(self._table_name))

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (pid, source, last_date_modified))
            row = cur.fetchone()

        if row:
            return True # exists and is up to date
        return False # either doesn't exist or is outdated
