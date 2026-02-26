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
from psycopg_pool import ConnectionPool
from torch import Tensor

from repository.model import DocumentRecord
from services.knowledge.models import DatabaseWikipediaItem

load_dotenv()

# establish connection pool at module level to be shared across repository instances
host = os.getenv("POSTGRES_HOST", "localhost")
port = int(os.getenv("POSTGRES_PORT", "5432"))
dbname = os.getenv("POSTGRES_DB", "postgres")
user = os.getenv("POSTGRES_USER", "postgres")
password = os.getenv("POSTGRES_PASSWORD", "postgres")
pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
batch_size = int(os.getenv("POSTGRES_BATCH_SIZE", "500"))
conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

pool = ConnectionPool(conninfo = conninfo,
                      min_size=1,
                      max_size=pool_size,
                      open=False,
                      configure=register_vector)
pool.open()

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
    def from_item(cls, item: DatabaseWikipediaItem) -> "WikipediaDbRecord":
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

# Repository for postgres interaction with the documents table, which stores the ingested and processed Wikipedia records along with their embeddings
class WikipediaPgRepository:
    """Lightweight repository to write Wikipedia records into Postgres/pgvector."""
    TABLE_NAME = "documents"

    def __init__(self ) -> None:
        """Initialize the repository and open a connection pool."""
        self._pool = pool

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
        ).format(table=sql.Identifier(self.TABLE_NAME))
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
        ).format(table=sql.Identifier(self.TABLE_NAME))

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
        ).format(table=sql.Identifier(self.TABLE_NAME))

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
        ).format(table=sql.Identifier(self.TABLE_NAME))

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
        ).format(table=sql.Identifier(self.TABLE_NAME))

        pattern=pid

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_sql, (pattern,))
            rows = cur.fetchall()
        return rows

    def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool:
            self._pool.close()

# For postgres interaction with the run_history table, which tracks ingestion and processing runs for observability and debugging purposes
class RunHistoryPGRepository:
    """Lightweight repository to run_history records into postgres."""
    TABLE_NAME = "run_history"

    def __init__(self ) -> None:
        """Initialize the repository and open a connection pool."""
        self._pool = pool

    def run_history_table_rows(self):
        query_sql = sql.SQL(
                """
                SELECT * FROM run_history ORDER BY id DESC;
                """
            )

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_sql)
            rows = cur.fetchall()
        return rows

    def update_history_table_start(self, start_time, service_name: str, status: str = "started", process_running: bool = True, ingest_running: bool = True) -> int:
        """Update the history table on run start"""

        query_sql = sql.SQL(
            """
            INSERT INTO run_history (start_time, service_name, status, process_running, ingest_running)
            VALUES (%s, %s, %s, %s, %s)
            RETURNING id
            """
        )

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (start_time, service_name, status, process_running, ingest_running))
            result = cur.fetchone()[0]
            conn.commit()
            return result

    def update_history_table_end(self, status, end_time, id: int, process_running: bool = False, ingest_running: bool = False) -> None:
        """Update the history table on run end"""

        query_sql = sql.SQL(
            """
            UPDATE run_history
            SET end_time = %s, status = %s, process_running = %s, ingest_running = %s
            WHERE id = %s
            """
        )

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (end_time, status, process_running, ingest_running, id))
            conn.commit()

    def update_process_step_end(self, id: int, process_running: bool = False) -> None:
        """Update the history table's process column on processing step end"""

        query_sql = sql.SQL(
            """
            UPDATE run_history
            SET process_running = %s
            WHERE id = %s
            """
        )

        with self._pool.connection() as conn, conn.cursor() as cur:
            cur.execute(query_sql, (process_running, id))
            conn.commit()

    def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool:
            self._pool.close()