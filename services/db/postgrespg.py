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

from services.db.model import DocumentRecord
from services.knowledge.models import DatabaseWikipediaItem

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

        #  (If running local connected to DB use 172.16.123.217 instead of localhost)
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        dbname = os.getenv("POSTGRES_DB", "postgres")
        user = os.getenv("POSTGRES_USER", "postgres")
        # For running local connected to DB use "postconninfotgres" instead
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        table_name = os.getenv("WIKIPEDIA_TABLE", "documents")
        pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        batch_size = int(os.getenv("POSTGRES_BATCH_SIZE", "500"))
        conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        # print('==================conn info')
        # print(conninfo)

        return cls(conninfo=conninfo, table_name=table_name, pool_size=pool_size, batch_size=batch_size)

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

    def get_record_content(self, title: str) -> list[DocumentRecord]:
        """Query specific record content based on title"""

        query_sql = sql.SQL(
            """
            SELECT name, title, content FROM {table}
            WHERE name LIKE %s
            """
        ).format(table=sql.Identifier(self._table_name))

        pattern=f"%{title}%"

        with self._pool.connection() as conn, conn.cursor(row_factory=dict_row) as cur:
            cur.execute(query_sql, (pattern,))
            rows = cur.fetchall()

        return rows

    def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool:
            self._pool.close()
