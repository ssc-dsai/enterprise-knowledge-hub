"""Postgres/pgvector helper utilities."""
from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from typing import Sequence

import numpy as np
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg import sql
from psycopg_pool import ConnectionPool
from torch import Tensor

from services.knowledge.models import DatabaseWikipediaItem

load_dotenv()

@dataclass(slots=True)
class WikipediaDbRecord:
    """Serializable record for Postgres storage."""
    pid: int
    chunk_index: int
    name: str
    title: str
    content: str
    last_modified_date: datetime | None
    embedding: list[float]

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

    def insert_many(self, rows: Sequence[WikipediaDbRecord]) -> None:
        """Insert rows using executemany batching (no psycopg2 extras required)."""
        if not rows:
            return

        insert_sql = sql.SQL(
            """
            INSERT INTO {table} (pid, chunk_index, name, title, content, last_modified_date, embedding)
            VALUES (%(pid)s, %(chunk_index)s, %(name)s, %(title)s, %(content)s, %(last_modified_date)s, %(embedding)s)
            ON CONFLICT (pid, chunk_index) DO UPDATE SET
                name = EXCLUDED.name,
                title = EXCLUDED.title,
                content = EXCLUDED.content,
                last_modified_date = EXCLUDED.last_modified_date,
                embedding = EXCLUDED.embedding
            """
        ).format(table=sql.Identifier(self._table_name))

        params = [row.as_mapping() for row in rows]
        with self._pool.connection() as conn, conn.cursor() as cur:
            for i in range(0, len(params), self._batch_size):
                batch = params[i : i + self._batch_size]
                cur.executemany(insert_sql.as_string(conn), batch)
            conn.commit()

    def close(self) -> None:
        """Close the underlying connection pool."""
        if self._pool:
            self._pool.close()
