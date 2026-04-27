"""Persistant layer models"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TypedDict
from datetime import datetime
from torch import Tensor
import numpy as np

from services.knowledge.wikipedia.models import WikipediaItemProcessed

# TO DO AR: i forgot why this was used TypedDict.  Probably an old implementation?  Not used many places.
# Check at some point to move to dataclass?
class DocumentRecord(TypedDict):
    """Model for records in kb_wikipedia table"""
    name: str
    content: str
    chunk_index: int
    similarity: float | None

@dataclass(slots=True)
class WikipediaDbRecord: #pylint: disable=too-many-instance-attributes
    """Serializable record for Postgres storage."""
    pid: int
    chunk_index: int
    name: str
    content: str
    last_modified_date: datetime | None
    embedding: list[float]
    source: str | None = None

    @classmethod
    def from_item(cls, item: WikipediaItemProcessed) -> WikipediaDbRecord:
        """Build a record from a domain object, coercing embeddings to floats."""
        embedding = cls._to_floats(item.embeddings)
        return cls(
            pid=item.pid,
            chunk_index=item.chunk_index,
            name=item.name,
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
