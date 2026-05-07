"""Persistence models forknowledge base, defining the structure of records stored in the database and providing"""
from __future__ import annotations

from torch import Tensor
from peewee import SQL, IntegerField, TextField
import numpy as np

from repository.base_model import BaseEmbeddingModel, VectorField
from services.knowledge.wikipedia.models import WikipediaItemProcessed

KB_TABLE_NAME = "kb_wikipedia"

class KnowledgeBaseWikipedia(BaseEmbeddingModel): #pylint: disable=too-many-instance-attributes
    """kb_wikipedia model"""
    pid: int = IntegerField()
    chunk_index: int = IntegerField()
    name: str = TextField()
    content: str = TextField()
    embedding: list[float] = VectorField(dimensions=512)
    source: str | None = TextField(null=True)

    #Computed field.  Not in table
    similarity: float | None

    class Meta:  # pylint: disable=too-few-public-methods
        """Configuration for the model"""
        db_table = KB_TABLE_NAME
        constraints = [
            SQL(
                'CONSTRAINT documents_pid_source_chunk_index_key '
                'UNIQUE (pid, source, chunk_index)'
            )
        ]

    @classmethod
    def from_item(cls, item: WikipediaItemProcessed) -> KnowledgeBaseWikipedia:
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
