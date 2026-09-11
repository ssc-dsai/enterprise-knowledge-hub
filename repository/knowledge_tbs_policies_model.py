"""Persistence model for TBS policies knowledge base."""
from __future__ import annotations

from dotenv import load_dotenv

from torch import Tensor
from peewee import SQL, IntegerField, TextField
import numpy as np

from repository.base_model import BaseEmbeddingModel, VectorField
from services.knowledge.tbs_policies.models import TBSPolicyItemProcessed

load_dotenv()

KB_TABLE_NAME = "kb_tbs_policies"


class KnowledgeBaseTBSPolicies(BaseEmbeddingModel):
    """kb_tbs_policies model"""
    page_id: int = IntegerField()
    chunk_index: int = IntegerField()
    name: str = TextField()
    content: str = TextField()
    embedding: list[float] = VectorField()
    embedding_dims: int = IntegerField()
    source: str | None = TextField(null=True)

    # Computed field. Not in table
    similarity: float | None

    class Meta:  # pylint: disable=too-few-public-methods
        """Configuration for the model"""
        db_table = KB_TABLE_NAME
        constraints = [
            SQL(
                'CONSTRAINT tbs_policies_page_id_source_chunk_index_dims_key '
                'UNIQUE (page_id, source, chunk_index, embedding_dims)'
            )
        ]

    @classmethod
    def from_item(cls, item: TBSPolicyItemProcessed) -> KnowledgeBaseTBSPolicies:
        """Build a record from a domain object, coercing embeddings to floats."""
        embedding = cls._to_floats(item.embeddings)
        return cls(
            page_id=item.page_id,
            chunk_index=item.chunk_index,
            name=item.name,
            content=item.content,
            last_modified_date=item.last_modified_date,
            embedding=embedding,
            embedding_dims=len(embedding),
            source=item.source,
        )

    def as_mapping(self) -> dict[str, object]:
        """Return a mapping compatible with psycopg executemany parameters."""
        return {
            "page_id": self.page_id,
            "chunk_index": self.chunk_index,
            "name": self.name,
            "content": self.content,
            "last_modified_date": self.last_modified_date,
            "embedding": self.embedding,
            "embedding_dims": self.embedding_dims,
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
