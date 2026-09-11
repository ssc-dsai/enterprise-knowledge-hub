"""Data models for TBS Policy items."""
from datetime import datetime

from pydantic import ConfigDict, field_serializer, field_validator
import numpy as np
import torch

from services.knowledge.models import KnowledgeItem, encode_embeddings, decode_embeddings

Tensor = torch.Tensor


class TBSPolicyItemRaw(KnowledgeItem):
    """Knowledge item representing a single TBS policy page."""
    content: str = ""
    page_id: int = 0
    source: str = "tbs-policies"
    last_modified_date: datetime | None = None
    chunk_index: int = 1
    chunk_count: int = 1


class TBSPolicyItemProcessed(TBSPolicyItemRaw):
    """TBS policy item with computed embeddings."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: np.ndarray | Tensor | None = None
    embedding_dims: int = 0

    @field_serializer("embeddings")
    def serialize_embeddings(self, value):
        """Custom serializer for embeddings prop."""
        return encode_embeddings(value)

    @field_validator("embeddings", mode="before")
    @classmethod
    def _val_embedding(cls, value):
        if value is None or isinstance(value, (np.ndarray, Tensor)):
            return value
        if isinstance(value, dict):
            return decode_embeddings(value)
        raise TypeError(f"Invalid embedding value type: {type(value)!r}")
