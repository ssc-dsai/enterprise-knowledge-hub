"""Data models for Wikipedia items."""
from datetime import datetime
from typing import Any, Dict, Optional, Union, Literal
import base64
from enum import StrEnum
from pydantic import ConfigDict, field_serializer, field_validator

import torch
import numpy as np

from services.knowledge.models import KnowledgeItem

Tensor = torch.Tensor

class Source(StrEnum):
    """Enumeration of knowledge item sources."""
    WIKIPEDIA_EN = "enwiki"
    WIKIPEDIA_FR = "frwiki"
    #MYSSCPLUS = "mysscplus

def _encode_embeddings(embedding: Union[np.ndarray, Tensor, None]) -> Optional[Dict[str, Any]]:
    """Convert embeddings into a JSON-serializable dict"""
    if embedding is None:
        return None

    if isinstance(embedding, Tensor):
        # Move to CPU + detach, then convert to numpy for consistent encoding
        embedding_np = embedding.detach().to("cpu").numpy()
        kind: Literal["torch"] = "torch"
    elif isinstance(embedding, np.ndarray):
        embedding_np = embedding
        kind = "numpy"
    else:
        raise TypeError(f"Unsupported embeddings type: {type(embedding)!r}")

    # Ensure contiguous so .tobytes() matches shape/dtype correctly
    embedding_np = np.ascontiguousarray(embedding_np)
    raw = embedding_np.tobytes(order="C")
    data_b64 = base64.b64encode(raw).decode("ascii")

    return {
        "kind": kind,  # tells you what to rebuild as (numpy vs torch)
        "dtype": str(embedding_np.dtype),  # e.g. "float32"
        "shape": list(embedding_np.shape),
        "data_b64": data_b64,
    }

def _decode_embeddings(payload: Optional[Dict[str, Any]]) -> Union[np.ndarray, Tensor, None]:
    """Reverse of _encode_embeddings"""
    if payload is None:
        return None

    kind = payload["kind"]
    dtype = np.dtype(payload["dtype"])
    shape = tuple(payload["shape"])
    raw = base64.b64decode(payload["data_b64"].encode("ascii"))

    arr = np.frombuffer(raw, dtype=dtype).reshape(shape)

    if kind == "numpy":
        return arr

    if kind == "torch":
        return torch.from_numpy(arr)

    raise ValueError(f"Unknown embeddings kind: {kind!r}")

class WikipediaItemRaw(KnowledgeItem):
    """Knowledge item representing a Wikipedia page."""
    title: str = ""
    content: str = ""  # Wiki markup content
    last_modified_date: datetime | None = None
    pid: int = 0
    source: Source | None = Source.WIKIPEDIA_EN
    chunk_index: int = 1
    chunk_count: int = 1

class WikipediaItemProcessed(WikipediaItemRaw):
    """Knowledge item representing a Wikipedia page stored in a database."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    embeddings: np.ndarray | Tensor | None = None

    @field_serializer("embeddings")
    def serialize_embeddings(self, value):
        """custom serializer for embeddings prop"""
        return _encode_embeddings(value)

    @field_validator("embeddings", mode="before")
    @classmethod
    def _val_embedding(cls, value):
        if value is None or isinstance(value, (np.ndarray, Tensor)):
            return value
        if isinstance(value, dict):
            return _decode_embeddings(value)
        raise TypeError(f"Invalid embedding value type: {type(value)!r}")
