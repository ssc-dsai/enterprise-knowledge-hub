"""Data models for knowledge items."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, Optional, Union, Literal
import base64
from enum import StrEnum

import torch
import numpy as np

Tensor = torch.Tensor

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

class Source(StrEnum):
    """Enumeration of knowledge item sources."""
    WIKIPEDIA_EN = "enwiki"
    WIKIPEDIA_FR = "frwiki"
    #MYSSCPLUS = "mysscplus


@dataclass
class KnowledgeItem(ABC):
    """Base class for knowledge items that will be pushed to the queue."""

    name: str = field(default="")  # Unique identifier for the knowledge item

    @abstractmethod
    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for queue serialization."""
        raise NotImplementedError


@dataclass
class WikipediaItem(KnowledgeItem):
    """Knowledge item representing a Wikipedia page."""
    title: str = field(default="")
    content: str = field(default="")  # Wiki markup content
    last_modified_date: datetime | None = field(default=None)
    pid: int = field(default=0)  # Page ID
    source: Source | None = field(default=Source.WIKIPEDIA_EN)
    chunk_index: int = field(default=1)
    chunk_count: int = field(default=1)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "WikipediaItem":
        """Create WikipediaItem from dictionary (queue deserialization)."""
        data = data.copy()  # Don't mutate the input
        if data.get("source"):
            data["source"] = Source(data["source"])
        if data.get("last_modified_date"):
            data["last_modified_date"] = datetime.fromisoformat(data["last_modified_date"])
        return cls(**data)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for queue serialization."""
        return {
            "name": self.name,
            "title": self.title,
            "content": self.content,
            "last_modified_date": self.last_modified_date.isoformat() if self.last_modified_date else None,
            "pid": self.pid,
            "chunk_index": self.chunk_index,
            "chunk_count": self.chunk_count,
            "source": self.source.value,
        }

@dataclass
class DatabaseWikipediaItem(WikipediaItem):
    """Knowledge item representing a Wikipedia page stored in a database."""
    embeddings: np.ndarray | Tensor | None = field(default=None)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for queue serialization."""
        result = super().to_dict()

        result.update({
            "embeddings": _encode_embeddings(self.embeddings)
        })
        return result

    @classmethod
    def from_rabbitqueue_dict(cls, data: Dict[str, Any]) -> "DatabaseWikipediaItem":
        """Build back from RabbitMQ message dict"""
        data = data.copy()
        if data.get("embeddings"):
            data["embeddings"] = _decode_embeddings(data["embeddings"])
        return cls(**data)
