"""Data models for knowledge items."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum

from torch import Tensor
import numpy as np

class Source(StrEnum):
    """Enumeration of knowledge item sources."""
    WIKIPEDIA_EN = "enwiki"
    WIKIPEDIA_FR = "frwiki"
    #MYSSCPLUS = "mysscplus"


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
