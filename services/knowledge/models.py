"""Data models for knowledge items."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime

from torch import Tensor


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
        }

@dataclass
class DatabaseWikipediaItem(WikipediaItem):
    """Knowledge item representing a Wikipedia page stored in a database."""
    embeddings: Tensor | None = field(default=None)

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for queue serialization."""
        base_dict = super().to_dict()
        base_dict.update({"embeddings": self.embeddings.__dict__ if self.embeddings is not None else None})
        return base_dict
