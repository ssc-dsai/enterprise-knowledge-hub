"""Data models for knowledge items."""
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime


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

    def to_dict(self) -> dict[str, object]:
        """Convert to dictionary for queue serialization."""
        return {
            "name": self.name,
            "title": self.title,
            "content": self.content,
            "last_modified_date": self.last_modified_date.isoformat() if self.last_modified_date else None,
            "pid": self.pid,
        }
