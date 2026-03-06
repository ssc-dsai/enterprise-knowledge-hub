"""
Base class for queue providers.
"""
from collections.abc import Iterator
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from services.knowledge.models import KnowledgeItem

@dataclass
class QueueProvider(ABC):
    """Abstract base class for queue configurations"""
    url: str #connection string for the queue system
    logger: logging.Logger

    @abstractmethod
    def close(self):
        """Close queue channel."""
        raise NotImplementedError

    @abstractmethod
    def read(self, queue_name: str) -> Iterator[tuple[dict[str, object], int]]:
        """Read from the specified queue."""
        raise NotImplementedError

    @abstractmethod
    def read_ack(self, delivery_tag: int, successful: bool = True) -> None:
        """Acknowledge or negatively acknowledge a message from the specified queue."""
        raise NotImplementedError

    @abstractmethod
    def write(self, queue_name: str, message: KnowledgeItem) -> None:
        """Write to the specified queue."""
        raise NotImplementedError
