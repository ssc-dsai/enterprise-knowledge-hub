"""
Base class for queue providers.
"""
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass

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
    def read(self, queue_name: str) -> dict[str, object]:
        """Read from the specified queue."""
        raise NotImplementedError

    @abstractmethod
    def write(self, queue_name: str, message: dict[str, object]) -> None:
        """Write to the specified queue."""
        raise NotImplementedError
