"""Queue service to handle queue operations."""
from collections.abc import Iterator
from dataclasses import dataclass
import logging
from provider.queue.base import QueueProvider
from services.knowledge.models import KnowledgeItem

@dataclass
class QueueService:
    """Service to manage queue operations."""
    queue_provider: QueueProvider
    logger: logging.Logger

    def read(self, queue_name: str) -> Iterator[tuple[dict[str, object], int]]:
        """Read messages from the specified queue."""
        return self.queue_provider.read(queue_name)

    def read_ack(self, delivery_tag: int, successful: bool = True) -> None:
        """Acknowledge or negatively acknowledge a message from the specified queue."""
        return self.queue_provider.read_ack(delivery_tag, successful)

    def write(self, queue_name: str, message: KnowledgeItem) -> None:
        """Write a message to the specified queue."""
        return self.queue_provider.write(queue_name, message)
