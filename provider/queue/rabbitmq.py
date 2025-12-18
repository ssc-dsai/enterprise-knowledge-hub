from collections.abc import Iterator
from contextlib import contextmanager
import json
from typing import Any
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import UnroutableError, NackError

from provider.queue.base import QueueProvider

class RabbitMQProvider(QueueProvider):
    """RabbitMQ queue configuration provider"""

    _publish_properties = pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,
    )

    @contextmanager
    def channel(self) -> Iterator[BlockingChannel]:
        #self.logger.debug("Establishing RabbitMQ channel to %s", self.url)
        connection = pika.BlockingConnection(pika.URLParameters(self.url))
        try:
            channel = connection.channel()
            yield channel
        finally:
            channel.close()
            connection.close()

    def read(self, queue_name: str) -> Iterator[Any]:
        """Read all messages from the specified RabbitMQ queue."""
        with self.channel() as channel:
            channel.queue_declare(queue=queue_name, durable=True)
            while True:
                method_frame, header_frame, body = channel.basic_get(queue=queue_name, auto_ack=True)
                if method_frame is None:
                    break
                yield json.loads(body.decode('utf-8'))

    def write(self, queue_name: str, message: dict[str, object]) -> None:
        """Write to the specified RabbitMQ queue."""
        body = json.dumps(message).encode('utf-8')
        with self.channel() as channel:
            channel.queue_declare(queue=queue_name, durable=True)
            try:
                channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=body,
                    properties=self._publish_properties,
                )
            except UnroutableError as e:
                self.logger.error("Unroutable error: %s", e, exc_info=True)
            except NackError as e:
                self.logger.error("Nack error: %s", e, exc_info=True)
