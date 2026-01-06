"""
RabbitMQ queue configuration provider
"""
from collections.abc import Iterator
import json
import threading
from typing import Any
import pika
from pika.adapters.blocking_connection import BlockingChannel
from pika.exceptions import UnroutableError, NackError, AMQPConnectionError

from provider.queue.base import QueueProvider

class RabbitMQProvider(QueueProvider):
    """RabbitMQ queue configuration provider"""

    _publish_properties = pika.BasicProperties(
        content_type="application/json",
        delivery_mode=2,
    )

    def __init__(self, url: str, logger):
        super().__init__(url=url, logger=logger)
        self._local = threading.local()
        self._declared_queues: set[str] = set()
        self._lock = threading.Lock()

    def _get_connection(self) -> pika.BlockingConnection:
        """Get or create a thread-local connection."""
        conn = getattr(self._local, 'connection', None)
        if conn is None or conn.is_closed:
            self._local.connection = pika.BlockingConnection(pika.URLParameters(self.url))
            self._local.channel = None  # Reset channel when connection is new
        return self._local.connection

    def _get_channel(self) -> BlockingChannel:
        """Get or create a thread-local channel."""
        conn = self._get_connection()
        channel = getattr(self._local, 'channel', None)
        if channel is None or channel.is_closed:
            self._local.channel = conn.channel()
            self._local.declared_queues = set()  # Reset declared queues for new channel
        return self._local.channel

    def _ensure_queue_declared(self, channel: BlockingChannel, queue_name: str) -> None:
        """Declare queue only once per channel."""
        declared = getattr(self._local, 'declared_queues', set())
        if queue_name not in declared:
            channel.queue_declare(queue=queue_name, durable=True)
            declared.add(queue_name)
            self._local.declared_queues = declared

    def read(self, queue_name: str) -> Iterator[(Any, int)]:
        """Read all messages from the specified RabbitMQ queue using persistent connection."""
        channel = self._get_channel()
        self._ensure_queue_declared(channel, queue_name)

        while True:
            try:
                method_frame, _, body = channel.basic_get(queue=queue_name)
                if method_frame is None:
                    break
                yield json.loads(body.decode('utf-8')), method_frame.delivery_tag
            except AMQPConnectionError:
                # Reconnect and retry
                self.logger.warning("Connection lost during read, reconnecting...")
                channel = self._get_channel()
                self._ensure_queue_declared(channel, queue_name)

    def read_ack(self, delivery_tag, successful: bool = True):
        channel = self._get_channel()
        if successful:
            channel.basic_ack(delivery_tag=delivery_tag)
        else:
            channel.basic_nack(delivery_tag=delivery_tag, requeue=True)

    def write(self, queue_name: str, message: dict[str, object]) -> None:
        """Write to the specified RabbitMQ queue using persistent connection."""
        body = json.dumps(message).encode('utf-8')

        for attempt in range(3):
            try:
                channel = self._get_channel()
                self._ensure_queue_declared(channel, queue_name)
                channel.basic_publish(
                    exchange="",
                    routing_key=queue_name,
                    body=body,
                    properties=self._publish_properties,
                )
                return
            except (AMQPConnectionError, UnroutableError, NackError) as e:
                self.logger.warning("Publish failed (attempt %d): %s", attempt + 1, e)
                # Force reconnection on next attempt
                self._local.channel = None
                self._local.connection = None
                if attempt == 2:
                    raise

    def close(self) -> None:
        """Close the thread-local connection."""
        conn = getattr(self._local, 'connection', None)
        if conn and not conn.is_closed:
            conn.close()
        self._local.connection = None
        self._local.channel = None
