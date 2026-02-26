"""
Docstring for services.queue.queue_worker
"""
from logging import Logger
import time
from threading import Event
from dataclasses import dataclass

from services.queue.queue_service import QueueService

@dataclass
class QueueWorker:
    """
    Worker class encompasses polling from queues 
    """
    def __init__(self, queue_service, logger, stop_event, poll_interval=0.5):
        self.queue_service: QueueService = queue_service
        self.logger: Logger = logger
        self.stop_event: Event = stop_event
        self.poll_interval = poll_interval

    def run(self, service_name: str, queue_name: str, handler, should_exit):
        """
        Main function to poll from queues.

        :param self: Description
        :param service_name: name for specific service implementation running this
        :type service_name: str
        :param queue_name: name for queue
        :type queue_name: str
        :param handler: callable function.  processes item
        :param should_exit: callable function. exit condition for loop
        """
        while not self.stop_event.is_set():
            drained_any = False # to check if any messages read/drained this iteration

            for item, delivery_tag in self.queue_service.read(queue_name):
                drained_any = True
                try:
                    if self.stop_event.is_set():
                        self.logger.info("Stop event is true. Stopping process: %s - %s", service_name, queue_name)
                        self._acknowledge(delivery_tag, successful=False)
                        break
                    is_handler_manages_ack = handler(item, delivery_tag)
                    # if handler manages acknowledgement back to queue then ignore
                    if is_handler_manages_ack is False:
                        self._acknowledge(delivery_tag, successful=True)
                    else:
                        pass

                except Exception as e:
                    self.logger.exception(
                        "Error processing item in queue %s - %s", queue_name, service_name
                    )
                    self.logger.exception("Error: %s", e)
                    self._acknowledge(delivery_tag, successful=False)
            # Queue is empty - check if we should exit or wait
            if should_exit(drained_any):
                break
            time.sleep(self.poll_interval)

    def _acknowledge(self, delivery_tag, successful: bool):
        """Acknowledge message back to queue """
        if delivery_tag is not None:
            self.queue_service.read_ack(delivery_tag, successful=successful)
