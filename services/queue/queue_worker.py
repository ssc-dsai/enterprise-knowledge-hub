"""
Docstring for services.queue.queue_worker
"""
import time
from dataclasses import dataclass

@dataclass
class QueueWorker:
    """
    Docstring for QueueWorker
    """
    def __init__(self, queue_service, logger, stop_event, poll_interval=0.5):
        self.queue_service = queue_service
        self.logger = logger
        self.stop_event = stop_event
        self.poll_interval = poll_interval

    def run(self, service_name: str, queue_name: str, handler, should_exit):
        """
        Docstring for run
        
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
                    handler(item)
                    self._acknowledge(delivery_tag, successful=True)
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
        if delivery_tag is not None:
            self.queue_service.read_ack(delivery_tag, successful=successful)
