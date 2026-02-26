"""BatchHandler to enable the ability to handle multiple items at once vs processing one item at a time"""
from typing import Callable, List, Tuple
from dataclasses import dataclass
from logging import Logger

from services.knowledge.models import KnowledgeItem

@dataclass
class BatchHandler:
    """Handles multiple queue items at once."""
    def __init__(
        self,
        process_batch: Callable[[List[KnowledgeItem]], None],
        acknowledge: Callable[[str, bool], None],
        batch_size: int,
        logger: Logger
    ):
        self.process_batch = process_batch
        self.acknowledge = acknowledge
        self.batch_size = batch_size
        self.item_list: List[Tuple[KnowledgeItem, str]] = []
        self.logger: Logger = logger

    def __call__(self, item: KnowledgeItem, delivery_tag:str) -> None:
        """The method that runs after init, when invoked like:
            handler = BatchHandler(self.process_item, acknowledge, batch_size, self.logger)
            handler here being a callback
        """
        self.item_list.append((item, delivery_tag))

        # don't process until we have the required batch_size
        if len(self.item_list) < self.batch_size:
            return

        items = [items for items, tags in self.item_list]
        tags = [tags for items, tags in self.item_list]
        self.item_list = []

        try:
            self.process_batch(items)
            for t in tags:
                self.acknowledge(t, True)
        except Exception:
            for t in tags:
                self.acknowledge(t, False)
            raise
