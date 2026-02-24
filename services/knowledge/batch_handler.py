from typing import Callable, Generic, List, TypeVar, Tuple
from dataclasses import dataclass

T = TypeVar("T")

@dataclass
class BatchHandler(Generic[T]):
    def __init__(
        self, 
        process_batch: Callable[[List[T]], None], 
        acknowledge: Callable[[str, bool], None], 
        batch_size: int
    ):
        self.process_batch = process_batch
        self.acknowledge = acknowledge
        self.batch_size = batch_size
        self.item_list: List[Tuple[T, str]] = []

    def __call__(self, item: T, delivery_tag:str) -> None:
        self.item_list.append((item, delivery_tag))
        
        # don't process until we have the required batch_size
        if len(self.item_list) < self.batch_size:
            return

        items = [items for items, tags in self.item_list]
        tags = [tags for items, tags in self.item_list]
        self.item_list = []

        try:
            self.process_batch(items)
        except Exception:
            # exception messGE?
            for t in tags:
                self.acknowledge(t, False)
            raise
        else:
            for t in tags:
                self.acknowledge(t, True)
