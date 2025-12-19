"""
Embedding Provider base
"""
from abc import ABC, abstractmethod
from typing import List
import numpy as np

class EmbeddingBackendProvider(ABC):
    """
    Embedding Provider base
    """
    model_name: str
    device: str
    max_seq_length: int

    def set_device(self, device: str):
        """
        Set device type
        """
        raise NotImplementedError

    @abstractmethod
    def embed(self, text: List[str]) -> np.ndarray:
        """
        embedding abstract method
        """
        raise NotImplementedError
