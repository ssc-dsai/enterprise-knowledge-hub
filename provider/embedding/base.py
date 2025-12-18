from contextlib import contextmanager
import numpy as np
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

class EmbeddingBackendProvider(ABC):
    model_name: str
    device: str
    max_seq_length: int

    @abstractmethod
    def embed(text: List[str]) -> np.ndarray:
        raise NotImplementedError
