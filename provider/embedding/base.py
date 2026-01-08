"""Base interface for embedding backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

class EmbeddingBackendProvider(ABC):
    """Contract for embedding providers to implement."""
    model: Any
    model_name: str
    device: str
    max_seq_length: int

    @abstractmethod
    def embed(self, text: str) -> np.ndarray:
        """used to create embeddings for a text input"""
        raise NotImplementedError

    @abstractmethod
    def chunk_text_by_tokens(self, text: str, max_tokens: int = None, overlap_tokens: int = 200) -> list[str]:
        """Split text into chunks based on token count with overlap."""
        raise NotImplementedError
