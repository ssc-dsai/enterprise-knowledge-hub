"""Base interface for embedding backends."""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from provider.embedding.tokenizer import ThreadTokenizer

# Qwen3 embedding models require instruction prefixes for queries
# See: https://huggingface.co/Qwen/Qwen3-Embedding-0.6B
QWEN3_QUERY_INSTRUCTION = "Instruct: Given a query, retrieve relevant Wikipedia passages that answer the query\nQuery: "


class EmbeddingBackendProvider(ABC):
    """Contract for embedding providers to implement."""
    model: Any
    model_name: str
    device: str
    max_seq_length: int
    dimensions: int
    tokenizer: ThreadTokenizer | None = None
    batch_size: int = 32

    @abstractmethod
    def embed(self, text: Any, is_query: bool = False) -> np.ndarray:
        """Generate embeddings for text.

        Args:
            text: The text to embed.
            is_query: If True, prepend query instruction for asymmetric retrieval.
                     Documents should be embedded with is_query=False.
                     Queries should be embedded with is_query=True.
        """
        raise NotImplementedError

    @abstractmethod
    def chunk_text_by_tokens(self, text: str, max_tokens: int = None, overlap_tokens: int = 10) -> list[str]:
        """Split text into chunks based on token count with overlap."""
        raise NotImplementedError

    def is_model_loaded(self) -> bool:
        """Returns true is model is loaded"""
        return getattr(self, "model", None) is not None

    def get_tokenizer(self) -> Any:
        """
        Get tokenizer for current thread
        """
        return self.tokenizer.get()
    
    def get_batch_size(self) -> int:
        """
        Get batch size for embedding generation, which may be dynamic based on model/device
        """
        return self.batch_size
