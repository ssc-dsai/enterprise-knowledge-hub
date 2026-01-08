"""Qwen3 embeddings using llama.cpp backend."""

# pylint: disable=duplicate-code
import logging
import os

import numpy as np
import torch
from dotenv import load_dotenv
from llama_cpp import Llama  # pylint: disable=no-name-in-module

from provider.embedding.base import EmbeddingBackendProvider

load_dotenv()

class Qwen3LlamaCpp(EmbeddingBackendProvider):
    """Qwen3 Llama CPP embedding provider."""
    def __init__(self):
        self.max_seq_length = int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_MAX_LENGTH", "4096"))
        self.model = Llama.from_pretrained(
            repo_id="Qwen/Qwen3-Embedding-0.6B-GGUF",
            filename="Qwen3-Embedding-0.6B-Q8_0.gguf",
            embedding=True,
            n_ctx=self.max_seq_length,
            n_batch=self.max_seq_length,  # Set batch size equal to context for embeddings
            pooling_type=1,  # 1 = LLAMA_POOLING_TYPE_MEAN (mean pooling)
        )
        self.logger = logging.getLogger(__name__)
        self.logger.debug("Model max sequence length: %d", self.max_seq_length)

    def embed(self, text: str) -> np.ndarray:
        """Generate embeddings for the provided text, chunking if necessary."""
        chunks = self.chunk_text_by_tokens(text, max_tokens=self.max_seq_length)
        self.logger.debug("Split into %d chunks", len(chunks))

        # Encode the string chunks
        raw_embeddings = self.model.embed(chunks)

        # Standardize shape: always return 2D array [num_chunks, dim]
        embeddings = raw_embeddings if isinstance(raw_embeddings,
                                                   np.ndarray) else np.asarray(raw_embeddings, dtype=np.float32)
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(1, -1)

        # Aggressive cleanup for MPS
        if os.getenv("WIKIPEDIA_EMBEDDING_MODEL_CLEANUP", "False").lower() == "true":
            self.logger.debug("Performing aggressive cleanup of model resources")
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()

        return embeddings

    def chunk_text_by_tokens(self, text: str, max_tokens: int = None, overlap_tokens: int = 200) -> list[str]:
        """Split text into chunks based on token count with overlap."""
        if max_tokens is None:
            max_tokens = self.max_seq_length

        # Tokenize the entire text via llama-cpp
        tokens = self.model.tokenize(text.encode("utf-8"), add_bos=False, special=False)

        # If text fits in one chunk, return as-is
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start_idx = 0
        stride = max(max_tokens - overlap_tokens, 1)  # prevent infinite loop when overlap >= max_tokens

        while start_idx < len(tokens):
            # Get chunk of tokens
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.model.detokenize(chunk_tokens).decode("utf-8", errors="ignore")
            chunks.append(chunk_text)

            # Move forward with overlap
            if end_idx >= len(tokens):
                break
            start_idx += stride

        return chunks
