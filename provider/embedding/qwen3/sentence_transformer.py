"""Qwen3 embeddings backed by sentence-transformers."""

# pylint: disable=duplicate-code
import logging
import os

import numpy as np
import torch
import torch.cuda
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from provider.embedding.base import EmbeddingBackendProvider

load_dotenv()

class Qwen3SentenceTransformer(EmbeddingBackendProvider):
    """Qwen3 Sentence Transformer embedding provider."""
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(os.getenv("PYTORCH_CUDA_GPU_CAP", "0.8")))
        # Reduce allocator chunk size to limit 4 GB blocks
        os.environ.setdefault("PYTORCH_ALLOC_CONF", "max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:False")

        dtype_env = os.getenv("WIKIPEDIA_EMBEDDING_MODEL_DTYPE", "float16").lower()
        dtype_map = {"float16": torch.float16, "float32": torch.float32}
        dtype = dtype_map.get(dtype_env, torch.float16)
        if dtype_env not in dtype_map:
            self.logger.warning("Invalid WIKIPEDIA_EMBEDDING_MODEL_DTYPE '%s', defaulting to float16", dtype_env)

        # Prefer float32 on MPS for stability
        if torch.backends.mps.is_available():
            dtype = torch.float32

        model_device = torch.device("cuda") if torch.cuda.is_available() else torch.device("mps") if torch.backends.mps.is_available() else "auto" #pylint: disable=line-too-long

        self.model = SentenceTransformer(
            "Qwen/Qwen3-Embedding-0.6B",
            model_kwargs={
                "device_map": model_device,
                "dtype": dtype,
                "attn_implementation": "flash_attention_2" if torch.cuda.is_available() else "",
            },
            tokenizer_kwargs={"padding_side": "left"},
        )
        self.model.max_seq_length = int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_MAX_LENGTH", "4096"))
        self.logger.debug("Model loaded on device: %s", self.model.device)
        self.logger.debug("Model max sequence length: %d", self.model.max_seq_length)

    def embed(
        self,
        text: str,
        dim: int = int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_MAX_DIM", "1024")),
    ) -> np.ndarray:
        """Generate embeddings for text, chunking when necessary."""
        chunks = self.chunk_text_by_tokens(text, max_tokens=self.model.max_seq_length)
        self.logger.debug("Split into %d chunks", len(chunks))

        # Encode the string chunks
        embeddings = self.model.encode(
            chunks,
            convert_to_tensor=False,
            show_progress_bar=bool(os.getenv("MODEL_SHOW_PROGRESS", "True").lower() == "true"),
            # Lower batch size for potentially large chunks
            batch_size=int(os.getenv("WIKIPEDIA_EMBEDDING_MODEL_BATCH_SIZE", "1")),
            truncate_dim=dim
        )

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
            max_tokens = self.model.max_seq_length

        # Tokenize the entire text
        tokens = self.model.tokenizer.encode(text, add_special_tokens=False)

        # If text fits in one chunk, return as-is
        if len(tokens) <= max_tokens:
            return [text]

        chunks = []
        start_idx = 0

        while start_idx < len(tokens):
            # Get chunk of tokens
            end_idx = min(start_idx + max_tokens, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]

            # Decode back to text
            chunk_text = self.model.tokenizer.decode(chunk_tokens, skip_special_tokens=True)
            chunks.append(chunk_text)

            # Move forward with overlap
            start_idx += max_tokens - overlap_tokens

        return chunks
