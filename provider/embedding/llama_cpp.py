"""Torch-based embeddings backed by llama.cpp models."""

# pylint: disable=duplicate-code
import os

from transformers import AutoModel, AutoTokenizer

from provider.embedding.base import EmbeddingBackendProvider

class LlamaCPPEmbeddingBackend(EmbeddingBackendProvider):
    """Embedding backend powered by a llama.cpp compatible model."""

    def __init__(self, model_name: str, device: str = "cuda", max_seq_len: int = 512):
        self.model_name = model_name
        self.device = device
        self.max_seq_len = max_seq_len

        # for testing purposes here
        local_dir = os.path.expanduser(
            "~/.cache/huggingface/hub/models--Qwen--Qwen3-Embedding-0.6B-GGUF/"
            "snapshots/370f27d7550e0def9b39c1f16d3fbaa13aa67728/"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            local_dir,
            gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
            local_files_only=True,
        )
        self.model = AutoModel.from_pretrained(
            local_dir,
            gguf_file="Qwen3-Embedding-0.6B-Q8_0.gguf",
            local_files_only=True,
        ).to(device)

        # self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # self.model = AutoModel.from_pretrained(model_name).to(device)

        self.model.eval()

    def set_device(self, device: str) -> None:
        """Update the active device (e.g., cpu/cuda)."""
        self.device = device
