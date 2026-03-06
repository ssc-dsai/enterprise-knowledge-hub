"""Fake embedding generator"""
import os
import hashlib
from typing import List

import numpy as np

class RNGEmbedder:
    """
    Fake embedding generator for pipeline testing.
    """

    def __init__(self, embedding_dim: int):
        self.embedding_dim = embedding_dim
        self.global_seed = int(os.getenv("WIKIPEDIA_RNG_SEED", "0"))

    @staticmethod
    def hash_to_uint32(text: str) -> int:
        """
            Text to fixed sized numeric int value
        """
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        return int.from_bytes(digest[:4], "little", signed=False)

    def generate_vector(self, text: str) -> np.ndarray:
        """Generate RNG vectors"""
        text_seed = self.hash_to_uint32(text)
        seed = text_seed ^ self.global_seed

        rng = np.random.default_rng(seed)

        vec = rng.standard_normal(self.embedding_dim, dtype=np.float32)

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        return vec.astype(np.float32)

    def encode(self, texts: List[str]) -> np.ndarray:
        """
        encode function to mimic base.py encode
        """
        vectors = [self.generate_vector(s) for s in texts]
        return np.vstack(vectors).astype(np.float32)
