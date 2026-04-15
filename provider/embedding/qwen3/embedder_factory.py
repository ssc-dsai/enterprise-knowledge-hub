"""Factory to choose embedder"""
from functools import lru_cache

@lru_cache(maxsize=1)
def get_embedder():
    """get SentenceTransformer embedder"""
    from provider.embedding.qwen3.sentence_transformer import Qwen3SentenceTransformer # pylint: disable=import-outside-toplevel
    return Qwen3SentenceTransformer()
