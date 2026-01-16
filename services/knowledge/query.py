"""Service layer to query embedding in persistance layer"""
from dataclasses import dataclass
import os
import logging
from services.db.postgrespg import WikipediaPgRepository

import logging

if os.getenv("WIKIPEDIA_EMBEDDING_MODEL_BACKEND", "LLAMA").upper() == "SENTENCE_TRANSFORMER":
    from provider.embedding.qwen3.sentence_transformer import Qwen3SentenceTransformer
    embedder = Qwen3SentenceTransformer()
else:
    from provider.embedding.qwen3.llama_embed import Qwen3LlamaCpp
    embedder = Qwen3LlamaCpp()

@dataclass
class QueryService():
    """Service to query wiki embeddings"""

    logger: logging.Logger

    def __init__(self, repository: WikipediaPgRepository | None = None):
        self._repository = repository or WikipediaPgRepository.from_env()

    def test(self):
        """Test method to verify service layer functionality."""
        print("servicelayer ok")
        result = self._repository.get_record()
        print("result")
        print(result)

    def search(self, query: str, limit: int =10):
    """Search Wikipedia articles by query embedding."""
        query_embedding = embedder.embed(query)
        results = self._repository.search_by_embedding(query_embedding, limit)
        return results
