"""Service layer to query embedding in persistance layer"""
import logging
from dataclasses import dataclass

from provider.embedding.qwen3.embedder_factory import get_embedder
from services.db.model import DocumentRecord
from services.db.postgrespg import WikipediaPgRepository


@dataclass
class QueryService():
    """Service to query wiki embeddings"""

    logger: logging.Logger

    def __init__(self, repository: WikipediaPgRepository | None = None):
        self._repository = repository or WikipediaPgRepository.from_env()

    @property
    def embedder(self):
        """Get embedder"""
        return get_embedder()

    def search(self, query: str, limit: int =10) -> list[DocumentRecord]:
        """Search Wikipedia articles by query embedding."""
        # Use is_query=True to apply the Qwen3 query instruction prefix
        # This is critical for asymmetric retrieval (query vs document)
        query_embedding = self.embedder.embed(query, is_query=True)
        results = self._repository.search_by_embedding(query_embedding, limit, probes=150)
        parsed_results = []
        for res in results:
            res = DocumentRecord(
                chunk_index=res[2],
                name=res[0],
                title=res[0],
                content=res[1],
                similarity=res[3],)
            parsed_results.append(res)
        #results = self._repository.search_by_embedding(query_embedding, min_similarity=0.8, probes=60)
        return parsed_results

    def get_article_content_by_title(self, title: str) -> list[DocumentRecord]:
        """Get article contentn based on title"""

        result = self._repository.get_record_content(title)
        return result
