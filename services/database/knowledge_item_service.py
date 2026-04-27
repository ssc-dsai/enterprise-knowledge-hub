"""
Knowledge item service to handle operations related to knowledge items, such as searching, retrieving content, and
managing records in the database.
"""
from datetime import datetime
import logging
from dataclasses import dataclass

from provider.embedding.qwen3.embedder_factory import get_embedder
from repository.model import DocumentRecord, WikipediaDbRecord
from repository.knowledge_wikipedia import KnowledgeWikipediaRepository
from repository.pool_provider import PoolProvider


@dataclass
class KnowledgeItemService():
    """Service to query wiki embeddings"""

    logger: logging.Logger
    _repository: KnowledgeWikipediaRepository

    def __init__(self, logger):
        self._logger = logger
        pool = PoolProvider.get_pool()
        self._repository = KnowledgeWikipediaRepository(pool)

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
                content=res[1],
                similarity=res[3],)
            parsed_results.append(res)
        #results = self._repository.search_by_embedding(query_embedding, min_similarity=0.8, probes=60)
        return parsed_results

    def get_article_content_by_title(self, title: str, source: str) -> str:
        """Get article content based on title and source"""
        print(f"Getting article content (all chunks) for title: {title}")

        article_pid = self._repository.get_pid_by_title(title, source)
        full_chunks = self._repository.get_record_full_chunks_content(article_pid, source)
        return full_chunks

    def delete_by_pid_source(self, pid: int, source: str) -> None:
        """Delete all records by PID and source"""
        self._repository.delete_by_pid_source(pid, source)

    def insert(self, row: WikipediaDbRecord) -> None:
        """Insert a record"""
        self._repository.insert(row)

    def record_is_up_to_date(self, pid: int, source: str, last_date_modified: datetime) -> bool:
        """
        Queries the database for the documents with pid and checks if the date is currently
        more recent than the one in the database

        --- Query needs to return a record (it needs to exists) AND make sure current db date is
            greater or equal to current passed date

        Returns True if the record EXISTS AND is UP TO DATE, False otherwise

        """
        return self._repository.record_is_up_to_date(pid, source, last_date_modified)
