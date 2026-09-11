"""Service layer for TBS policy knowledge items."""
from datetime import datetime
import logging
from dataclasses import dataclass

import numpy as np

from repository.knowledge_tbs_policies_model import KnowledgeBaseTBSPolicies
from repository.knowledge_tbs_policies import KnowledgeTBSPoliciesRepository


@dataclass
class TBSPolicyItemService:
    """Service to manage TBS policy embeddings."""

    logger: logging.Logger
    _repository: KnowledgeTBSPoliciesRepository

    def __init__(self, logger):
        self._logger = logger
        self._repository = KnowledgeTBSPoliciesRepository()

    def insert(self, row: dict) -> KnowledgeBaseTBSPolicies:
        """Insert a record."""
        # First row at a new dimension triggers its partial ivfflat index automatically.
        self._repository.ensure_dim_index(row['embedding_dims'])
        return self._repository.create(
            page_id=row['page_id'],
            chunk_index=row['chunk_index'],
            name=row['name'],
            content=row['content'],
            last_modified_date=row['last_modified_date'],
            embedding=row['embedding'],
            embedding_dims=row['embedding_dims'],
            source=row['source'],
        )

    def search_by_embedding(self, embedding: list[float], limit: int = 100) -> list[dict]:
        """Semantic search over kb_tbs_policies by embedding similarity."""
        # Rows of multiple embedding lengths share this table, so the query's own
        # length picks which partial ivfflat index (embedding_dims) gets used.
        vector = embedding[0] if isinstance(embedding[0], (list, tuple, np.ndarray)) else embedding
        return self._repository.search_by_embedding(embedding, limit=limit, dimensions=len(vector))

    def delete_by_page_id_source(self, page_id: int, source: str) -> None:
        """Delete all chunks for a page_id and source."""
        self._repository.delete_by_page_id_source(page_id, source)

    def record_is_up_to_date(self, page_id: int, source: str, last_date_modified: datetime) -> bool:
        """Return True if the record exists and is at least as recent as last_date_modified."""
        if last_date_modified is None:
            return False
        result = self._repository.get_by_page_id_source_modified_date(page_id, source, last_date_modified)
        return result is not None
