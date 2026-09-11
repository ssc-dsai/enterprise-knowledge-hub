"""Postgres/pgvector repository for TBS policies knowledge base."""
from __future__ import annotations

from datetime import datetime

from repository.base import EmbeddingRepository
from repository.knowledge_tbs_policies_model import KnowledgeBaseTBSPolicies

# Dimensions the partial ivfflat index has already been created for in this process.
_indexed_dimensions: set[int] = set()


class KnowledgeTBSPoliciesRepository(EmbeddingRepository):
    """Repository to read/write TBS policy records."""

    id_field_name = "page_id"

    def __init__(self):
        super().__init__(KnowledgeBaseTBSPolicies)

    def get_by_page_id_source(self, page_id: int, source: str) -> list[KnowledgeBaseTBSPolicies]:
        """Get all chunks for a given page_id and source."""
        return self.get_chunks_by_id_source(page_id, source)

    def get_by_page_id_source_modified_date(self, page_id: int, source: str,
                                            last_date_modified: datetime,
                                            embedding_dims: int | None = None) -> KnowledgeBaseTBSPolicies | None:
        """Get a record by page_id and source if it was modified after last_date_modified."""
        return self.get_by_id_source_modified_date(page_id, source, last_date_modified, embedding_dims)

    def delete_by_page_id_source(self, page_id: int, source: str) -> None:
        """Delete all chunks for a given page_id and source."""
        self.delete_by_id_source(page_id, source)

    def ensure_dim_index(self, dimensions: int, lists: int = 100) -> None:
        """Create the partial ivfflat index for this embedding length, if not already done.

        Checks pg_indexes for the desired index if it's not in the cache.
        """
        if dimensions in _indexed_dimensions:
            return
        index_name = f"tbs_policies_embedding_idx_{dimensions}"
        db = self.model._meta.database  # pylint: disable=protected-access
        exists = db.execute_sql(
            "SELECT 1 FROM pg_indexes WHERE indexname = %s", (index_name,)
        ).fetchone()
        if exists is None:
            db.execute_sql(
                f"CREATE INDEX CONCURRENTLY IF NOT EXISTS {index_name} ON kb_tbs_policies "
                f"USING ivfflat ((embedding::vector({dimensions})) vector_cosine_ops) "
                f"WITH (lists = {lists}) WHERE embedding_dims = {dimensions};"
            )
        _indexed_dimensions.add(dimensions)
