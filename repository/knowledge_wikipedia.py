"""Postgres/pgvector repository for Wikipedia knowledge base."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from peewee import Expression, SQL

from repository.base import BaseRepository
from repository.knowledge_wikipedia_model import KnowledgeBaseWikipedia

KB_TABLE_NAME = "kb_wikipedia"

class KnowledgeWikipediaRepository(BaseRepository):
    """Repository to write Wikipedia records"""

    def __init__(self):
        super().__init__(KnowledgeBaseWikipedia)

    def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 100,
        probes: int = 100,
    ) -> list[dict]:
        """Search for similar embeddings using pgvector's <=> operator.

        Args:
            embedding: The query embedding vector.
            limit: Maximum number of results to return (acts as a safety cap).
            probes: Number of IVFFlat lists to search. Higher = better recall but slower.
                    With 3464 lists, recommended range is 60-350 (sqrt(lists) to lists/10).
        """

        # Based on:
        # SET doesn't support parameterized values, so format directly (int is safe)
        # set_probes_sql = sql.SQL("SET LOCAL ivfflat.probes = {}").format(sql.Literal(probes))

        # Use a larger limit for index scan, then filter by similarity threshold
        # The WHERE clause filters after the index scan finds candidates
        # query_sql = sql.SQL(
        #     """
        #     SELECT name, content, chunk_index, 1 - (embedding <=> %s::vector) AS similarity
        #     FROM {table}
        #     ORDER BY embedding <=> %s::vector
        #     LIMIT %s
        #     """

        embedding_vector = embedding[0] if isinstance(embedding[0], (list, tuple, np.ndarray)) else embedding
        db = self.model._meta.database

        def cosine_distance(column, embedding):
            if hasattr(embedding, 'tolist'):
                embedding = embedding.tolist()
            return Expression(column, '<=>', SQL("%s::vector", [embedding]))

        # atomic ensures all done in one transaction
        with db.atomic():
            db.execute_sql(f"SET LOCAL ivfflat.probes = {int(probes)}")
            query = (self.model
                    .select(
                        self.model.name,
                        self.model.content,
                        self.model.chunk_index,
                        (1 - cosine_distance(self.model.embedding, embedding_vector)).alias('similarity')
                    )
                    .order_by(cosine_distance(self.model.embedding, embedding_vector))
                    .limit(limit))
            results = list(query.dicts())

        return results

    def get_first_by_title_source(self, title: str, source: str) -> KnowledgeBaseWikipedia | None:
        """Query for based on title"""

        query = (self.model
                 .select()
                 .where(
                     (self.model.name == title) &
                     (self.model.source == source)
                 )
                 .get_or_none())

        return query

    def get_by_pid_source(self, pid: int, source: str) -> list[KnowledgeBaseWikipedia]:
        """Query for entire record chunks based on article title/name"""

        query = (self.model
                 .select()
                 .where(
                     (self.model.pid == pid) &
                     (self.model.source == source)
                 ))


        return query

    def get_by_pid_source_modified_date(self, pid: int, source: str, last_date_modified: datetime) -> KnowledgeBaseWikipedia | None:
        """
        Queries the database for the documents with pid and checks if the date is currently
        more recent than the one in the database

        --- Query needs to return a record (it needs to exists) AND make sure current db date is
            greater or equal to current passed date

        Returns True if the record EXISTS AND is UP TO DATE, False otherwise
        """

        query = (self.model
                 .select()
                 .where(
                     (self.model.pid == pid) &
                     (self.model.source ** source) &
                     (self.model.last_modified_date >= last_date_modified)
                 )
                 .get_or_none()
                 )
        return query

    def delete_by_pid_source(self, pid: int, source: str) -> None:
        """Delete chunks for a given pid and source"""
        query = (self.model
                 .delete()
                 .where(
                     (self.model.pid == pid) &
                     (self.model.source == source)
                 ))
        query.execute()

