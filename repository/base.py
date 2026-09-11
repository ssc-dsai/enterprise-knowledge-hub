"""Base repository class"""
from __future__ import annotations

from datetime import datetime

import numpy as np
from peewee import Expression, Model, SQL


class BaseRepository:
    """Base repository class"""

    def __init__(self, model: type[Model]):
        self.model = model

    def get_by_id(self, pk: int) -> Model | None:
        """Get by id"""
        return self.model.get_or_none(self.model.id == pk)

    def list_all(self) -> list[Model]:
        """Get all"""
        return list(self.model.select())

    def create(self, **data) -> Model:
        """Insert and return model"""
        return self.model.create(**data)

    def update(self, pk: int, **data) -> bool:
        """Update query"""
        query = self.model.update(**data).where(self.model.id == pk)
        return query.execute() > 0

    def delete(self, pk: int) -> bool:
        """Delete query"""
        query = self.model.delete().where(self.model.id == pk)
        return query.execute() > 0


class EmbeddingRepository(BaseRepository):
    """Base repository for pgvector-backed knowledge base tables.

    Subclasses represent chunked embedding tables that share the same
    (id_field, source, chunk_index) shape (e.g. kb_wikipedia's `pid`,
    kb_tbs_policies's `page_id`). Set `id_field_name` to the name of the
    model's identifying column and reuse the generic query helpers below.
    """

    #: Name of the model field used to identify a source document (e.g. "pid", "page_id").
    id_field_name: str = "id"

    def _id_column(self):
        return getattr(self.model, self.id_field_name)

    def search_by_embedding(
        self,
        embedding: list[float],
        limit: int = 100,
        probes: int = 100,
        dimensions: int | None = None,
    ) -> list[dict]:
        """Search for similar embeddings using pgvector's <=> cosine distance operator.

        Args:
            embedding: The query embedding vector.
            limit: Maximum number of results to return (acts as a safety cap).
            probes: Number of IVFFlat lists to search. Higher = better recall but slower.
            dimensions: If set, restricts the search to rows whose embedding_dims equals this value.
        """
        embedding_vector = embedding[0] if isinstance(embedding[0], (list, tuple, np.ndarray)) else embedding
        db = self.model._meta.database  # pylint: disable=protected-access

        def cosine_distance(column, emb):
            if hasattr(emb, 'tolist'):
                emb = emb.tolist()
            if dimensions is not None:
                return Expression(
                    SQL(f"({column.column_name}::vector({dimensions}))"),
                    '<=>', SQL(f"%s::vector({dimensions})", [emb])
                )
            return Expression(column, '<=>', SQL("%s::vector", [emb]))

        with db.atomic():
            db.execute_sql(f"SET LOCAL ivfflat.probes = {int(probes)}")
            query = self.model.select(
                self.model.name,
                self.model.content,
                self.model.chunk_index,
                (1 - cosine_distance(self.model.embedding, embedding_vector)).alias('similarity')
            )
            if dimensions is not None:
                # Exclude other-dimension rows before the distance expression is built.
                query = query.where(SQL("embedding_dims = %s", [dimensions]))
            query = query.order_by(cosine_distance(self.model.embedding, embedding_vector)).limit(limit)
            results = list(query.dicts())

        return results

    def get_chunks_by_id_source(self, id_value: int, source: str) -> list[Model]:
        """Get all chunks for a given identifier and source, ordered by chunk_index."""
        query = (self.model.select().where(
                     (self._id_column() == id_value) &
                     (self.model.source == source)
                 ).order_by(self.model.chunk_index))
        return list(query)

    def get_by_id_source_modified_date(self, id_value: int, source: str,
                                       last_date_modified: datetime) -> Model | None:
        """Get a record by identifier and source if modified after last_date_modified."""
        query = (self.model.select().where(
                     (self._id_column() == id_value) &
                     (self.model.source == source) &
                     (self.model.last_modified_date >= last_date_modified)
                 )
                 .get_or_none())
        return query

    def delete_by_id_source(self, id_value: int, source: str) -> None:
        """Delete all chunks for a given identifier and source."""
        query = (self.model.delete().where(
                     (self._id_column() == id_value) &
                     (self.model.source == source)
                 ))
        query.execute()
