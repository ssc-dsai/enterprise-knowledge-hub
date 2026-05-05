"""Base repository class for databases, providing pool ."""

from peewee import Model
from typing import List, Optional, Type

class BaseRepository:
    """Base repository class for postgres"""

    def __init__(self, model: Type[Model]):
        self.model = model

    def get_by_id(self, pk: int) -> Optional[Model]:
        """Get by id"""
        return self.model.get_or_none(self.model.id == pk)

    def list_all(self) -> List[Model]:
        """Get all"""
        return [row for row in self.model.select()]

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
