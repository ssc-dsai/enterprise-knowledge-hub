"""Base repository class"""

from peewee import Model

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
