"""Base models"""
from datetime import datetime, timezone
from peewee import AutoField, DateTimeField, Field, Model

from repository.database import db

class TimestampTZField(DateTimeField):
    """TimestempTZ column ORM representation for Peewee"""
    field_type = "TIMESTAMPTZ"

class VectorField(Field):
    """Vector column ORM representation for Peewee"""
    field_type = "VECTOR"

    def __init__(self, dimensions, *args, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def get_modifiers(self):
        return [self.dimensions]

class BaseEmbeddingModel(Model):
    """Base model for embedding tables"""
    id: int = AutoField()
    last_modified_date = TimestampTZField(
        default=lambda: datetime.now(timezone.utc), null=True
    )

    @property
    def embedding(self):
        """embedding field.  Need to define in subclasses"""
        raise NotImplementedError("Subclasses must define embedding field")

    class Meta:  # pylint: disable=too-few-public-methods
        """Configuration for the model"""
        database = db
        abstract = True
