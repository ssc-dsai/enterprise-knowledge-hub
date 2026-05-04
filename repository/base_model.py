from peewee import AutoField, DateTimeField, Field, Model, IntegerField, TextField, Check
from datetime import datetime, timezone
from repository.database import db

class TimestampTZField(DateTimeField):
    field_type = "TIMESTAMPTZ"

class VectorField(Field):
    field_type = "VECTOR"

    def __init__(self, dimensions, *args, **kwargs):
        self.dimensions = dimensions
        super().__init__(*args, **kwargs)

    def get_modifiers(self):
        return [self.dimensions]

class BaseEmbeddingModel(Model):
    id: int = AutoField()
    last_modified_date = TimestampTZField(
        default=lambda: datetime.now(timezone.utc)
    )
    embedding = None

    class Meta:
        database = db
        abstract = True
