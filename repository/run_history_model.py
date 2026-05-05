from datetime import datetime

from peewee import AutoField, IntegerField, TextField, Model
from playhouse.postgres_ext import BinaryJSONField
from repository.base_model import TimestampTZField
from repository.database import db

RUN_HISTORY_TABLE_NAME = "run_history"

class RunHistory(Model): #pylint: disable=too-many-instance-attributes
    """Serializable record for Postgres storage."""
    id: int = AutoField()
    run_id: int | None = IntegerField(null=True)
    service_name: str = TextField()
    status: str = TextField()
    metadata = BinaryJSONField(null=True)
    timestamp: datetime = TimestampTZField()

    class Meta:
        database = db
        db_table = RUN_HISTORY_TABLE_NAME
