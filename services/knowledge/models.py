"""Base data model for knowledge items."""
from enum import StrEnum

from pydantic import BaseModel

class KnowledgeItem(BaseModel):
    """Base class for knowledge items that will be pushed to the queue."""

    name: str = ""

class RunStatus(StrEnum):
    """Enumeration of run status (and cronjob status) for the run_history table."""
    RUN_STARTED = "Run Started"
    INGESTION_STARTED = "Ingestion Started"
    INGESTION_COMPLETED = "Ingestion Completed"
    PROCESSING_STARTED = "Processing Started"
    PROCESSING_COMPLETED = "Processing Completed"
    STORING_STARTED = "Storing Started"
    STORING_COMPLETED = "Storing Completed"
    RUN_ENDED = "Run Completed"
    DUMP_LINK_UPDATED = "New Dump Link Detected and Downloaded"
    BATCH_AVERAGE_TIME = "Processing Batch Average Time"
