"""Base data model for knowledge items."""
from pydantic import BaseModel

class KnowledgeItem(BaseModel):
    """Base class for knowledge items that will be pushed to the queue."""

    name: str = ""  # Unique identifier for the knowledge item

class RunStatus(StrEnum):
    """Enumeration of run status for the run_history table."""
    RUN_STARTED = "Run Started"
    INGESTION_STARTED = "Ingestion Started"
    INGESTION_COMPLETED = "Ingestion Completed"
    PROCESSING_STARTED = "Processing Started"
    PROCESSING_COMPLETED = "Processing Completed"
    STORING_STARTED = "Storing Started"
    STORING_COMPLETED = "Storing Completed"
    RUN_ENDED = "Run Completed"