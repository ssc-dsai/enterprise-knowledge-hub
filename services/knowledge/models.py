"""Base data model for knowledge items."""
from pydantic import BaseModel

class KnowledgeItem(BaseModel):
    """Base class for knowledge items that will be pushed to the queue."""

    name: str = ""  # Unique identifier for the knowledge item
