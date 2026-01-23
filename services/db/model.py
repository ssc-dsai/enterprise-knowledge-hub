"""Persistant layer models"""
from typing import TypedDict

class DocumentRecord(TypedDict):
    """Model for records in documents table"""
    name: str
    title: str
    content: str
    chunk_index: int
    similarity: float | None
