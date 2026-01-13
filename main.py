"""
Contains the main FastAPI application for the Enterprise Knowledge Hub.
"""
import logging
from fastapi import FastAPI

from router.knowledge.endpoints import KNOWLEDGE_BASE
from router.knowledge.endpoints import router as endpoints
from router.database_interaction.endpoints import router as db_endpoints

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logging.getLogger("router.knowledge.endpoints").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(endpoints, prefix=KNOWLEDGE_BASE, tags=["knowledge","indexer","ingest","vector"])
app.include_router(db_endpoints, prefix="/database", tags=["database interaction"])

@app.get("/health")
def hp():
    """Health check endpoint."""
    return {"status": "Healthy"}
