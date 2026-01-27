"""
Contains the main FastAPI application for the Enterprise Knowledge Hub.
"""
import logging
from fastapi import FastAPI

from router.root.run_management_endpoints import KNOWLEDGE_BASE
from router.root.run_management_endpoints import router as endpoints
from router.root.search_retrieve_endpoints import router as db_endpoints
from router.frontend.frontend import router as frontend_router

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

logging.getLogger("router.knowledge.endpoints").setLevel(logging.DEBUG)

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(frontend_router, prefix="/frontend", tags=["Frontend"])
app.include_router(endpoints, prefix=KNOWLEDGE_BASE, tags=["Knowledge","Indexer","Ingest","Vector"])
app.include_router(db_endpoints, prefix="/database", tags=["Database Interaction"])

@app.get("/health")
def hp():
    """Health check endpoint."""
    return {"status": "Healthy"}
