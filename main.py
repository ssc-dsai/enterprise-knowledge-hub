"""
Contains the main FastAPI application for the Enterprise Knowledge Hub.
"""
import logging
from fastapi import FastAPI
from fastapi.responses import FileResponse

from router.root.run_management_endpoints import KNOWLEDGE_BASE
from router.root.run_management_endpoints import router as endpoints
from router.root.search_retrieve_endpoints import router as db_endpoints

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

# Serve index.html at root
@app.get("/")
def read_index():
    """Serve the index.html file."""
    return FileResponse("index.html")
