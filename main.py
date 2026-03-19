"""
Contains the main FastAPI application for the Enterprise Knowledge Hub.
"""
import logging
import logging.config
import os

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI

from router.frontend.frontend import router as frontend_router
from router.root.run_management_endpoints import KNOWLEDGE_BASE
from router.root.run_management_endpoints import router as endpoints
from router.root.search_retrieve_endpoints import router as db_endpoints

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

if(log_config := os.getenv("LOGGING_CONFIG_FILE")):
    with open(log_config) as f:
        logging.config.dictConfig(yaml.safe_load(f))

logger = logging.getLogger(__name__)

app = FastAPI()
app.include_router(frontend_router, prefix="/frontend", tags=["Frontend"])
app.include_router(endpoints, prefix=KNOWLEDGE_BASE, tags=["Knowledge (Indexing Operations)"])
app.include_router(db_endpoints, prefix="/database", tags=["Database Interaction"])

@app.get("/health")
def hp():
    """Health check endpoint."""
    return {"status": "Healthy"}

@app.get("/logging-info")
def logging_info():
    """Logging info endpoint."""
    def info(name: str) -> dict:
        log = logging.getLogger(name)
        return {
            "level": logging.getLevelName(log.level),
            "effective_level": logging.getLevelName(log.getEffectiveLevel()),
        }
    return {
        "root": logging.getLevelName(logging.root.manager.root.level),
        "router.root": info("router.root"),
        "router.root.run_management_endpoints": info("router.root.run_management_endpoints"),
        "router.root.search_retrieve_endpoints": info("router.root.search_retrieve_endpoints"),
        "provider": info("provider"),
        "services": info("services"),
    }
