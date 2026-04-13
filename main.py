"""
Contains the main FastAPI application for the Enterprise Knowledge Hub.
"""
import logging
import logging.config
import os

import yaml
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi_crons import Crons, get_cron_router

from router.frontend.frontend import router as frontend_router
from router.root.run_management_endpoints import KNOWLEDGE_BASE
from router.root.run_management_endpoints import router as endpoints
from router.root.search_retrieve_endpoints import router as db_endpoints

from content_scraper.base_cronjob import main as kb_scraper_main

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)

if(log_config := os.getenv("LOGGING_CONFIG_FILE")):
    with open(log_config, encoding="UTF-8") as f:
        logging.config.dictConfig(yaml.safe_load(f))

logger = logging.getLogger(__name__)

app = FastAPI()
crons = Crons(app)

# THIS LINE CONFIGURES FASTAPI-CRONS ENDPOINTS, WHICH DO NOT WORK!!!!!!!!!
# app.include_router(get_cron_router(), prefix="/_internal/crons", tags=["internal"])
app.include_router(frontend_router, prefix="/frontend", tags=["Frontend"])
app.include_router(endpoints, prefix=KNOWLEDGE_BASE, tags=["Knowledge (Indexing Operations)"])
app.include_router(db_endpoints, prefix="/database", tags=["Database Interaction"])

# THIS WORKS PERFECTLY FINE, WITH OUTPUT SENT TO FASTAPI TERMINAL. 3 FILES ARE THEN CREATED, cron_state.db,
# cron_state.db-shm, cron_state.db-wal, WHICH STORE THE CRONJOB STATES IN SQLITE- WHICH IS VIEWABLE AND WORKS
# -HOWEVER, IT ONLY STORES LATEST RUN TIME AND THATS IT, ONE ROW
@crons.cron("* * 1 * *", name = "run_knowledge_base_scraper")
def run_knowledge_base_scraper():
    """Cronjob that runs the knowledge base scraper to update new knowledge base dumps/files"""
    # BELOW LINE CALLS APPROPRIATE SCRIPTS, THIS WORKS!
    # kb_scraper_main()
    print(crons.jobs)
    for job in crons.jobs:
        logger.info("Scheduled job: %s", job.name)

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
