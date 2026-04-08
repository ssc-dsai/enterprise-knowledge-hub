"""
Main endpoints for knowledge management (creation/delete/update).
"""
import logging
import os
from dotenv import load_dotenv
from fastapi import APIRouter
from fastapi import BackgroundTasks

from provider.queue.rabbitmq import RabbitMQProvider
from router.root.run_state import RunState
from services.database.run_history_service import RunHistoryService
from services.knowledge.wikipedia.wikipedia import WikipediaKnowledgeService
from services.queue.queue_service import QueueService

load_dotenv()

logger = logging.getLogger(__name__)

router = APIRouter()

KNOWLEDGE_BASE = "/knowledge"

# initialize the queue service here
_queue_service = QueueService(queue_provider=RabbitMQProvider(url=os.getenv("RABBITMQ_URL"),
                                                              logger=logger), logger=logger)
_run_history_service = RunHistoryService(logger)
_wikipedia_service = WikipediaKnowledgeService(queue_service=_queue_service, logger=logger,
                                               run_history_service=_run_history_service)
_wikipedia_state = RunState()

def _run_wikipedia_task():
    """Wrapper that manages the running state flag."""
    try:
        _wikipedia_service.run()
    finally:
        _wikipedia_state.stop()

@router.get("/wikipedia/stop")
async def stop_wikipedia_run():
    """
    Endpoint to stop current running process
    """
    if not _wikipedia_state.is_running():
        return {"message": "No wikipedia run is currently in progress"}

    _wikipedia_service.request_stop()
    _wikipedia_state.stop()
    return {"message": "Stop event requested for current wikipedia run"}

@router.get("/wikipedia/run")
def wikipedia_run(background_tasks: BackgroundTasks):
    """Endpoint to trigger Wikipedia full run"""
    if not _wikipedia_state.try_start():
        return {
            "message": "Wikipedia run already in progress.",
            "details": "Follow progress at frontend/status"
        }

    background_tasks.add_task(_run_wikipedia_task)
    return {
        "message": "Wikipedia run started.",
        "details": "Follow progress at frontend/status"
    }
