"""
Frontend router for serving static files.
"""
import logging

from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from services.database.run_history_service import RunHistoryService

router = APIRouter()

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="router/frontend/templates")
run_history_service = RunHistoryService(logger)

@router.get("/")
def dev_frontend():
    """Serve the index.html file."""
    return FileResponse("router/frontend/index.html")

@router.get("/status")
def status(request: Request):
    """Serve the status page with run history."""
    rows = run_history_service.run_history_table_rows()
    return templates.TemplateResponse("status.html", {"request": request, "rows": rows})
