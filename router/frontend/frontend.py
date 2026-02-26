"""
Frontend router for serving static files.
"""
from fastapi import APIRouter, Request
from fastapi.responses import FileResponse
from fastapi.templating import Jinja2Templates
from repository.postgrespg import RunHistoryPGRepository

router = APIRouter()

_run_history_table = RunHistoryPGRepository
templates = Jinja2Templates(directory="router/frontend/templates")

@router.get("/")
def dev_frontend():
    """Serve the index.html file."""
    return FileResponse("router/frontend/index.html")

@router.get("/status")
def status(request: Request):
    rows = _run_history_table().run_history_table_rows()
    return templates.TemplateResponse("status.html", {"request": request, "rows": rows})

# Here we can add frontend endpoints related to status of db (size, number of articles, etc.)
