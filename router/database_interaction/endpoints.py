"""
Endpoints for interacting with the knowledge database.
"""
from fastapi import APIRouter, Query
from services.knowledge.query import QueryService

router = APIRouter()
_query_service = QueryService()

@router.get("/test")
def retrieve_wiki_articles(
):
    """Test endpoint to verify database interaction."""
    print("hit test endpoint success")

@router.get("/search")
def search_database(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return")
):
    """Endpoint to search Wikipedia articles by query."""
    # Call the service layer to perform the search
    results = _query_service.search(query, limit)
    return {
        "query": query,
        "results": results
    }
