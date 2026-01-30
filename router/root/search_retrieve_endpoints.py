"""
Endpoints for interacting with the knowledge database.
"""
from fastapi import APIRouter, Query
from services.database.database_service import QueryService

router = APIRouter()
_query_service = QueryService()

@router.get("/search")
def search_database(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return")
):
    """Endpoint to search Wikipedia articles by query."""
    # Call the service layer to perform the search
    print(f"Searching database for query: {query} with limit: {limit}")
    results = _query_service.search(query, limit)
    return {
        "query": query,
        "results": results
    }

@router.get("/retrieve/{title}")
def retrieve_wiki_articles(
    title:str
):
    """Get wiki article content"""
    print(f"(search_retrieve endpoints) Retrieving wiki articles for title: {title}")
    return _query_service.get_article_content_by_title(title)
