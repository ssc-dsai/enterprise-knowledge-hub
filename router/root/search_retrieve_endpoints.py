"""
Endpoints for interacting with the knowledge database.
"""
import logging

from fastapi import APIRouter, Query
from services.database.knowledge_item_service import KnowledgeItemService

router = APIRouter()
logger = logging.getLogger(__name__)
_knowledge_item_service = KnowledgeItemService(logger)

@router.get("/search")
def search_database(
    query: str = Query(..., description="Search query"),
    limit: int = Query(10, description="Number of results to return")
):
    """Endpoint to search Wikipedia articles by query."""
    # Call the service layer to perform the search
    print(f"Searching database for query: {query} with limit: {limit}")
    results = _knowledge_item_service.search(query, limit)
    return {
        "query": query,
        "results": results
    }

@router.get("/retrieve/{title}")
def retrieve_wiki_articles(
    title: str,
    source: str = 'enwiki' # this needs to be fixed assume enwiki for now.
):
    """Get wiki article content"""
    print("HELLO")
    print(f"(search_retrieve endpoints) Retrieving wiki articles for title: {title}")
    return _knowledge_item_service.get_article_content_by_title(title, source)
