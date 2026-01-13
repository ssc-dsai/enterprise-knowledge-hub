from fastapi import APIRouter, Query
from services.knowledge.models import WikipediaItem
from services.knowledge.query import QueryService


router = APIRouter()

# @router.get("/search")
#     def search_database(
#         query: str = Query(..., description="Search query")
#         # , limit: int = Query(10, description="Number of results to return")
#     )

_query_service = QueryService()

@router.get("/test")
def retrieve_wiki_articles(
):
    print("hit test endpoint success")
    _query_service.test()