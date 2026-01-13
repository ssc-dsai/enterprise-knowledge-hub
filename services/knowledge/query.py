"""Service layer to query embedding in persistance layer"""
from dataclasses import dataclass
from services.db.postgrespg import WikipediaPgRepository

import logging

@dataclass
class QueryService():
    """Service to query wiki embeddings"""
    
    logger: logging.Logger
    
    def __init__(self, repository: WikipediaPgRepository | None = None):
        self._repository = repository or WikipediaPgRepository.from_env()
        
    def test(self):
        print("servicelayer ok")
        result = self._repository.get_record()
        print("result")
        print(result)
        return result