"""Base class for repository classes"""
from psycopg_pool import ConnectionPool

INSERT_BATCH_SIZE = 500

# pylint: disable=too-few-public-methods
class BaseRepository:
    """Base repository class for postgres"""
    
    _batch_size: int = INSERT_BATCH_SIZE
    
    def __init__(self, pool: ConnectionPool):
        self._pool = pool
