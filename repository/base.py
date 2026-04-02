from psycopg_pool import ConnectionPool

class BaseRepository:
    """Base repository class for postgres"""
    def __init__(self, pool: ConnectionPool):
        self._pool = pool