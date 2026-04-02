import os
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()

class PoolProvider:
    _pool: ConnectionPool | None = None

    def initialize(self):
        """Init connection pool"""
        pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        dbname = os.getenv("POSTGRES_DB", "postgres")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"
        
        self._pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=pool_size,
            open=False,
            configure=register_vector,
        )
        self._pool.open()  #TODO AR: we manually open, so we gotta manually close?

    @classmethod
    def get_pool(cls) -> ConnectionPool:
        """Getter for connection pool"""
        if cls._pool is None:
            cls.initialize()
        return cls._pool

# class ConnectionPoolPG:
    
#     _batch_size: int
    
#     def __init__(
#         self,
#         conninfo: str,
#         pool_size: int = 5,
#         batch_size: int = 500,
#     ) -> None:
#         """Open a connection pool."""
#         self._batch_size = batch_size
#         self._pool = ConnectionPool(
#             conninfo,
#             min_size=1,
#             max_size=pool_size,
#             open=False,
#             configure=register_vector,
#         )
#         self.open()  #TODO AR: we manually open, so we gotta manually close?
        
        
#     @classmethod
#     def from_env(cls) -> "ConnectionPool":
#         """
#         Docstring for from_env


#         :param cls: Description
#         :return: Description
#         :rtype: WikipediaPgRepository
#         """

#         host = os.getenv("POSTGRES_HOST", "localhost")
#         port = int(os.getenv("POSTGRES_PORT", "5432"))
#         dbname = os.getenv("POSTGRES_DB", "postgres")
#         user = os.getenv("POSTGRES_USER", "postgres")
#         password = os.getenv("POSTGRES_PASSWORD", "postgres")
#         pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
#         batch_size = int(os.getenv("POSTGRES_BATCH_SIZE", "500"))
#         conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

#         return cls(conninfo=conninfo, pool_size=pool_size, batch_size=batch_size)
    
#     def open(self):
#         """Open the pool explictly, when open=False"""
#         self._pool.open()
    
#     @contextmanager
#     def connection(self):
#         """
#         Context manager that yields a connection.
#         """
#         with self._pool.connection() as conn:
#             yield conn