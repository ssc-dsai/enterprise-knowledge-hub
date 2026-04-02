"""Pool Provider class"""
import os
from dotenv import load_dotenv
from pgvector.psycopg import register_vector
from psycopg_pool import ConnectionPool

load_dotenv()

class PoolProvider:
    """Pool provider to ensure singleton in repo classes"""
    _pool: ConnectionPool | None = None

    @classmethod
    def initialize(cls):
        """Init connection pool"""
        pool_size = int(os.getenv("POSTGRES_POOL_SIZE", "5"))
        host = os.getenv("POSTGRES_HOST", "localhost")
        port = int(os.getenv("POSTGRES_PORT", "5432"))
        dbname = os.getenv("POSTGRES_DB", "postgres")
        user = os.getenv("POSTGRES_USER", "postgres")
        password = os.getenv("POSTGRES_PASSWORD", "postgres")
        conninfo = f"postgresql://{user}:{password}@{host}:{port}/{dbname}"

        cls._pool = ConnectionPool(
            conninfo,
            min_size=1,
            max_size=pool_size,
            open=False,
            configure=register_vector,
        )
        cls._pool.open()
        # TODO AR: we manually open, so we gotta manually close

    @classmethod
    def get_pool(cls) -> ConnectionPool:
        """Getter for connection pool"""
        if cls._pool is None:
            cls.initialize()
        return cls._pool
