"""Bootstrap module"""
import os

from peewee import DatabaseProxy
from playhouse.postgres_ext import PooledPostgresqlExtDatabase

db = DatabaseProxy()
_is_initialized = False  # pylint: disable=invalid-name

def get_conn_info():
    """Get connection info to postgres from env file"""
    host = os.getenv("POSTGRES_HOST", "localhost")
    port = int(os.getenv("POSTGRES_PORT", "5432"))
    dbname = os.getenv("POSTGRES_DB", "rag")
    user = os.getenv("POSTGRES_USER", "admin")
    password = os.getenv("POSTGRES_PASSWORD", "admin")
    return {
        "host": host,
        "port": port,
        "dbname": dbname,
        "user": user,
        "password": password
    }

def build_database() -> PooledPostgresqlExtDatabase:
    """Build Pooled connection with peewee"""
    conninfo = get_conn_info()
    return PooledPostgresqlExtDatabase(
        conninfo["dbname"],
        user=conninfo["user"],
        password=conninfo["password"],
        host=conninfo["host"],
        port=conninfo["port"],
        max_connections=8,
        stale_timeout=300,
    )

def initialize_database() -> None:
    """"Init database and guard against it being run multiple times."""
    global _is_initialized  # pylint: disable=global-statement
    if _is_initialized:
        return

    db.initialize(build_database())
    _is_initialized = True

def close_database() -> None:
    """close db connection"""
    if not _is_initialized:
        return
    if not db.is_closed():
        db.close()
