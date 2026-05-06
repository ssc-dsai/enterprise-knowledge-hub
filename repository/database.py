from peewee import DatabaseProxy
from playhouse.postgres_ext import PooledPostgresqlExtDatabase
from repository.migration.initial_baseline import run_init_migration

import os

def get_conn_info():
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

db = DatabaseProxy()

def initialize_database():
    conninfo = get_conn_info()
    pgdb = PooledPostgresqlExtDatabase(conninfo["dbname"], user=conninfo["user"], password=conninfo["password"],
                            host=conninfo["host"], port=conninfo["port"], max_connections=8, stale_timeout=300,)
    db.initialize(pgdb)
    run_init_migration(db)
