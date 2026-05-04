from peewee import PostgresqlDatabase,DatabaseProxy
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

def get_postgresqldatabase_peewee():
    conninfo = get_conn_info()
    return PostgresqlDatabase(conninfo["dbname"], user=conninfo["user"], password=conninfo["password"], host=conninfo["host"], port=conninfo["port"])

db = DatabaseProxy()
