from peewee import PostgresqlDatabase
from repository.knowledge_wikipedia_model import KnowledgeBaseWikipedia
from repository.run_history_model import RunHistory


def run_init_migration(db: PostgresqlDatabase):
    db.connect()
    db.execute_sql("CREATE EXTENSION IF NOT EXISTS vector;")
    db.create_tables([KnowledgeBaseWikipedia, RunHistory], safe=True)
    db.execute_sql("""
                   CREATE INDEX IF NOT EXISTS wikipedia_embedding_index
                   ON kb_wikipedia USING ivfflat (embedding vector_cosine_ops) WITH (lists = 3464);
                   """)
    db.close()
