CREATE TABLE IF NOT EXISTS kb_wikipedia (
    id SERIAL PRIMARY KEY,
    pid INT,
    name TEXT,
    chunk_index INT,
    content TEXT,
    source TEXT,
    last_modified_date TIMESTAMPTZ,
    embedding VECTOR(512),
    CONSTRAINT documents_pid_source_chunk_index_key UNIQUE (pid, source, chunk_index)
);

CREATE TABLE run_history (
   id SERIAL PRIMARY KEY,
   run_id INT,
   service_name TEXT,
   status TEXT,
   metadata json,
   timestamp TIMESTAMP
);

-- ivfflat index for pgvector
CREATE INDEX IF NOT EXISTS wikipedia_embedding_index
    ON kb_wikipedia USING ivfflat (embedding vector_cosine_ops) WITH (lists = 3464);

-- Indexes for text search
CREATE INDEX IF NOT EXISTS documents_name_idx ON kb_wikipedia (name);
CREATE INDEX IF NOT EXISTS documents_source_idx ON kb_wikipedia (source);
