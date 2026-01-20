# Description

Information on the setup for the Postgres DB we have setup with vectors (pg vectors).

## Setup

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
   id SERIAL PRIMARY KEY,
   pid INT,
   name TEXT,
   chunk_index INT,
   title TEXT,
   content TEXT,
   source TEXT,
   last_modified_date DATE,
   embedding VECTOR(512),
   CONSTRAINT documents_pid_chunk_index_key UNIQUE (pid, chunk_index)
);

-- Recommended ANN index for pgvector
-- using sqrt(12millions) using approximate record size for wikipedia.
SET maintenance_work_mem = '8GB';
CREATE INDEX IF NOT EXISTS wikipedia_embedding_index
   ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 3464);

-- Indexes for text search on name and title
CREATE INDEX IF NOT EXISTS documents_name_idx ON documents (name);
CREATE INDEX IF NOT EXISTS documents_title_idx ON documents (title);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);

-- Readonly user created
CREATE USER readonly_user WITH PASSWORD 'readonly';
GRANT pg_read_all_data TO readonly_user;
```

## Migrations

### Adding source column

```sql
-- Add the source column to existing table
ALTER TABLE documents ADD COLUMN source TEXT;

-- Populate existing entries with 'enwiki'
UPDATE documents SET source = 'enwiki' WHERE source IS NULL;
```

## Gathering info

```sql
SELECT pg_indexes_size();
SELECT pg_size_pretty(pg_relation_size('documents_embedding_idx'));

-- Check index sizes for all tables
SELECT relname as table_name,
       pg_size_pretty(pg_indexes_size(relid)) as "Index Size",
       pg_size_pretty(pg_total_relation_size(relid)) As "Total Size"
FROM pg_catalog.pg_statio_user_tables
ORDER BY pg_total_relation_size(relid) DESC;
```

## Considerations

* using `half_vec` instead of vectors for the `embedding VECTOR(512)` column. Halfing bit size (16bit instead of 32bit)

## Documentation

* [PG Vector myths...](https://www.thenile.dev/blog/pgvector_myth_debunking)