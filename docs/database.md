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
-- using sqrt(12millions) using approximate record size for wikipedia.
SET maintenance_work_mem = '8GB';
CREATE INDEX IF NOT EXISTS wikipedia_embedding_index
   ON documents USING ivfflat (embedding vector_cosine_ops) WITH (lists = 3464);

-- OR
-- HNSW index
SET maintenance_work_mem = '20GB';
SET max_parallel_maintenance_workers = 24;
CREATE INDEX wikipedia_embedding_index
   ON documents USING hnsw (embedding vector_cosine_ops)
   WITH (m = 16, ef_construction = 64);
-- indexing progress
-- https://github.com/pgvector/pgvector/blob/master/README.md#hnsw
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;

-- Indexes for text search on name and title
CREATE INDEX IF NOT EXISTS documents_name_idx ON documents (name);
CREATE INDEX IF NOT EXISTS documents_title_idx ON documents (title);
CREATE INDEX IF NOT EXISTS documents_source_idx ON documents (source);

-- Readonly user created
CREATE USER readonly_user WITH PASSWORD 'readonly';
GRANT pg_read_all_data TO readonly_user;
```

## Migrations

### Adding source column/ dropping templates

First run mistakes! Here are the fixes.

```sql
-- Add the source column to existing table
ALTER TABLE documents DROP COLUMN IF EXISTS source;
ALTER TABLE documents ADD COLUMN source TEXT DEFAULT 'enwiki';
DELETE FROM documents WHERE title LIKE 'Template:%';
```

### Adding source to unique constraint

When ingesting both English and French Wikipedia dumps, `pid` values can collide across sources.
The unique constraint must include `source` to allow the same `pid + chunk_index` pair from different sources.

```sql
-- Drop the old constraint
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pid_chunk_index_key;

-- Add the new constraint with source
ALTER TABLE documents ADD CONSTRAINT documents_pid_source_chunk_index_key UNIQUE (pid, source, chunk_index);
```

### Changing documents table name

```sql
ALTER TABLE documents RENAME TO kb_wikipedia 
```

### Removing title column

```sql
ALTER TABLE kb_wikipedia DROP COLUMN title
```

### Changing last_modified_date from DATE to TIMESTAMPTZ

The `DATE` type silently truncates timestamps to day precision, which breaks the
"is up to date" comparison when re-ingesting dumps (hours/minutes/seconds are lost).

```sql
-- 1. Change column type from DATE to TIMESTAMPTZ (existing values become midnight UTC)
ALTER TABLE kb_wikipedia
    ALTER COLUMN last_modified_date TYPE TIMESTAMPTZ
    USING last_modified_date::TIMESTAMPTZ;

-- 2. Backfill existing rows: push the truncated date to end-of-day (23:59:59 UTC)
--    so that incoming records with the same calendar day but an earlier time
--    are still correctly detected as "up to date".
UPDATE kb_wikipedia
    SET last_modified_date = last_modified_date::date + INTERVAL '23 hours 59 minutes 59 seconds'
    WHERE last_modified_date = last_modified_date::date;  -- only rows still at midnight
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