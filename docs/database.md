# Description

Information on the setup for the Postgres DB we have setup with vectors (pg vectors).

## Database Schema

The canonical schema lives in the Helm chart SQL files and is used directly by the
pgvector initdb ConfigMap at deploy time:

* [`deployments/ekh/sql/00_init_pgvector.sql`](../deployments/ekh/sql/00_init_pgvector.sql) — enables the `vector` extension
* [`deployments/ekh/sql/01_init_ekh_schema.sql`](../deployments/ekh/sql/01_init_ekh_schema.sql) — creates tables and indexes

### Alternate index (HNSW instead of ivfflat)

```sql
-- HNSW index (alternative to ivfflat, better recall but slower build)
SET maintenance_work_mem = '20GB';
SET max_parallel_maintenance_workers = 24;
CREATE INDEX wikipedia_embedding_index
   ON kb_wikipedia USING hnsw (embedding vector_cosine_ops)
   WITH (m = 16, ef_construction = 64);
-- indexing progress
-- https://github.com/pgvector/pgvector/blob/master/README.md#hnsw
SELECT phase, round(100.0 * blocks_done / nullif(blocks_total, 0), 1) AS "%" FROM pg_stat_progress_create_index;

-- Readonly user
CREATE USER readonly_user WITH PASSWORD 'readonly';
GRANT pg_read_all_data TO readonly_user;
```

## Migrations

### v0.1.0 → v1.10+

If you are upgrading from the original `documents` table (v0.1.0) to the current
`kb_wikipedia` schema, run the following migration in order:

```sql
-- 1. Add source column
ALTER TABLE documents ADD COLUMN IF NOT EXISTS source TEXT DEFAULT 'enwiki';

-- 2. Fix unique constraint to include source
ALTER TABLE documents DROP CONSTRAINT IF EXISTS documents_pid_chunk_index_key;
ALTER TABLE documents ADD CONSTRAINT documents_pid_source_chunk_index_key UNIQUE (pid, source, chunk_index);

-- 3. Rename table
ALTER TABLE documents RENAME TO kb_wikipedia;

-- 4. Remove title column
ALTER TABLE kb_wikipedia DROP COLUMN IF EXISTS title;

-- 5. Change last_modified_date from DATE to TIMESTAMPTZ
ALTER TABLE kb_wikipedia
    ALTER COLUMN last_modified_date TYPE TIMESTAMPTZ
    USING last_modified_date::TIMESTAMPTZ;

-- 6. Backfill existing rows to end-of-day so "is up to date" checks work
UPDATE kb_wikipedia
    SET last_modified_date = last_modified_date::date + INTERVAL '23 hours 59 minutes 59 seconds'
    WHERE last_modified_date = last_modified_date::date;
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