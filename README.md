# enterprise-knowledge-hub
Enterprise Knowledge Hub

## Initial setup

Make sure you add this file to the root: `.env` (refer to `.env.example`)

To start the docker container: `docker compose up -d`

### Files for KB implementation

* Wikipedia; ensure you `mkdir wikipedia` within the `./content/` folder and drop your files there. 

### Database Setup

```bash
docker exec -it postgres-ekh psql -U admin -d rag
```

On first run ensure you have this table created:

```sql
CREATE EXTENSION IF NOT EXISTS vector;
CREATE TABLE documents (
   id SERIAL PRIMARY KEY,
   pid INT,
   name TEXT,
   chunk_index INT,
   title TEXT,
   content TEXT,
   last_modified_date DATE,
   embedding VECTOR(512),
   CONSTRAINT documents_pid_chunk_index_key UNIQUE (pid, chunk_index)
);

-- Recommended ANN index for pgvector
CREATE INDEX IF NOT EXISTS documents_embedding_idx
   ON documents USING ivfflat (embedding vector_l2_ops) WITH (lists = 100);
```

### Running locally

**Requires UV**, see [isntallation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
# IF YOUR MACHINE DOESN'T SUPPORT CUDA skip the --extra flag
uv sync --extra cuda
# see how to populate your .content/<kbprovider> folder first in the README.md there
uv run fastapi dev main.py
```

### CUDA Support

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local

### L40 GPU configurations

Those are the configs we used for our first run on the L40 GPU

```bash
WIKIPEDIA_EMBEDDING_MODEL_BACKEND=SENTENCE_TRANSFORMER
WIKIPEDIA_EMBEDDING_MODEL_CLEANUP=False
WIKIPEDIA_EMBEDDING_MODEL_BATCH_SIZE=8
WIKIPEDIA_EMBEDDING_MODEL_MAX_LENGTH=4096
WIKIPEDIA_EMBEDDING_MODEL_MAX_DIM=512 #do not change, tied to DB PgVector column config
POSTGRES_BATCH_SIZE=1000
PYTORCH_ALLOC_CONF="max_split_size_mb:128,garbage_collection_threshold:0.6,expandable_segments:True"
PYTORCH_CUDA_GPU_CAP=0.9
MODEL_SHOW_PROGRESS=False
```
