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
   file_name TEXT NOT NULL,
   content TEXT,
   embedding VECTOR(1024),
   CONSTRAINT uniquename UNIQUE (file_name)
);
```

### Running locally

**Requires UV**, see [isntallation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
uv sync
# see how to populate your .content/<kbprovider> folder first in the README.md there
uv run fastapi dev main.py
```

### CUDA Support

https://developer.nvidia.com/cuda-downloads?target_os=Linux&target_arch=x86_64&Distribution=WSL-Ubuntu&target_version=2.0&target_type=deb_local
