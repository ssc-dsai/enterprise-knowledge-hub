# enterprise-knowledge-hub

## Initial setup

Make sure you add this file to the root: `.env` (refer to `.env.example`)

To start the docker container: `docker compose up -d`

### Files for KB implementation

* Wikipedia; ensure you `mkdir wikipedia` within the `./content/` folder and drop your files there.

### Cronjob Setup (Knowledge base links and sources update)
IMPORTANT: For knowledge base scraping cronjob to run, you require a `mkdir wikipedia` within the `./content`
folder, if not already made

Instead of using the crontab, we are using the fastapi-crons library to control cronjobs through our
fastapi server, with cronjob timings configurable in main.py. Cronjobs automatically run on server start.

To manual test the cronjob, visit the /crons/{job_name}/run endpoint (MORE INFO IN FASTAPI SERVER START DOCS)

### Database Setup

```bash
docker exec -it postgres-ekh psql -U admin -d rag
```

On first run **ensure you have this table created**, please see setup section in [database documentation](docs/database.md#setup).

### Running locally

**Requires UV**, see [installation](https://docs.astral.sh/uv/getting-started/installation/)

```bash
uv venv
source .venv/bin/activate
# IF YOUR MACHINE DOESN'T SUPPORT CUDA skip the --extra flag
uv sync --extra cuda
# see how to populate your .content/<kbprovider> folder first in the README.md there
uv run --env-file .env fastapi dev main.py
```

#### Tests

You can run tests via `uv run -m pytest` command.

#### Pre-commit hooks

This project uses [pre-commit](https://pre-commit.com/) to run code quality checks (e.g.`pyupgrade`) before each commit. After cloning, run:

```bash
uv run pre-commit install
```

This only needs to be done once. From then on, hooks run automatically on `git commit`. To run them manually on all files:

```bash
uv run pre-commit run --all-files
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

## Notes and help

### SSL issues

Corporate firewall might intercepts some http requests and you might end up with a SSL untrusted cert issue.
To fix this you have to install the ICM certificates from the chain that your system doesn't trust.

[Here is how to fix it](./docs/ssl-issue.md).

### WSL Space management

Running this project can be space intensive especially the `./cache/huggingface` folder.

For those running WSL you can easily check what is the  space of the current VDI in PowerShell:

```bash
# replace distro by distro name ... (wsl --list --verbose)
wsl.exe --system -d Ubuntu-22.04 df -h /mnt/wslg/distro
```

Docker clean up that is also safe, it will prune all unused images, containers and volumes that are dangling.

```bash
docker system prune -a --volumes -f
```

## Logger configuration

If you wish to customize the logging level you can do so by dropping in the root directory a `touch ./logging.yaml` file with the following content:

The `disable_existing_loggers: false` is important since it allows you to keep config from the code.

```yaml
version: 1
disable_existing_loggers: false
loggers:
  __main__:
    level: DEBUG
  router.root.run_management_endpoints:
    level: DEBUG
root:
  level: INFO

```

Keep in mind you can hit http://localhost:8000/logging-info endpoint to get info on current log levels.


## File Descriptions and Repo Structure

`main.py`: Entry point for running the Enterprise Knowledge Hub.
---
### provider/
#### embedding/
- **`base.py`**: Abstract base class for implementing embedding models.
- **`rng_embedder.py`**: Fake embedding generator for testing.
- **`tokenizer.py`**: Tokenizer module.

- **qwen3/**:
  - `embedder_factory.py`: A factory to dynamically select embedding models at runtime.
  - `sentence_transformer.py`: Embedding implementation using sentence transformers.

#### queue/
- **`base.py`**: Abstract base class for queue providers.
- **`RabbitMQ.py`**: Handles RabbitMQ interaction, implementing queue operations defined in base.py.
---
### repository/
- **`base.py`**: Base repository class for databases.
- **`knowledge_wikipedia.py`**: Postgres/pgvector repository for Wikipedia knowledge base.
- **`model.py`**: Persistence models for knowledge base, defining the structure of records stored in
the database (PostgreSQL).
- **`pool_provider.py`**: Handles PostgreSQL communication and database interaction.
- **`run_history.py`**: Postgres/pgvector repository for run-history table.

---
### router/

#### frontend/
- **`index.html`**: The user-facing (developers only) UI served by the Enterprise Knowledge Hub.
- **`frontend.py`**: Python backend for serving the frontend.

- **templates/**:
    - `status.html`: Template page for the dev frontend's status page (run-history table).

#### root/
- **`search_retrieve_endpoints.py`**: Contains endpoints to search and retrieve data from the knowledge database.
- **`run_management_endpoints.py`**: Endpoints to manage creation, deletion, updates, and index runs.
- **`run_state.py`**: Manages the state of index processing runs.
---
### services/

#### content_scraper/
- **`base_cronjob.py`**: Meant to hold the scripts for knowledgebase content updating cronjob (for example, downloading new wikipedia dumps when available)

#### database/
- **`knowledge_item_service.py`**: Service to query and log to knowledge base table.
- **`run_history_service.py`**: Service to interact with run history table.

#### knowledge/
- **`base.py`**: Core logic for building and running knowledge bases.
- **`models.py`**: Data models for knowledge base items.
- **`batch_handler.py`**: BatchHandler class to process multiple items together, improving efficiency and reducing overhead in
queue processing.

- **wikipedia/**:
    - **`wikipedia.py`**: Processes Wikipedia XML files and converts them to a format suitable for use in the system (implementing the abstract class)
    - **`models.py`**: Data models for Wikipedia items.

#### queue/
- **`queue_service.py`**: Handles read and write operations in the queue system.
- **`queue_worker.py`**: Manages worker tasks for queue read operations.
