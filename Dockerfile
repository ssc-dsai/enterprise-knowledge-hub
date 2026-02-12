FROM nvidia/cuda:13.1.1-cudnn-devel-ubuntu24.04
#FROM python:3.12.12-trixie

ARG UV_ARGS="--extra cuda"

# install uv (from https://docs.astral.sh/uv/guides/integration/docker/#installing-uv)
# The installer requires curl (and certificates) to download the release archive
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV UV_NO_DEV=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/root/.local/bin/:$CUDA_HOME/bin:$PATH"

WORKDIR /app

COPY main.py ./
COPY provider ./provider
COPY repository ./repository
COPY router ./router
COPY services ./services

COPY pyproject.toml uv.lock ./

# need extra work for llama_cpp stuff. see pyproject.toml for details.

RUN uv sync --locked $UV_ARGS

# ENV VIRTUAL_ENV=/app/.venv \
# 	PATH=/app/.venv/bin:$PATH

CMD ["uv", "run", "fastapi", "run", "main.py"]