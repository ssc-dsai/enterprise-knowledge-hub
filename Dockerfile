# syntax=docker/dockerfile:1
# BASE_IMAGE: "cpu" or "cuda" - selects which base image to use
ARG BASE_IMAGE=cpu

# ── CUDA builder ───────────────────────────────────────────────────────────────
# nvidia/cuda devel image is needed to compile flash-attn; it is NOT used at runtime.
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS builder-cuda
# For local dev:  docker build --build-arg BASE_IMAGE=cuda --build-arg CUDA_ARCH="8.6" -t ekh:local-8.6 .
# Version that must match the GPU you are running this service on --> https://developer.nvidia.com/cuda/gpus
ARG CUDA_ARCH
ARG JOBS_AND_THREADS=8

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates curl git \
    && rm -rf /var/lib/apt/lists/*

# Install UV (build stage only - not carried to runtime)
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV UV_NO_DEV=1
ENV PATH="/root/.local/bin/:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

# Compile flash-attn from source (no binary wheel exists).
# All .o / .ptx / .cubin build artefacts and the UV cache stay in THIS stage only
# and are automatically discarded when the final runtime stage is assembled.
RUN export MAX_JOBS=${JOBS_AND_THREADS} && \
    export NVCC_THREADS=${JOBS_AND_THREADS} && \
    export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" && \
    export FLASH_ATTENTION_FORCE_BUILD="TRUE" && \
    export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" && \
    export FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" && \
    export FLASH_ATTN_CUDA_ARCHS="$(echo ${CUDA_ARCH} | tr -d '.')" && \
    uv sync --extra flash --locked

# ── CPU builder ────────────────────────────────────────────────────────────────
FROM python:3.12-trixie AS builder-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

ENV UV_NO_DEV=1
ENV UV_PYTHON_PREFERENCE=only-system
ENV PATH="/root/.local/bin/:$PATH"

WORKDIR /app
COPY pyproject.toml uv.lock ./

RUN uv sync --locked

# ── Alias: select the right builder under a fixed name ───────────────────────
# Docker does not support build args in COPY --from, but does support them in
# FROM.  Creating this alias stage gives COPY a stable literal name to target.
ARG BASE_IMAGE
FROM builder-${BASE_IMAGE} AS builder

# ── CUDA runtime ───────────────────────────────────────────────────────────────
# cudnn-runtime is ~4 GB lighter than cudnn-devel; no compiler/headers.
FROM nvidia/cuda:12.9.1-cudnn-runtime-ubuntu24.04 AS runtime-cuda

# ── CPU runtime ────────────────────────────────────────────────────────────────
# python:slim contains only the interpreter + minimal OS; no compiler toolchain.
FROM python:3.12-slim-trixie AS runtime-cpu

# ── Final stage ────────────────────────────────────────────────────────────────
ARG BASE_IMAGE
FROM runtime-${BASE_IMAGE} AS final

WORKDIR /app

# Copy only the pre-built virtual environment from the appropriate builder.
# Nothing else from the builder (UV cache, build tools, CUDA devel files,
# flash-attn .o/.ptx artefacts, apt lists) is carried over.
COPY --from=builder /app/.venv /app/.venv

COPY main.py ./
COPY provider ./provider
COPY repository ./repository
COPY router ./router
COPY services ./services

# Use the venv directly - no UV needed at runtime
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV=/app/.venv

CMD ["fastapi", "run", "main.py"]