# BASE_IMAGE: "cpu" or "cuda" - selects which base image to use
ARG BASE_IMAGE=cpu
ARG CUDA_ARCH
ARG JOBS_AND_THREADS=8

# Select base image based on BASE_IMAGE arg
FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04 AS base-cuda
FROM python:3.12-trixie AS base-cpu
FROM base-${BASE_IMAGE} AS base

# For local dev:  docker build --build-arg BASE_IMAGE=cuda --build-arg CUDA_ARCH="8.6" -t ekh:local-8.6 .
ARG BASE_IMAGE
# Version that must match the GPU you are running this service on --> https://developer.nvidia.com/cuda/gpus
ARG CUDA_ARCH
ARG JOBS_AND_THREADS

RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates git

# Install UV
ADD https://astral.sh/uv/install.sh /uv-installer.sh
RUN sh /uv-installer.sh && rm /uv-installer.sh

# UV: Ensure the installed binary is on the `PATH`
ENV UV_NO_DEV=1
ENV PATH="/root/.local/bin/:$PATH"
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="$CUDA_HOME/bin:$PATH"

WORKDIR /app

COPY main.py ./
COPY provider ./provider
COPY repository ./repository
COPY router ./router
COPY services ./services
COPY pyproject.toml uv.lock ./

# Build with or without flash-attn based on BASE_IMAGE
# FLASH_ATTN_CUDA_ARCHS needs shell expansion, so compute it inline
RUN if [ "$BASE_IMAGE" = "cuda" ]; then \
      export CUDA_HOME=/usr/local/cuda && \
      export MAX_JOBS=${JOBS_AND_THREADS} && \
      export NVCC_THREADS=${JOBS_AND_THREADS} && \
      export TORCH_CUDA_ARCH_LIST="${CUDA_ARCH}" && \
      export FLASH_ATTENTION_FORCE_BUILD="TRUE" && \
      export FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE" && \
      export FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE" && \
      export FLASH_ATTN_CUDA_ARCHS="$(echo ${CUDA_ARCH} | tr -d '.')" && \
      uv sync --extra flash --locked; \
    else \
      uv sync --locked; \
    fi

CMD ["uv", "run", "fastapi", "run", "main.py"]