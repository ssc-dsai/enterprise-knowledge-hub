FROM nvidia/cuda:12.9.1-cudnn-devel-ubuntu24.04
#FROM python:3.12.12-trixie

# For local dev:  docker build --build-arg UV_ARGS="--extra flash" --build-arg CUDA_ARCH="8.6" -t ekh:local-8.6 .
ARG UV_ARGS
# CUDA compute capabilities - see: https://developer.nvidia.com/cuda-gpus (use version found there for your GPU)
ARG CUDA_ARCH

# install uv (from https://docs.astral.sh/uv/guides/integration/docker/#installing-uv)
# The installer requires curl (and certificates) to download the release archive
# git is required for flash-attn to fetch CUTLASS submodule during build
RUN apt-get update && apt-get install -y --no-install-recommends build-essential curl ca-certificates git

# Download the latest installer
ADD https://astral.sh/uv/install.sh /uv-installer.sh

# Run the installer then remove it
RUN sh /uv-installer.sh && rm /uv-installer.sh

# Ensure the installed binary is on the `PATH`
ENV UV_NO_DEV=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="/root/.local/bin/:$CUDA_HOME/bin:$PATH"

# flash-attn build configuration (see: https://github.com/Dao-AILab/flash-attention)
# MAX_JOBS: Limit parallel ninja jobs to avoid OOM during compilation
# NVCC_THREADS: Limit nvcc threads per job
# TORCH_CUDA_ARCH_LIST: Target GPU architectures (used by PyTorch cpp_extension)
ENV MAX_JOBS=4
ENV NVCC_THREADS=2
ENV TORCH_CUDA_ARCH_LIST=${CUDA_ARCH}

ENV FLASH_ATTENTION_FORCE_BUILD="TRUE"
ENV FLASH_ATTENTION_FORCE_CXX11_ABI="FALSE"
ENV FLASH_ATTENTION_SKIP_CUDA_BUILD="FALSE"

WORKDIR /app

COPY main.py ./
COPY provider ./provider
COPY repository ./repository
COPY router ./router
COPY services ./services

COPY pyproject.toml uv.lock ./

# need extra work for llama_cpp stuff. see pyproject.toml for details.
# FLASH_ATTN_CUDA_ARCHS needs shell expansion, so compute it inline
RUN FLASH_ATTN_CUDA_ARCHS="$(echo ${CUDA_ARCH} | tr -d '.')" uv sync $UV_ARGS --locked

# ENV VIRTUAL_ENV=/app/.venv \
# 	PATH=/app/.venv/bin:$PATH

CMD ["uv", "run", "fastapi", "run", "main.py"]