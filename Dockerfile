FROM python:3.13-slim

WORKDIR /app

# build-essential + libpq-dev: kept as a source-build fallback in case any
# pinned package ever lacks a prebuilt wheel for this Python version.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# uv instead of pip — same dependency resolution, materially faster installs
# and rebuilds. Matches the tooling used in the LLM Zoomcamp reference setup.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

COPY requirements.txt .

# ENV UV_HTTP_TIMEOUT=300


# Phase 4 pulls in sentence-transformers, which depends on torch. Installed
# separately, first, from PyTorch's CPU-only wheel index — otherwise the
# resolver pulls torch from default PyPI, which bundles several
# nvidia-*-cu12 packages (100-700MB each) that are dead weight in a
# container with no GPU. Installing the CPU build here first means the
# requirements.txt install below finds torch already satisfied.
#
# --no-cache (not --mount=type=cache) on both installs below — this repo's
# docker-compose doesn't have the buildx CLI plugin registered, so --mount
# fails with "the --mount option requires BuildKit" instead of falling back.
# Costs a re-download of wheels on every requirements.txt change; revisit
# once buildx is confirmed available.
RUN uv pip install --system --no-cache torch --index-url https://download.pytorch.org/whl/cpu

RUN uv pip install --system --no-cache -r requirements.txt

COPY . .

# Keeps the container alive so you can `docker compose exec app <command>`
# for whichever phase you're working on (ingestion script, dbt run,
# streamlit, eval harness, etc.) instead of hard-coding one entrypoint.
CMD ["tail", "-f", "/dev/null"]
