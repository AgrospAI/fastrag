# syntax=docker.io/docker/dockerfile:1

# Build from sdk/, not from app/ — every path below is relative to that
# context: docker-compose.yaml's `backend` service builds with `context: .`,
# `dockerfile: app/Dockerfile`. Follows the same uv-in-Docker idiom as
# tests/Dockerfile.e2e and docs/Dockerfile.docs: dependencies first (cached
# across edits to app/ or src/), then source, then this task's own package.
FROM ghcr.io/astral-sh/uv:python3.13-bookworm-slim AS deps

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /app

# Only the extras the reference pipeline (app/pipeline.py) actually imports:
# html for ParserRouter's HTML branch, chunkers for RecursiveCharacterTextSplitter,
# model2vec for the embedder, milvus for the vector store, openai for
# OpenAIChatModel (Ollama speaks its wire protocol), serving for create_app.
# Not --all-extras: docling/datasets/geoip are not on this path.
ARG APP_EXTRAS="--extra html --extra chunkers --extra model2vec --extra milvus --extra openai --extra serving"
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --locked --no-install-project $APP_EXTRAS

COPY src ./src
COPY app ./app
RUN uv sync --locked $APP_EXTRAS

# Non-root: the standing rule for the image, not just this stage's `runtime`
# target — the dev container also bind-mounts as this user, so files it
# writes back (e.g. __pycache__, if PYTHONDONTWRITEBYTECODE were ever
# dropped) stay owned by the host's own user on a typical single-user
# Docker Desktop setup.
#
# Both cache dirs live under /app (already chowned to fastrag) rather than
# under $HOME (/home/fastrag): docker-compose.yaml mounts a named volume at
# the huggingface path, and Docker only inherits an existing path's ownership
# into a freshly-created named volume when that path already exists — with
# correct ownership — in the image, otherwise it (and its parent directories)
# get created root-owned, which then also blocks uv's own cache directory
# from being created alongside it.
RUN useradd --create-home --uid 1000 fastrag \
    && mkdir -p /app/.cache/huggingface /app/.cache/uv \
    && chown -R fastrag:fastrag /app
ENV UV_CACHE_DIR=/app/.cache/uv \
    HF_HOME=/app/.cache/huggingface \
    PYTHONDONTWRITEBYTECODE=1
USER fastrag

# -- dev: hot reload against bind-mounted app/ and src/ (docker-compose.yaml's
# `backend` service, dev profile). uvicorn's --factory + --reload spawns a
# subprocess that reimports app.serve on every change under --reload-dir;
# app.serve.build_app_sync() is the synchronous factory that makes that
# possible for an app whose real resources are opened async — see its
# docstring for why this path is dev-only.
FROM deps AS dev
EXPOSE 8000
# Same probe as `runtime` below: task 010's root-level compose depends_on's
# other services (prometheus) on `backend: condition: service_healthy`, and
# that condition applies to whichever target is actually running — without
# this, swapping in `dev` via docker-compose.dev.yaml makes Compose refuse
# to start anything depending on `backend` at all ("no healthcheck
# configured"), even though the dev server itself serves /healthz exactly
# the same way.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=3)"
CMD ["uv", "run", "uvicorn", "app.serve:build_app_sync", "--factory", "--reload", \
     "--reload-dir", "app", "--reload-dir", "src", \
     "--host", "0.0.0.0", "--port", "8000"]

# -- runtime: what ships. `ingest` and `serve` are both `uv run python -m
# app.<name>`; the image's default CMD is serve — ingest is a one-off
# (`docker compose run --rm backend uv run python -m app.ingest`).
FROM deps AS runtime
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request as u; u.urlopen('http://localhost:8000/healthz', timeout=3)"
CMD ["uv", "run", "python", "-m", "app.serve"]
