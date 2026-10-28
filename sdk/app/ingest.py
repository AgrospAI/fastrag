"""Populate Milvus from ``pipeline.py``.

Idempotent: ``PostgresCache`` fingerprints each step's inputs, so rerunning
this against an unchanged site fetches, parses, chunks and embeds nothing new
— every document reports ``cached=True`` and nothing is duplicated in Milvus.

    docker compose run --rm backend uv run python -m app.ingest
"""

from __future__ import annotations

import asyncio
import sys

from app import pipeline as pl
from app.config import MissingEnvironmentVariable, load_settings
from fastrag.ingestion import Pipeline
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.logger import ConsoleLogger


async def main() -> None:
    """Read the environment, build the pipeline, and run it to completion."""
    settings = load_settings()

    async with (
        PostgresCache(url=settings.database_url) as cache,
        pl.build_vector_store(settings) as vector_store,
    ):
        source = pl.build_source(cache)
        parser = pl.build_parser(cache)
        chunker = pl.build_chunker(cache)
        embedder = pl.build_embedder(cache)

        parsed_documents = parser(source)
        chunked_documents = chunker(parsed_documents)
        embedded_documents = embedder(chunked_documents)
        saved_documents = vector_store.save(embedded_documents)

        pipeline = Pipeline(saved_documents)
        await ConsoleLogger(pipeline).run()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except MissingEnvironmentVariable as error:
        print(f"error: {error}", file=sys.stderr)
        sys.exit(1)
