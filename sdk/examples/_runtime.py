"""Execution-time settings and output checks shared by numbered examples."""

from collections.abc import AsyncIterable, AsyncIterator
from dataclasses import dataclass
import os
from typing import TypeVar

from fastrag.ingestion import Pipeline, Policy
from fastrag.ingestion.events import DocumentSaved, ItemFailed, StepFailed
from fastrag.ingestion.steps.fetching.renderers import HttpRenderer, PlaywrightRenderer

T = TypeVar("T")


@dataclass(frozen=True)
class ExampleConfig:
    sitemap_url: str = "https://agrospai.udl.cat/en/sitemap.xml"
    web_url: str = "https://docs.pontus-x.eu/"
    database_url: str = "postgresql://postgres:postgres@localhost:5433/fastrag"
    cache_schema: str = "fastrag_cache"
    milvus_uri: str = "http://localhost:19530"
    collection: str = "fastrag_documents"
    embedding_model: str = "minishlab/potion-base-8M"
    max_urls: int = 200
    question: str = "What is AgrospAI?"
    chat_model: str = "llama3.2:3b"
    chat_base_url: str = "http://localhost:11434/v1"
    rerank_model: str = "bge-reranker-v2-m3"
    rerank_base_url: str = "http://localhost:8000/v1"
    route_pattern: str = r"https?://[^/]+/(?:[^/]+/)?docs/.*"
    renderer: str = "http"
    timeout: float = 60.0

    def __post_init__(self):
        if self.max_urls < 1 or self.timeout <= 0:
            raise ValueError("max_urls and timeout must be positive")
        if self.renderer not in {"http", "browser"}:
            raise ValueError("FASTRAG_RENDERER must be http or browser")
        if not all(
            (self.collection.strip(), self.cache_schema.strip(), self.question.strip())
        ):
            raise ValueError("collection, cache schema and question must be nonempty")

    def make_renderer(self):
        return HttpRenderer() if self.renderer == "http" else PlaywrightRenderer()

    def policy(self):
        return Policy(timeout=self.timeout, retries=0, on_error="raise")


def config_from_env() -> ExampleConfig:
    """Read nonsecret overrides when main executes, never during import."""
    names = {
        "sitemap_url": "FASTRAG_SITEMAP_URL",
        "web_url": "FASTRAG_WEB_URL",
        "database_url": "FASTRAG_DATABASE_URL",
        "cache_schema": "FASTRAG_CACHE_SCHEMA",
        "milvus_uri": "FASTRAG_MILVUS_URI",
        "collection": "FASTRAG_COLLECTION",
        "embedding_model": "FASTRAG_LOCAL_EMBEDDING_MODEL",
        "question": "FASTRAG_QUESTION",
        "chat_model": "FASTRAG_CHAT_MODEL",
        "chat_base_url": "FASTRAG_CHAT_BASE_URL",
        "rerank_model": "FASTRAG_RERANKER_MODEL",
        "rerank_base_url": "FASTRAG_RERANKER_BASE_URL",
        "route_pattern": "FASTRAG_ROUTE_PATTERN",
        "renderer": "FASTRAG_RENDERER",
    }
    values = {
        field: os.environ[env] for field, env in names.items() if env in os.environ
    }
    return ExampleConfig(
        **values,
        max_urls=int(os.environ.get("FASTRAG_MAX_URLS", "200")),
        timeout=float(os.environ.get("FASTRAG_TIMEOUT", "60")),
    )


async def require_output(items: AsyncIterable[T]) -> AsyncIterator[T]:
    """Fail if the actual lazy step exhausts without output."""
    count = 0
    async for item in items:
        count += 1
        yield item
    if count == 0:
        raise RuntimeError("Example produced no output")


class CheckedPipeline(Pipeline):
    """Keep normal Pipeline events while requiring actual successful saves."""

    async def _events(self):
        saved = 0
        async for event in super().__aiter__():
            if isinstance(event, (ItemFailed, StepFailed)):
                raise RuntimeError(f"Example pipeline failed: {event}")
            if isinstance(event, DocumentSaved):
                saved += 1
            yield event
        if saved == 0:
            raise RuntimeError("Example pipeline produced no output at its destination")

    def __aiter__(self):
        return self._events()
