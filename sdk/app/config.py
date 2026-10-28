"""Read the environment once, here — never inside ``fastrag`` or ``pipeline.py``.

``fastrag.src`` has a standing rule that the library itself never reads
``os.environ``: every coordinate and credential is passed in by the caller.
This module is that caller's one place to do it. ``load_settings()`` is
called at the top of both entry points (``ingest.py``, ``serve.py``), before
anything async happens, so a missing required variable fails at startup with
a message naming it — not later, at first request or first fetch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass


class MissingEnvironmentVariable(RuntimeError):
    """Raised at startup when a required environment variable is unset."""

    def __init__(self, name: str) -> None:
        """Name the missing variable in the message, not just its absence."""
        super().__init__(
            f"missing required environment variable: {name} "
            "(see .env.example in sdk/app/)"
        )


def _required(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise MissingEnvironmentVariable(name)
    return value


def _origins(value: str) -> list[str]:
    return [origin.strip() for origin in value.split(",") if origin.strip()]


@dataclass(frozen=True, kw_only=True)
class Settings:
    """Everything the pipeline and the serving app need from the environment.

    Args:
        database_url: Postgres URL backing both the ingestion cache
            (``PostgresCache``) and the chat store (``PostgresChatStore``) —
            they use separate schemas in the same database, so one URL is
            enough.
        milvus_uri: Milvus's URI.
        chat_model: The chat model name passed to ``OpenAIChatModel``.
        chat_api_key: The API key for the chat endpoint. Ollama ignores the
            value but the OpenAI-compatible client still requires one.
        chat_base_url: The chat endpoint's base URL. ``None`` targets the
            real OpenAI API.
        allowed_origins: CORS origins the serving app accepts, from a
            comma-separated ``FASTRAG_ALLOWED_ORIGINS``.
        metrics_token: Optional bearer token guarding ``GET /metrics``.
            ``None`` leaves it unguarded.
    """

    database_url: str
    milvus_uri: str
    chat_model: str
    chat_api_key: str
    chat_base_url: str | None
    allowed_origins: list[str]
    metrics_token: str | None


def load_settings() -> Settings:
    """Read and validate every variable the pipeline needs.

    Raises:
        MissingEnvironmentVariable: A required variable is unset or empty.
    """
    return Settings(
        database_url=_required("FASTRAG_DATABASE_URL"),
        milvus_uri=_required("FASTRAG_MILVUS_URI"),
        chat_model=_required("FASTRAG_CHAT_MODEL"),
        chat_api_key=_required("FASTRAG_CHAT_API_KEY"),
        chat_base_url=os.environ.get("FASTRAG_CHAT_BASE_URL") or None,
        allowed_origins=_origins(_required("FASTRAG_ALLOWED_ORIGINS")),
        metrics_token=os.environ.get("FASTRAG_METRICS_TOKEN") or None,
    )
