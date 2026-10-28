"""Querying: turning retrieved documents and a question into a generated answer.

:class:`QueryEngine` composes a :class:`~fastrag.retrieval.base.Retriever`, a
:class:`~fastrag.query.prompts.templates.PromptTemplate`, and a
:class:`~fastrag.query.llm.base.ChatModel` into one ``ask``/``stream``
surface. Chat models live under :mod:`fastrag.query.llm`, separately from
this package's own re-exports, the same way embedders live under
:mod:`fastrag.ingestion.steps.embedding`.
"""

from __future__ import annotations

from fastrag.query.engine import (
    Answer,
    DoneChunk,
    QueryEngine,
    SourcesChunk,
    StreamChunk,
    TokenChunk,
)
from fastrag.query.prompts.templates import (
    DEFAULT_RAG_PROMPT,
    PromptTemplate,
    group_sources,
)

__all__ = [
    "DEFAULT_RAG_PROMPT",
    "Answer",
    "DoneChunk",
    "PromptTemplate",
    "QueryEngine",
    "SourcesChunk",
    "StreamChunk",
    "TokenChunk",
    "group_sources",
]
