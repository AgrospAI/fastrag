"""Prompt templates: retrieved documents plus a question to chat messages."""

from __future__ import annotations

from fastrag.query.prompts.templates import (
    DEFAULT_RAG_PROMPT,
    PromptTemplate,
    group_sources,
)

__all__ = ["DEFAULT_RAG_PROMPT", "PromptTemplate", "group_sources"]
