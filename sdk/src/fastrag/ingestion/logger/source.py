"""Inspect fetched source bytes without losing binary content."""

from __future__ import annotations

import base64
from typing import Any

from fastrag.ingestion.events import DocumentFetched, PipelineEvent
from fastrag.ingestion.logger._inspection import _InspectionLogger


class SourceLogger(_InspectionLogger):
    """Display unique fetched sources with complete metadata and content.

    Accepts ``pipeline``, ``format="rich"`` or ``"jsonl"``, ``stream``,
    ``configuration`` and optional ``stages``. UTF-8 content is displayed as
    literal text; other content is displayed as base64. JSON Lines always
    includes ``content_base64`` for exact byte recovery. ``source_count``
    counts unique source IDs, deduplicating events from Sources aggregates.
    Counters reset on each run; failures propagate after a partial summary.

    Set ``content_lines=N`` to show only the first and last N text lines,
    separated by an omitted-line marker. Short content is shown in full.
    The default ``None`` shows everything. This applies to Rich and JSONL
    text output; metadata, base64 bytes and size statistics remain complete.
    ``content_chars=N`` additionally caps preview text at N characters, useful
    for minified HTML with very few lines. Both limits default to ``None``.

    ``metadata_only=True`` omits text, ``content_base64`` and embeddings,
    keeping metadata and size statistics. Non-text content is trimmed to a
    short base64 preview (``content_truncated``) unless ``show_binary=True``.
    """

    # content_lines limits displayed text; statistics always use full content.
    label = "Source inspection"

    def _records(self, event: PipelineEvent) -> list[dict[str, Any]]:
        if not isinstance(event, DocumentFetched):
            return []
        source = event.source
        try:
            text = source.content.decode("utf-8")
        except UnicodeDecodeError:
            text = None
        return [
            {
                "type": "source",
                "id": source.id,
                "source_id": source.id,
                "uri": source.uri,
                "content_type": source.content_type,
                "content_hash": source.content_hash,
                "etag": source.etag,
                "last_modified": source.last_modified,
                "last_validated": source.last_validated,
                "step": event.step,
                "event_name": event.event_name,
                "cached": source.cached,
                "metadata": source.metadata,
                "length_bytes": len(source.content),
                "length": len(text) if text is not None else 0,
                "word_count": len(text.split()) if text is not None else 0,
                "line_count": len(text.splitlines()) if text is not None else 0,
                "text": text,
                "content_base64": base64.b64encode(source.content).decode("ascii"),
            }
        ]
