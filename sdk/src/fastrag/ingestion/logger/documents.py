"""Inspect documents produced by parsing, chunking, embedding or indexing."""

from __future__ import annotations

from typing import Any

from fastrag.ingestion.events import (
    DocumentChunked,
    DocumentEmbedded,
    DocumentParsed,
    DocumentSaved,
    PipelineEvent,
)
from fastrag.ingestion.logger._inspection import _InspectionLogger
from fastrag.models import Document


class DocumentsLogger(_InspectionLogger):
    """Display complete text, metadata, embeddings, lineage and size statistics.

    Accepts ``pipeline``, ``format="rich"`` or ``"jsonl"``, ``stream``,
    ``configuration`` and ``stages``. By default only the pipeline root's
    stage is inspected; ``stages=()`` inspects none, and an explicit sequence
    such as ``("Parsing", "Chunking")`` inspects multiple stages. Streams
    without a root are inspected at every document stage. All failures are
    reported regardless of the stage selection. Rich requires the logging
    extra; JSON Lines needs no optional dependencies.

    ``document_count`` counts output documents (including chunks),
    ``chunk_count`` counts chunk records. Counters reset for each run.

    Set ``content_lines=N`` to show only the first and last N text lines,
    separated by an omitted-line marker. Short content is shown in full.
    The default ``None`` shows everything. This applies to Rich and JSONL
    text output; metadata, base64 bytes and size statistics remain complete.

    ``metadata_only=True`` omits text and embeddings, keeping metadata and
    size statistics. ``show_binary=True`` disables binary content trimming.
    """

    # content_lines limits displayed text; statistics always use full content.
    label = "Document inspection"
    _inspect_final_stage = True

    def _records(self, event: PipelineEvent) -> list[dict[str, Any]]:
        if not isinstance(
            event, (DocumentParsed, DocumentChunked, DocumentEmbedded, DocumentSaved)
        ):
            return []
        root_stage = getattr(getattr(self.pipeline, "root", None), "event_name", None)
        if (
            self._inspect_final_stage
            and self.stages is None
            and root_stage is not None
            and event.event_name != root_stage
        ):
            return []
        if isinstance(event, DocumentChunked):
            return [
                {
                    "type": "document_result",
                    "id": event.document.id,
                    "source_id": event.document.source_id,
                    "source_uri": event.document.metadata.get("source_uri", ""),
                    "step": event.step,
                    "event_name": event.event_name,
                    "length": len(event.document.content),
                    "chunks": len(event.chunks),
                },
                *[
                    self._document_record(
                        chunk,
                        event,
                        type="chunk",
                        parent_document_id=event.document.id,
                        position=f"{index}/{len(event.chunks)}",
                    )
                    for index, chunk in enumerate(event.chunks, 1)
                ],
            ]
        return [self._document_record(event.document, event)]

    @staticmethod
    def _document_record(
        document: Document, event: PipelineEvent, **extra: Any
    ) -> dict[str, Any]:
        return {
            "type": "document",
            "id": document.id,
            "source_id": document.source_id,
            "source_uri": document.metadata.get("source_uri", ""),
            "step": getattr(event, "step", ""),
            "event_name": getattr(event, "event_name", ""),
            "step_config_hash": document.step_config_hash,
            "cached": document.cached,
            "length": len(document.content),
            "length_bytes": len(document.content.encode("utf-8")),
            "word_count": len(document.content.split()),
            "line_count": len(document.content.splitlines()),
            "metadata": document.metadata,
            "text": document.content,
            "embedding": document.embedding,
            **extra,
        }
