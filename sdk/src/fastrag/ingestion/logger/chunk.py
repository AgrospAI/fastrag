"""Compatibility import for the former chunk-only logger."""

from typing import Any

from fastrag.ingestion.events import DocumentChunked, PipelineEvent
from fastrag.ingestion.logger.documents import DocumentsLogger


class ChunkLogger(DocumentsLogger):
    """Compatibility logger for chunking events anywhere in the graph.

    New code should use DocumentsLogger. The summary's ``documents`` and
    ``parent_documents`` both count chunking inputs, as in the former API.
    """

    _inspect_final_stage = False

    def _summary_documents(self, parents: int) -> int:
        return parents

    def _records(self, event: PipelineEvent) -> list[dict[str, Any]]:
        if not isinstance(event, DocumentChunked):
            return []
        return super()._records(event)
