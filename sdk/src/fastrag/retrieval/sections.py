"""Database-independent exact source context for structural retrieval hits."""

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Literal

from fastrag.models import Document
from fastrag.normalization import NormalizedDocument, SourceSpan


@dataclass(frozen=True, slots=True)
class SectionContext:
    """Original retained section syntax, with document-scoped provenance."""

    document_id: str
    section_id: str
    content: str
    source_spans: tuple[SourceSpan, ...]
    heading_path: tuple[str, ...]


class SectionIndex:
    """Keep source parents outside vector chunks and resolve merged sections."""

    def __init__(self, documents: Iterable[NormalizedDocument] = ()) -> None:
        """Build an index of validated, document-scoped parents."""
        self._documents: dict[str, NormalizedDocument] = {}
        for document in documents:
            self.add(document)

    def add(self, document: NormalizedDocument) -> None:
        """Register a validated parent; conflicting document IDs are rejected."""
        document.validate()
        old = self._documents.get(document.document.id)
        if old is not None and old != document:
            raise ValueError("Conflicting normalized document ID")
        self._documents[document.document.id] = document

    def resolve(
        self,
        chunk: Document,
        *,
        ancestors: bool = False,
        missing: Literal["raise", "skip"] = "raise",
    ) -> tuple[SectionContext, ...]:
        """Return all source sections in source order, optionally with ancestors."""
        if missing not in {"raise", "skip"}:
            raise ValueError("missing must be raise or skip")
        document_id = chunk.metadata.get("document_id", chunk.metadata.get("parent_id"))
        document = self._documents.get(document_id)
        if document is None:
            if missing == "skip":
                return ()
            raise KeyError(f"Missing normalized document {document_id}")
        nodes = {n.id: n for n in document.walk()}
        selected = set()
        for ident in chunk.metadata.get(
            "source_section_ids", [chunk.metadata.get("parent_section_id")]
        ):
            node = nodes.get(ident)
            if node is None or node.kind not in {"document", "section"}:
                if missing == "skip":
                    continue
                raise KeyError(f"Missing section {ident} in {document_id}")
            selected.add(node.id)
            while ancestors and node.parent_id:
                node = nodes[node.parent_id]
                if node.kind == "section":
                    selected.add(node.id)
        result = []
        for node in document.walk():
            if node.id not in selected:
                continue
            path = []
            current = node
            while current:
                if current.kind == "section":
                    path.append(current.title or "")
                current = nodes.get(current.parent_id)
            result.append(
                SectionContext(
                    document.document.id,
                    node.id,
                    node.content,
                    node.spans,
                    tuple(reversed(path)),
                )
            )
        return tuple(result)
