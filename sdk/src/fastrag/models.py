"""``Document``: the text-based unit shared across the pipeline and beyond.

A ``Document`` is what a :class:`~fastrag.ingestion.models.Source` becomes
once parsed, and what parsing, chunking, and embedding each hand to the next
step in turn. It is not an ingestion-only concept: retrieval, querying,
serving and benchmarking all consume it too, which is why it lives at the top
level rather than under :mod:`fastrag.ingestion`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class ContentFormat(StrEnum):
    """The shape of a :class:`Document`'s ``content`` string.

    Every parser stamps ``metadata["content_format"]`` with one of these
    members (see
    :meth:`~fastrag.ingestion.steps.parsing.base.Parser.build_document`), so
    a downstream step — a chunker choosing a Markdown-aware splitter, an
    embedding guardrail, a
    :class:`~fastrag.ingestion.filters.ContentFormatFilter` — can dispatch on
    what kind of string ``content`` actually is, rather than guessing from
    ``metadata["content_type"]`` (the *source*'s media type, not what the
    parser turned it into). Being a :class:`~enum.StrEnum`, every member
    serializes to a plain string wherever ``metadata`` is persisted — a
    cache's JSONB column, Milvus's JSON metadata field — with no special
    handling needed.
    """

    #: Plain, unstructured text. The default for a parser that declares no
    #: format of its own, and the format every document that predates this
    #: attribute is treated as.
    TEXT = "text"

    #: Markdown — headings, lists, tables, links preserved as syntax rather
    #: than flattened.
    MARKDOWN = "markdown"

    #: A serialized ``DoclingDocument``, for a parser whose output format
    #: depends on its own configuration rather than being fixed per class.
    DOCLING = "docling"

    #: Raw HTML markup.
    HTML = "html"

    #: reStructuredText source.
    RST = "rst"


@dataclass(frozen=True, slots=True)
class Document:
    """A text-based unit processed along the pipeline.

    The same type represents a whole parsed document and a chunk of one;
    ``metadata`` and ``step_config_hash`` carry lineage rather than a
    separate subtype per stage.

    Attributes:
        id: Unique identifier for this unit, typically a content or
            configuration hash.
        source_id: Identifier of the originating
            :class:`~fastrag.ingestion.models.Source`.
        content: Text content of the document or chunk.
        step_config_hash: Hash of the upstream transform's configuration,
            used for cache invalidation.
        embedding: Dense vector representation produced by an embedder step.
        metadata: Extended metadata tracking lineage, chunk offsets, URL
            filters, and parser outputs. Reserved keys include
            ``source_uri`` and ``content_type`` (the originating
            :class:`~fastrag.ingestion.models.Source`'s URI and media type)
            and ``content_format`` (a :class:`~fastrag.models.ContentFormat`
            value describing what kind of string ``content`` is) — all three
            stamped by
            :meth:`~fastrag.ingestion.steps.parsing.base.Parser.build_document`
            so no parser can forget them.
        cached: Whether this instance was served from the cache rather than
            freshly produced. Runtime-only: never persisted.
    """

    id: str
    source_id: str
    content: str
    step_config_hash: str = ""
    embedding: list[float] | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    cached: bool = False
