"""The ``Chunker`` family base.

``docs/specification/subclasses.md`` (a historical design record — see that
file's header): ``Chunker`` is the ``DocumentStep`` every concrete chunker
(``RecursiveCharacterTextSplitter``, ``CharacterTextSplitter``,
``DoclingChunker``, ``ChunkerRouter``, and ``SemanticChunker``) extends. Its one
job beyond ``DocumentStep`` itself is owning chunk identity — the same
property the ``Chunker`` contract demands: a subclass never stamps ``id`` or
``metadata["chunk_index"]``
itself. It implements :meth:`Chunker.split`, a pure generator of chunk texts,
and this base class turns that into fully identified
:class:`~fastrag.models.Document` chunks, whatever :meth:`~fastrag.ingestion.steps.chunking.base.Chunker.split`
does. Synchronous chunkers implement ``split``; model-backed chunkers
override ``asplit`` for asynchronous calls. Both use this base class for
chunk identity, metadata and events.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Iterator, Mapping
from typing import Any

from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion.events import DocumentChunked
from fastrag.ingestion.steps.base import DocumentStep, ExecutionContext
from fastrag.models import ContentFormat, Document

#: One piece :meth:`Chunker.split` yields: either plain text (the historical
#: contract, still what :class:`~fastrag.ingestion.steps.chunking.recursive.RecursiveCharacterTextSplitter`
#: and :class:`~fastrag.ingestion.steps.chunking.character.CharacterTextSplitter`
#: yield), or a ``(text, metadata)`` pair for a chunker whose structure
#: carries information plain text can't — see :meth:`Chunker.split`.
SplitPiece = str | tuple[str, Mapping[str, Any]]


def structural_chunk_id(parent_id, index, content, metadata, config):
    """Versioned structural identity; runtime diagnostics are excluded."""
    from fastrag.ingestion.steps.chunking.structure import build_embedding_text

    keys = (
        "document_id",
        "position",
        "parent_section_id",
        "source_section_ids",
        "heading_ids",
        "heading_path",
        "heading_levels",
        "displayed_heading_ids",
        "source_spans",
        "retained_intervals",
        "serialization_map",
        "repeated_context",
        "overlap_spans",
        "embedding_text_version",
        "chunk_type",
        "source_format",
        "tokenizer_identity",
        "atomic_structure_lost",
        "overflow",
    )
    return content_hash(
        json.dumps(
            {
                "version": "structural-id:v1",
                "parent": parent_id,
                "index": index,
                "content": content,
                "config": config,
                "embedding": build_embedding_text(content, metadata),
                "structure": {key: metadata[key] for key in keys},
            },
            sort_keys=True,
        )
    )


class Chunker(DocumentStep, ABC):
    r"""Splits one parsed document into the units that get embedded.

    Chunking is where chunk identity is decided, and this base class owns
    that decision so no implementation can get it wrong: every piece
    :meth:`~fastrag.ingestion.steps.chunking.base.Chunker.split` yields leaves :meth:`transform` as a
    :class:`~fastrag.models.Document` carrying a deterministic
    ``id`` (a stable function of the parent document's own id, the chunk's
    index, and its text) and ``metadata["chunk_index"]``, whatever
    :meth:`~fastrag.ingestion.steps.chunking.base.Chunker.split` does. Binding an upstream
    (``chunker(parsed_documents)``, via the inherited ``__call__``) and
    evaluating it (``async for document in chunker`` /
    :meth:`~fastrag.ingestion.steps.base.PipelineStep.run`) never splits
    eagerly — only consuming the resulting stream does.

    Caching (per parent document id and this chunker's own configuration) and
    retries are entirely :class:`~fastrag.ingestion.steps.base.DocumentStep`'s
    job, applied around :meth:`transform`; nothing here re-implements either.

    Writing one is a class and a generator::

        from collections.abc import Iterator

        from fastrag.models import Document
        from fastrag.ingestion.steps.chunking import Chunker


        class ParagraphChunker(Chunker):
            def __init__(self, *, max_chars: int = 400, cache=False, policy=None):
                super().__init__(cache=cache, policy=policy)
                self.max_chars = max_chars

            def config(self):
                return {"max_chars": self.max_chars}

            def split(self, document: Document) -> Iterator[str]:
                for paragraph in document.content.split("\\n\\n"):
                    yield paragraph[: self.max_chars]

    Every event a chunker emits is labelled ``"Chunking"``, inherited by
    every concrete chunker (``RecursiveCharacterTextSplitter``,
    ``CharacterTextSplitter``, ``ChunkerRouter``, ...) unless it declares an
    ``event_name`` of its own.
    """

    event_name = "Chunking"

    #: None preserves permissive custom subclasses; sets may include custom strings.
    accepted_content_formats: frozenset[str] | None = None

    def validate_input(self, item: Document) -> None:
        """Reject incompatible representations before splitting or cache replay."""
        actual = item.metadata.get("content_format", ContentFormat.TEXT)
        accepted = self.accepted_content_formats
        if accepted is not None and (
            not isinstance(actual, str) or actual not in accepted
        ):
            from fastrag.exceptions import ChunkingError

            raise ChunkingError(
                f"{self.name} cannot process document {item.id!r} with "
                f"content_format={actual!r}; accepts {sorted(accepted)!r}. "
                "Route to a compatible chunker (DoclingChunker for native DOCLING), "
                "convert with a parser, or register a structural normalizer."
            )

    #: Overrides the parent document's ``content_format`` in every chunk
    #: this chunker produces. ``None`` — the default — changes nothing:
    #: metadata inheritance in ``_build_chunk`` already carries the
    #: parent's ``content_format`` (and everything else in its metadata)
    #: forward unchanged. Set this on a chunker whose output format differs
    #: from its input's (e.g. one that flattens a
    #: :attr:`~fastrag.models.ContentFormat.DOCLING` document into Markdown
    #: chunks) so the chunks it produces describe themselves truthfully.
    content_format: ContentFormat | None = None

    @abstractmethod
    def split(self, document: Document) -> Iterator[SplitPiece]:
        """Split ``document.content``, yielding one piece per chunk.

        Pure computation, so a plain generator, never ``async``. A
        whitespace-only piece is dropped by the base class before it is ever
        numbered, so an implementation need not filter those itself. Yield
        nothing for a document with no usable text.

        Each yielded piece is either a plain ``str`` — the text of the chunk,
        as every chunker before this one yielded — or a ``(text, metadata)``
        tuple, for a chunker whose structure carries information plain text
        can't (e.g. :class:`~fastrag.ingestion.steps.chunking.docling.DoclingChunker`'s
        ``headings``/``page_no``). ``metadata`` is merged into the built
        chunk's metadata on top of the parent document's own — so it may
        override an inherited key like ``title`` — but ``parent_id`` and
        ``chunk_index`` are set afterwards and always win, and
        :attr:`content_format` (when set) is applied last and always wins
        too: neither is ever something a chunker's own ``split`` can shadow.

        Raises:
            fastrag.exceptions.ChunkingError: The document could not be split.
        """

    async def asplit(self, document: Document) -> AsyncIterator[SplitPiece]:
        """Yield split pieces asynchronously; override for model-backed chunking.

        The default delegates to the synchronous ``split`` hook. Both hooks
        share chunk identity, metadata, caching and events through ``transform``.
        """
        for piece in self.split(document):
            yield piece

    async def transform(
        self, item: Document, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Build every chunk of ``item``, numbering them as they are produced.

        Numbering lives on this one call's stack, not on anything shared by
        the chunker instance, so two documents chunked concurrently by the
        same chunker can never share a counter — each call to
        :meth:`transform` starts its own counter from zero.

        This method only runs on a cache miss — a cached parent document's
        chunks are replayed by :class:`~fastrag.ingestion.steps.base.DocumentStep`
        without calling :meth:`transform` at all. Emitting
        :class:`~fastrag.ingestion.events.DocumentChunked` is
        ``DocumentStep._on_produced``'s job,
        not this method's, precisely so the event fires the same way for a
        cache hit as for a fresh split.
        """
        self.validate_input(item)
        chunks = []
        index = 0
        async for piece in self.asplit(item):
            text, extra = piece if isinstance(piece, tuple) else (piece, None)
            if stripped := text.strip():
                chunks.append(self._build_chunk(item, index, stripped, extra))
                index += 1
        for chunk in chunks:
            yield chunk

    async def _on_produced(
        self, item: Document, produced: list[Document], context: ExecutionContext
    ) -> None:
        """Emit one :class:`~fastrag.ingestion.events.DocumentChunked` per parent document.

        Called once per upstream document regardless of whether ``produced``
        came from the cache or a fresh :meth:`transform` call, so
        ``chunk.cached`` on each chunk in ``produced`` is trustworthy either
        way — see ``DocumentStep._on_produced``.
        """
        self._order_output(produced)
        await self.validate_output(item, produced, context)
        await context.emit(
            DocumentChunked(
                document=item,
                chunks=produced,
                step=self.name,
                event_name=self.event_name,
            )
        )

    @staticmethod
    def _order_output(produced: list[Document]) -> None:
        """Restore sibling order independently of custom validation overrides.

        Legacy cache snapshots may have gaps from discarded blank pieces.
        Those indices remain valid, but ambiguous or malformed indices do not.
        """
        from fastrag.exceptions import ChunkingError

        indices = [chunk.metadata.get("chunk_index") for chunk in produced]
        if any(type(index) is not int or index < 0 for index in indices) or len(
            set(indices)
        ) != len(indices):
            raise ChunkingError("Invalid chunk indices in resolved output")
        produced.sort(key=lambda chunk: chunk.metadata["chunk_index"])

    async def validate_output(self, item, produced, context) -> None:
        """Validate resolved output, including replay; routers forward this hook."""
        return None

    def _build_chunk(
        self,
        document: Document,
        index: int,
        content: str,
        extra: Mapping[str, Any] | None = None,
    ) -> Document:
        """Build one identified chunk of ``document``, carrying its provenance forward.

        The chunk inherits the parent document's metadata (so a title found
        while parsing survives into every chunk, and so does
        ``content_format`` — a chunk is the same kind of string as the
        document it was split from, unless :attr:`content_format` says
        otherwise), overlaid with ``extra`` (the per-chunk metadata a
        ``(text, metadata)`` piece from :meth:`split` carried, if any), then
        ``parent_id`` (the parent document's own ``id``) and ``chunk_index``
        (this chunk's position among its siblings) — set last, so neither
        ``extra`` nor the inherited metadata can shadow them. ``id`` is a
        content hash of the parent id, the index, and the chunk's own text —
        deterministic and stable across machines and processes for the same
        input.
        """
        metadata = {
            **document.metadata,
            **(extra or {}),
            "parent_id": document.id,
            "chunk_index": index,
        }
        if self.content_format is not None:
            metadata["content_format"] = self.content_format
        return Document(
            id=structural_chunk_id(
                document.id, index, content, metadata, self.config_hash
            )
            if extra and extra.get("embedding_text_version")
            else content_hash(f"{document.id}:{index}:{content}"),
            source_id=document.source_id,
            content=content,
            step_config_hash=self.config_hash,
            metadata=metadata,
        )
