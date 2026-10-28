"""Native chunking of a serialized ``DoclingDocument``, via `Docling <https://docling-project.github.io/docling/>`_'s own chunkers.

:class:`~fastrag.ingestion.steps.parsing.docling.DoclingParser` (task 011) can
put the structured document itself — a serialized ``DoclingDocument`` — in
``Document.content`` rather than flattened Markdown.
:class:`DoclingChunker` is the chunker that knows what to do with it: it
deserializes ``content`` back into a ``DoclingDocument`` and splits on the
real structure (headings, tables, list items) via one of Docling's own
chunkers, instead of re-splitting text that structure was already flattened
out of.

Which Docling chunker drives the split is a constructor choice
(``docling_core.transforms.chunker.BaseChunker``, any of
``HierarchicalChunker``, ``HybridChunker``, ``LineBasedTokenChunker``, or a
caller's own subclass). The default needs neither a tokenizer nor a model
download; the other two do — see ``DoclingChunker.__init__``.

Every emitted chunk is plain text again (``content_format`` is
:attr:`~fastrag.models.ContentFormat.TEXT` — see the class docstring for
why), embeddable and storable by every existing downstream step with no
further translation, carrying ``headings`` and (when Docling reports one)
``page_no`` in its metadata alongside the ``parent_id``/``chunk_index`` the
:class:`~fastrag.ingestion.steps.chunking.base.Chunker` base already owns.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import TYPE_CHECKING, Any, Literal

from fastrag.exceptions import ChunkingError, ConfigurationError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.chunking.base import Chunker, SplitPiece
from fastrag.models import ContentFormat, Document

if TYPE_CHECKING:
    from docling_core.transforms.chunker import BaseChunker
    from docling_core.types.doc.document import DoclingDocument as _DLDocument

try:
    from docling_core.transforms.chunker import (
        HierarchicalChunker as _HierarchicalChunkerImpl,
    )
    from docling_core.types.doc.document import (
        DoclingDocument as _DoclingDocumentImpl,
    )

    _DOCLING_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _DOCLING_AVAILABLE = False

#: The Docling classes, or unusable placeholders when the optional "docling"
#: extra is missing; ``DoclingChunker.__init__`` refuses to run until
#: they exist.
HierarchicalChunker: Any = _HierarchicalChunkerImpl if _DOCLING_AVAILABLE else None
DoclingDocument: Any = _DoclingDocumentImpl if _DOCLING_AVAILABLE else None

MISSING_EXTRA = (
    'DoclingChunker needs the "docling" extra: pip install "fastrag-ai[docling]"'
)


class DoclingChunker(Chunker):
    r"""Split a serialized ``DoclingDocument`` with one of Docling's own chunkers.

    Handed anything whose ``metadata["content_format"]`` is not
    :attr:`~fastrag.models.ContentFormat.DOCLING`, it refuses rather than
    chunking the wrong thing — route with a
    :class:`~fastrag.ingestion.filters.ContentFormatFilter` (see the module
    example below).

    Each chunk's text is Docling's own *contextualized* serialization
    (``chunker.contextualize(chunk)``) — the chunk's heading path travels
    with its text, rather than being lost the moment the chunk is embedded on
    its own. That serialization is a delimiter-joined concatenation of the
    heading path and the chunk's own (sometimes list-flavoured, never
    table-flavoured — a table row comes out as ``"Alice, Value = 42"``, not a
    Markdown table) text, not guaranteed-valid Markdown, so every chunk this
    class produces carries ``content_format =
    ~fastrag.models.ContentFormat.TEXT``, not ``.MARKDOWN``.

    .. code-block:: python

        from fastrag.ingestion.filters import ContentFormatFilter
        from fastrag.ingestion.steps.chunking import ChunkerRouter, DoclingChunker
        from fastrag.ingestion.steps.parsing import ParserRouter, DoclingParser, office_filter
        from fastrag.models import ContentFormat

        parser = ParserRouter(cache=cache).route(
            office_filter, DoclingParser(output_format="docling")
        )
        chunker = ChunkerRouter(cache=cache).route(
            ContentFormatFilter(include_formats=[ContentFormat.DOCLING]),
            DoclingChunker(),
        )
    """

    accepted_content_formats = frozenset(["docling"])

    content_format = ContentFormat.TEXT

    def __init__(
        self,
        *,
        chunker: BaseChunker | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``DoclingChunker``.

        Args:
            chunker: The Docling chunker driving the split — any
                ``docling_core.transforms.chunker.BaseChunker`` instance.
                ``None`` (the default) builds a plain
                ``docling_core.transforms.chunker.HierarchicalChunker()``,
                the only one of the three Docling chunkers the user may want
                that needs **no tokenizer and no model download**: it walks
                the document's own item tree and serializes each item (or
                group of items) as one chunk. ``HybridChunker`` (further
                merges/splits ``HierarchicalChunker``'s output to fit a token
                budget) and ``LineBasedTokenChunker`` (splits on line
                boundaries up to a token budget) are opt-in for exactly this
                reason: **both pull a tokenizer the first time they're used**
                (a Hugging Face model by default), which can mean a network
                call and a multi-hundred-megabyte download the first time —
                construct one yourself, with whatever tokenizer you already
                have, and pass it in here.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.

        Raises:
            fastrag.exceptions.ConfigurationError: The optional "docling"
                extra is not installed.
        """
        if not _DOCLING_AVAILABLE:
            raise ConfigurationError(MISSING_EXTRA)
        super().__init__(cache=cache, policy=policy)
        self._chunker: Any = chunker if chunker is not None else HierarchicalChunker()

    def config(self) -> Mapping[str, Any]:
        """This chunker's cache-identity configuration.

        Only the concrete Docling chunker's class name — switching from
        ``HierarchicalChunker`` to a caller-supplied ``HybridChunker`` (or
        vice versa) invalidates the cache. A ``docling_core`` ``BaseChunker``
        is a Pydantic model that (for ``HybridChunker``/
        ``LineBasedTokenChunker``) carries a ``tokenizer`` field and a
        ``serializer_provider`` field, neither of which is JSON-serializable
        in general (confirmed: ``HierarchicalChunker().model_dump(mode="json")``
        raises ``PydanticSerializationError`` on its own default
        ``serializer_provider``), so this cannot fingerprint the chunker's
        full configuration the way e.g.
        :meth:`~fastrag.ingestion.steps.chunking.recursive.RecursiveCharacterTextSplitter.config`
        does. A caller who reconfigures a chunker of the *same* class between
        runs against the same cache (e.g. a different ``max_tokens`` on a
        ``HybridChunker``) should pass ``cache=False`` or a fresh cache
        rather than rely on this to notice.
        """
        return {"chunker": type(self._chunker).__name__}

    def split(self, document: Document) -> Iterator[SplitPiece]:
        """Deserialize ``document.content`` and chunk it with the configured Docling chunker.

        Each yielded piece is ``(chunker.contextualize(chunk), metadata)``,
        where ``metadata`` carries ``headings`` (``list[str]``, possibly
        empty — Docling reports ``None`` for a chunk with no heading
        ancestor, normalized here to ``[]``) and, when at least one of the
        chunk's source items carries Docling provenance (paginated formats —
        confirmed for XLSX; not reported for a plain Markdown/text source),
        ``page_no`` (the first one found).

        Raises:
            fastrag.exceptions.ChunkingError: ``document`` is not
                :attr:`~fastrag.models.ContentFormat.DOCLING` — route with
                ``ContentFormatFilter(include_formats=[ContentFormat.DOCLING])``
                instead of handing this chunker anything else.
        """
        self.validate_input(document)
        content_format = document.metadata.get("content_format")
        if content_format != ContentFormat.DOCLING:
            raise ChunkingError(
                f"DoclingChunker got a document with content_format={content_format!r}, "
                f"not {ContentFormat.DOCLING!r} — route only DOCLING-format documents to "
                "it, e.g. via ContentFormatFilter(include_formats=[ContentFormat.DOCLING])"
            )
        try:
            dl_doc: _DLDocument = DoclingDocument.model_validate_json(document.content)
        except ValueError as error:
            raise ChunkingError(
                f"DoclingChunker received malformed Docling content in {document.id!r}: {error}"
            ) from error
        for chunk in self._chunker.chunk(dl_doc):
            metadata: dict[str, Any] = {"headings": list(chunk.meta.headings or [])}
            page_no = _page_no(chunk)
            if page_no is not None:
                metadata["page_no"] = page_no
            yield self._chunker.contextualize(chunk), metadata


def _page_no(chunk: Any) -> int | None:
    """The first page number any of ``chunk``'s source items reports, if any."""
    for item in chunk.meta.doc_items:
        for prov in item.prov:
            return prov.page_no
    return None
