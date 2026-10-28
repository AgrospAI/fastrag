"""Binary-format parsing via `Docling <https://docling-project.github.io/docling/>`_.

The shipped parsers (:class:`~fastrag.ingestion.steps.parsing.html.HtmlParser`,
:class:`~fastrag.ingestion.steps.parsing.default.DefaultParser`) only
understand text and HTML. Spreadsheets, Word documents, and PDFs need an
actual document converter, which is what
:class:`~fastrag.ingestion.steps.parsing.docling.DoclingParser` hands off to.

Two output shapes, selected by ``output_format``:

- ``"markdown"`` (the default) — the converted text, so any existing chunker
  keeps working unmodified and never needs to know Docling was involved.
- ``"docling"`` — the structured ``DoclingDocument`` itself, serialized as
  JSON, so a later native Docling chunker (task 012's ``DoclingChunker``) can
  split on the real structure instead of on flattened text.

``metadata["content_format"]`` (`~fastrag.models.ContentFormat.MARKDOWN` or
``.DOCLING``) is what tells a downstream step which shape it received.
"""

from __future__ import annotations

import asyncio
import dataclasses
import io
from collections.abc import AsyncIterator, Mapping, Sequence
from pathlib import PurePosixPath
from typing import TYPE_CHECKING, Any, Literal
from urllib.parse import urlparse

from fastrag.exceptions import ConfigurationError, ParseError
from fastrag.ingestion._hashing import content_hash
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.filters import MediaTypeFilter
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.models import ContentFormat, Document

if TYPE_CHECKING:
    from docling.datamodel.base_models import InputFormat
    from docling_core.types.doc.document import DoclingDocument

#: The Docling classes, resolved on first use by :func:`_load_docling`.
#: Importing Docling pulls in torch and transformers (seconds), which a
#: pipeline that never parses a binary document should not pay. They stay
#: ``None`` when the optional "docling" extra is missing;
#: `DoclingParser.__init__` refuses to run until they exist.
DocumentStream: Any = None
InputFormat: Any = None
DocumentConverter: Any = None
DocItemLabel: Any = None
PdfFormatOption: Any = None
PdfPipelineOptions: Any = None
create_pipeline_options_hash: Any = None
_loaded = False


def _load_docling() -> None:
    """Import Docling once, leaving the names ``None`` if it is not installed."""
    global _loaded, DocumentStream, InputFormat, DocumentConverter
    global DocItemLabel, create_pipeline_options_hash
    global PdfFormatOption, PdfPipelineOptions
    if _loaded:
        return
    _loaded = True
    try:
        from docling.datamodel.base_models import DocumentStream as stream
        from docling.datamodel.base_models import InputFormat as input_format
        from docling.datamodel.pipeline_options import (
            PdfPipelineOptions as pdf_options,
        )
        from docling.document_converter import DocumentConverter as converter
        from docling.document_converter import PdfFormatOption as pdf_format_option
        from docling.utils.pipeline_cache import (
            create_pipeline_options_hash as options_hash,
        )
        from docling_core.types.doc.labels import DocItemLabel as label
    except ImportError:  # pragma: no cover - installs without the extra
        return
    DocumentStream, InputFormat, DocumentConverter = stream, input_format, converter
    DocItemLabel, create_pipeline_options_hash = label, options_hash
    PdfFormatOption, PdfPipelineOptions = pdf_format_option, pdf_options


MISSING_EXTRA = (
    'DoclingParser needs the "docling" extra: pip install "fastrag-ai[docling]"'
)

#: What :class:`DoclingParser` can be asked to emit — see the module
#: docstring for what each yields.
OutputFormat = Literal["markdown", "docling"]

#: The ``output_format`` values understood by the constructor, and the
#: :class:`~fastrag.models.ContentFormat` each stamps.
_CONTENT_FORMATS: dict[OutputFormat, ContentFormat] = {
    "markdown": ContentFormat.MARKDOWN,
    "docling": ContentFormat.DOCLING,
}

#: The "office" media types :class:`DoclingParser` is meant to be routed to —
#: spreadsheets, word-processor documents, and presentations, both the
#: modern OOXML extensions and their legacy predecessors. Deliberately
#: narrower than every format Docling itself can convert (PDF and raw images
#: among them): those have their own, better-fitting existing routes (or
#: none yet), and "office" names what this filter is *for*, not an exhaustive
#: list of ``DocumentConverter``'s support matrix.
OFFICE_MEDIA_TYPES: frozenset[str] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
        "application/vnd.ms-excel",  # .xls
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",  # .docx
        "application/msword",  # .doc
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",  # .pptx
        "application/vnd.ms-powerpoint",  # .ppt
    }
)

#: The filter a :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter`
#: route to :class:`DoclingParser` matches against — the counterpart of
#: :data:`~fastrag.ingestion.steps.parsing.html.html_filter` for office
#: documents.
#:
#: .. code-block:: python
#:
#:     parser = ParserRouter(cache=cache).route(office_filter, DoclingParser())
office_filter = MediaTypeFilter(include_types=list(OFFICE_MEDIA_TYPES))


class DoclingParser(Parser):
    """Convert a binary document to Markdown or a serialized ``DoclingDocument``.

    Hands a :class:`~fastrag.ingestion.models.Source`'s bytes to Docling's
    ``DocumentConverter``, entirely from memory (never a temporary file):
    Docling infers the source format from a filename derived from the
    source's URI, not from anything written to disk.

    .. code-block:: python

        from fastrag.ingestion.steps.parsing import DoclingParser

        # Markdown, chunked by whatever chunker the caller already uses.
        parser = DoclingParser()

        # The structured document itself, for a Docling-native chunker.
        parser = DoclingParser(output_format="docling")

        # Describe the pictures found in PDFs (runs a vision model).
        parser = DoclingParser(describe_images=True)

    In ``"markdown"`` mode, a paginated document (PDF, slides) is organized
    **by page**: one :class:`~fastrag.models.Document` per non-empty page,
    each carrying ``metadata["page_no"]`` (1-based), ``metadata["page_count"]``
    and ``metadata["source_url"]``. For PDFs ``source_url`` is the source URI
    with a ``#page=N`` fragment, so every chunk inherits a link to the exact
    page it came from. Documents without pages (``.docx``, HTML, ...) are
    yielded whole, without ``page_no``.

    Conversion is synchronous and CPU-bound, so it always runs through
    ``asyncio.to_thread`` rather than blocking the event loop it is awaited
    from.
    """

    accepted_media_types = frozenset(
        {
            "application/pdf",
            "image/*",
            "text/*",
            "application/xhtml+xml",
            *OFFICE_MEDIA_TYPES,
        }
    )

    def __init__(
        self,
        *,
        output_format: OutputFormat = "markdown",
        describe_images: bool = False,
        picture_description_options: Any | None = None,
        allowed_formats: Sequence[InputFormat] | None = None,
        format_options: Mapping[InputFormat, Any] | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``DoclingParser``.

        Args:
            output_format: ``"markdown"`` (the default) yields
                ``DoclingDocument.export_to_markdown()``; ``"docling"``
                yields the document itself, serialized as JSON, for a
                Docling-native chunker to deserialize.
            describe_images: Enable image recognition for PDFs: Docling
                describes every picture with a vision model and the
                description is written into the output. Off by default as it
                is slow. Ignored for PDFs that ``format_options`` already
                configures explicitly.
            picture_description_options: Docling picture-description options
                (e.g. ``PictureDescriptionApiOptions`` to use a remote
                OpenAI-compatible model, or ``PictureDescriptionVlmEngineOptions``).
                ``None`` keeps Docling's default model. Only used when
                ``describe_images`` is true.
            allowed_formats: Restrict Docling to these input formats.
                ``None`` (the default) lets Docling accept anything it knows
                how to parse.
            format_options: Per-input-format Docling ``FormatOption``
                objects, handed straight to ``DocumentConverter`` — the way to reach
                pipeline options this parser does not surface itself, such as
                enabling picture description for spreadsheets. ``None`` (the
                default) leaves Docling's own defaults in place.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.

        Raises:
            fastrag.exceptions.ConfigurationError: The optional "docling"
                extra is not installed, or ``output_format`` is neither
                ``"markdown"`` nor ``"docling"``.
        """
        _load_docling()
        if DocumentStream is None or DocumentConverter is None or DocItemLabel is None:
            raise ConfigurationError(MISSING_EXTRA)
        if PdfFormatOption is None or PdfPipelineOptions is None:
            raise ConfigurationError(MISSING_EXTRA)
        if output_format not in _CONTENT_FORMATS:
            raise ConfigurationError(
                f"DoclingParser.output_format must be 'markdown' or 'docling', "
                f"got {output_format!r}"
            )
        super().__init__(cache=cache, policy=policy)
        self.output_format = output_format
        self.describe_images = describe_images
        self.allowed_formats = (
            tuple(allowed_formats) if allowed_formats is not None else None
        )
        options = dict(format_options) if format_options is not None else {}
        if describe_images and InputFormat.PDF not in options:
            pipeline_options = PdfPipelineOptions(
                do_picture_description=True,
                generate_picture_images=True,
                images_scale=2.0,
            )
            if picture_description_options is not None:
                pipeline_options.picture_description_options = (
                    picture_description_options
                )
            options[InputFormat.PDF] = PdfFormatOption(
                pipeline_options=pipeline_options
            )
        self.format_options = options or None
        # Built lazily on first use (see `_converter`) — construction itself
        # is cheap, but there is no reason to pay even that before this
        # parser has anything to convert.
        self._converter: Any = None

    def config(self) -> Mapping[str, Any]:
        """This parser's cache-identity configuration.

        ``output_format`` changes the *shape* of what a given file parses
        into — Markdown text versus a serialized ``DoclingDocument`` — so it
        is part of the fingerprint just as much as ``allowed_formats``
        (which formats Docling will even attempt) is: switching either
        invalidates the cache instead of serving a caller who asked for one
        shape the other shape cached earlier. ``format_options`` belongs for
        the same reason: it changes what Docling extracts from a given file
        (picture descriptions, say), so it must not serve a caller the
        pre-enrichment text cached earlier.
        """
        return {
            "output_format": self.output_format,
            "paged_documents": 1,
            "allowed_formats": (
                sorted(format_.value for format_ in self.allowed_formats)
                if self.allowed_formats is not None
                else None
            ),
            "format_options": (
                {
                    format_.value: _format_option_fingerprint(option)
                    for format_, option in self.format_options.items()
                }
                if self.format_options is not None
                else None
            ),
        }

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one document per non-empty page, or nothing if there is no text."""
        self.validate_input(item)
        if not item.content:
            return
        for document in await asyncio.to_thread(self._convert, item):
            yield document

    def _converter_instance(self) -> Any:
        """The lazily built, instance-reused ``DocumentConverter``."""
        if self._converter is None:
            self._converter = DocumentConverter(
                allowed_formats=(
                    list(self.allowed_formats)
                    if self.allowed_formats is not None
                    else None
                ),
                format_options=self.format_options,
            )
        return self._converter

    def _convert(self, source: Source) -> list[Document]:
        """Run one Docling conversion. Synchronous — always called via a thread.

        Raises:
            fastrag.exceptions.ParseError: Docling refused or failed on
                ``source``.
        """
        stream = DocumentStream(
            name=_filename(source.uri), stream=io.BytesIO(source.content)
        )
        try:
            result = self._converter_instance().convert(stream)
        except Exception as error:  # Docling raises its own ConversionError
            raise ParseError(
                f"{source.uri} could not be parsed by Docling: {error}"
            ) from error

        document: DoclingDocument = result.document
        page_numbers = sorted(document.pages)

        metadata: dict[str, Any] = {
            "docling_format": result.input.format,
            "page_count": len(page_numbers),
        }
        title = _title(document)
        if title:
            metadata["title"] = title
        content_format = _CONTENT_FORMATS[self.output_format]

        if self.output_format == "docling" or not page_numbers:
            markdown = document.export_to_markdown().strip()
            if not markdown:
                return []
            content = (
                markdown
                if self.output_format == "markdown"
                else document.model_dump_json()
            )
            return [
                self.build_document(
                    source, content, metadata, content_format=content_format
                )
            ]

        is_pdf = result.input.format == InputFormat.PDF
        pages: list[Document] = []
        for page_no in page_numbers:
            markdown = document.export_to_markdown(page_no=page_no).strip()
            if not markdown:
                continue
            page_metadata = {
                **metadata,
                "page_no": page_no,
                "source_url": (
                    f"{source.uri.split('#', 1)[0]}#page={page_no}"
                    if is_pdf
                    else source.uri
                ),
            }
            built = self.build_document(
                source, markdown, page_metadata, content_format=content_format
            )
            # Identical pages of one source must not collapse into one id.
            pages.append(
                dataclasses.replace(
                    built,
                    id=content_hash(f"{source.id}#page={page_no}:{markdown}"),
                )
            )
        return pages


def _format_option_fingerprint(option: Any) -> str:
    """A stable digest of one Docling ``FormatOption``'s effect on conversion.

    Docling's own ``create_pipeline_options_hash`` covers the pipeline
    options' values; the pipeline and backend classes around them are named
    separately, since two identical option sets still convert differently
    when fed to different pipelines.
    """
    parts = [
        type(option).__qualname__,
        getattr(option.pipeline_cls, "__qualname__", repr(option.pipeline_cls)),
        getattr(option.backend, "__qualname__", repr(option.backend)),
    ]
    if option.pipeline_options is not None:
        parts.append(create_pipeline_options_hash(option.pipeline_options))
    return "|".join(parts)


def _filename(uri: str) -> str:
    """Best-effort filename Docling can infer an input format from.

    Derived from ``uri``'s path component rather than anything on disk —
    conversion runs entirely from memory (see the class docstring).
    """
    name = PurePosixPath(urlparse(uri).path).name
    return name or "source"


def _title(document: DoclingDocument) -> str | None:
    """The document's declared title, or ``None`` if it has none.

    Docling labels at most one text item ``DocItemLabel.TITLE`` per
    document; most binary formats (a bare spreadsheet, for instance) declare
    none at all.
    """
    for text in document.texts:
        if text.label == DocItemLabel.TITLE and text.text.strip():
            return text.text.strip()
    return None
