"""The ``Parser`` family base.

``docs/specification/subclasses.md``: ``Parser`` is the ``DocumentStep`` every
concrete parser (``HtmlParser``, ``DefaultParser``, ``ParserRouter``) extends.
Parsers convert supplied content and may call content-selection or vision
models within the async ``transform`` contract. Fetchers own page downloads,
JavaScript rendering, and browser interaction.
"""

from __future__ import annotations

from abc import ABC
from collections.abc import AsyncIterator
from typing import Any

from fastrag.ingestion._hashing import config_hash, content_hash
from fastrag.ingestion._site import site_of
from fastrag.ingestion.events import DocumentParsed
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.base import DocumentStep, ExecutionContext
from fastrag.models import ContentFormat, Document


class Parser(DocumentStep, ABC):
    """Turns raw :class:`~fastrag.ingestion.models.Source` bytes into text.

    One source may yield several documents, which is why :meth:`transform`
    (inherited abstract from :class:`~fastrag.ingestion.steps.base.DocumentStep`)
    returns an async iterator rather than a single document. Binding an
    upstream (``parser(sources)``, via the inherited ``__call__``) and
    evaluating it (``async for document in parser`` /
    :meth:`~fastrag.ingestion.steps.base.PipelineStep.run`) never runs a
    parse eagerly — only consuming the resulting stream does.

    Every event a parser emits is labelled ``"Parsing"``, inherited by every
    concrete parser (``HtmlParser``, ``DocusaurusParser``, ``ParserRouter``,
    ...) unless it declares an ``event_name`` of its own.
    """

    event_name = "Parsing"

    #: The :class:`~fastrag.models.ContentFormat` this parser's documents
    #: carry by default, stamped by :meth:`build_document` into
    #: ``metadata["content_format"]``. Override on a subclass whose output is
    #: always one format (``HtmlParser``/``DocusaurusParser`` set
    #: ``MARKDOWN``); pass ``content_format=`` to a single
    #: :meth:`build_document` call instead for a parser whose output format
    #: depends on its own configuration.
    content_format: ContentFormat = ContentFormat.TEXT

    #: None retains legacy permissiveness; MIME patterns may use ``type/*``.
    accepted_media_types: frozenset[str] | None = None

    @property
    def config_hash(self) -> str:
        """Invalidate cached documents created before source-aware identities."""
        return config_hash(
            type(self).__name__,
            {**self.config(), "document_identity_version": 2},
        )

    def validate_input(self, item: Source) -> None:
        """Validate declared source capabilities before work and cache replay."""
        from fastrag.exceptions import ParseError
        from fastrag.ingestion.steps.parsing.capabilities import (
            accepts_media_type,
            source_media_type,
        )

        actual = source_media_type(item)
        accepted = self.accepted_media_types
        if accepted is not None and not accepts_media_type(actual, accepted):
            raise ParseError(
                f"{self.name} cannot parse source {item.uri!r} with media type {actual!r}; "
                f"accepts {sorted(accepted)!r}. Register a ParserRouter.route or default "
                "with a compatible parser (e.g. DoclingParser for PDF/office documents)."
            )

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Document]:
        """Emit :class:`~fastrag.ingestion.events.DocumentParsed` per document.

        Wraps :class:`~fastrag.ingestion.steps.base.DocumentStep`'s concrete
        ``evaluate`` (which already resolved, per item, whether the document
        came from cache or a fresh :meth:`transform` call — ``document.cached``
        is trustworthy either way) rather than emitting inside a concrete
        parser's own :meth:`transform`, so the event fires the same way for a
        cache hit as for a fresh parse, and exactly once regardless of how
        many layers of routing (e.g. :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter`) produced the
        document.
        """
        async for document in super().evaluate(context):
            await context.emit(
                DocumentParsed(
                    document=document,
                    step=self.name,
                    event_name=self.event_name,
                )
            )
            yield document

    def build_document(
        self,
        source: Source,
        content: str,
        metadata: dict[str, Any] | None = None,
        *,
        content_format: ContentFormat | None = None,
    ) -> Document:
        """Build one document from ``source``'s parsed ``content``.

        Sets ``id`` (a hash of the source identity and parsed content), ``source_id``,
        ``step_config_hash`` (this parser's own :attr:`config_hash`), and the
        reserved metadata keys every parser gets right without writing them
        itself: ``source_uri``, ``content_type`` (``source``'s own, as
        declared or guessed upstream), and ``content_format`` (what kind of
        string ``content`` is — :attr:`content_format` unless overridden for
        this one call). The ``site_name``/``favicon_url`` a fetcher found on
        ``source`` are carried over too, unless ``metadata`` sets its own.

        Args:
            source: The source ``content`` was parsed from.
            content: The parsed text.
            metadata: Extra metadata to carry, e.g. an HTML page's ``title``.
            content_format: Overrides :attr:`content_format` for this one
                document, for a parser whose output format depends on its
                own configuration rather than being fixed per class.
        """
        return Document(
            # Identical text at different URLs must retain independent lineage
            # in downstream chunk caches and vector-store records.
            id=content_hash(f"{source.id}:{content}"),
            source_id=source.id,
            content=content,
            step_config_hash=self.config_hash,
            metadata={
                **site_of(source.metadata),
                **(metadata or {}),
                "source_uri": source.uri,
                "content_type": source.content_type,
                "content_format": (
                    content_format
                    if content_format is not None
                    else self.content_format
                ),
            },
        )
