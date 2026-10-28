"""``DefaultParser``: decode the bytes and keep them as they are.

``docs/specification/API.md``: "just decodes the content as text, without any
parsing." Its capabilities accept textual media; binary formats need a parser.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Mapping
from typing import Any, Literal

from fastrag.exceptions import ParseError
from fastrag.ingestion.steps.parsing.capabilities import (
    TEXT_MEDIA_TYPES,
    source_media_type,
    text_format,
)
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.models import Source
from fastrag.models import ContentFormat, Document
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.parsing.base import Parser


class DefaultParser(Parser):
    """Decode the bytes as text, without any parsing.

    The text-only fallback parser :class:`~fastrag.ingestion.steps.parsing.router.ParserRouter` falls
    back to for text that has no more specific route.

    .. code-block:: python

        from fastrag.ingestion.steps.parsing import DefaultParser

        parser = DefaultParser(encoding="latin-1")
    """

    content_format = ContentFormat.TEXT
    accepted_media_types = TEXT_MEDIA_TYPES

    def __init__(
        self,
        *,
        encoding: str = "utf-8",
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize ``DefaultParser``.

        Args:
            encoding: The codec the bytes are decoded with — replacing what
                it cannot represent rather than raising, so one mis-encoded
                character does not drop an otherwise-good page.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
        """
        super().__init__(cache=cache, policy=policy)
        self.encoding = encoding

    def config(self) -> Mapping[str, Any]:
        """This parser's cache-identity configuration.

        ``encoding`` is part of the fingerprint: decoding the same bytes
        differently is a different parse.
        """
        return {"encoding": self.encoding, "text_capabilities": "v1"}

    async def transform(
        self, item: Source, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Yield one document holding the decoded text, or nothing if blank."""
        self.validate_input(item)
        if source_media_type(item) == "application/octet-stream":
            try:
                item.content.decode(self.encoding, errors="strict")
            except UnicodeDecodeError as error:
                raise ParseError(
                    f"DefaultParser refused unknown binary content from {item.uri!r}; "
                    "register a ParserRouter.route/default with a compatible parser."
                ) from error
        content = item.content.decode(self.encoding, errors="replace").strip()
        if (
            item.content.startswith(
                (
                    b"%PDF-",
                    b"PK\x03\x04",
                    b"PK\x05\x06",
                    b"PK\x07\x08",
                    b"\x89PNG",
                    b"GIF8",
                    b"\xff\xd8\xff",
                    b"\x1f\x8b",
                )
            )
            or "\x00" in content
            or (
                content
                and sum(ord(c) < 32 and c not in "\n\r\t\f" for c in content)
                / len(content)
                > 0.05
            )
        ):
            raise ParseError(
                f"DefaultParser refused binary content from {item.uri!r}; "
                "register a ParserRouter.route/default with a compatible parser."
            )
        if content:
            yield self.build_document(
                item, content, content_format=ContentFormat(text_format(item))
            )
