"""``RecursiveCharacterTextSplitter``: split on a separator hierarchy, largest first."""

from __future__ import annotations

from collections.abc import Iterator, Mapping, Sequence
from typing import Any, Literal

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.cache import CacheStore
from fastrag.models import Document
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.chunking.base import Chunker

try:
    from langchain_text_splitters import (
        RecursiveCharacterTextSplitter as _RecursiveCharacterTextSplitter,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _LANGCHAIN_AVAILABLE = False

#: The splitter class, or ``None`` when the optional "chunkers" extra is
#: missing; :meth:`RecursiveCharacterTextSplitter.__init__` refuses to run
#: without it.
_LangchainSplitter: Any = (
    _RecursiveCharacterTextSplitter if _LANGCHAIN_AVAILABLE else None
)

MISSING_EXTRA = (
    'RecursiveCharacterTextSplitter needs the "chunkers" extra: '
    'pip install "fastrag-ai[chunkers]"'
)


class RecursiveCharacterTextSplitter(Chunker):
    """Split on paragraphs, then lines, then words, then characters.

    The default chunker: it descends the separator hierarchy only when a
    piece is still too large, so a chunk breaks at a blank line whenever it
    can and mid-word only when it must. That makes it the right choice for
    prose regardless of source format, including the Markdown
    :class:`~fastrag.ingestion.steps.parsing.HtmlParser` produces.

    .. code-block:: python

        from fastrag.ingestion.steps.chunking import RecursiveCharacterTextSplitter

        chunker = RecursiveCharacterTextSplitter(
            chunk_size=500, chunk_overlap=50, cache=cache
        )
    """

    accepted_content_formats = frozenset(["text", "markdown", "html", "rst"])

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separators: Sequence[str] | None = None,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure the splitter, validating its bounds immediately.

        Args:
            chunk_size: Target chunk length in characters. Must be at least 1.
            chunk_overlap: Characters repeated between neighbours, so a
                sentence split across a boundary is still whole in one of
                them. Must be non-negative and smaller than ``chunk_size``.
            separators: The hierarchy, largest unit first. ``None`` uses
                paragraph, line, word, character — which suits both prose and
                the Markdown :class:`~fastrag.ingestion.steps.parsing.HtmlParser`
                produces.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.

        Raises:
            fastrag.exceptions.ConfigurationError: The "chunkers" extra is not
                installed, ``chunk_size < 1``, or ``chunk_overlap`` is not in
                ``[0, chunk_size)``.
        """
        if _LangchainSplitter is None:
            raise ConfigurationError(MISSING_EXTRA)
        if chunk_size < 1:
            raise ConfigurationError(
                "RecursiveCharacterTextSplitter.chunk_size must be at least 1"
            )
        if not 0 <= chunk_overlap < chunk_size:
            raise ConfigurationError(
                "RecursiveCharacterTextSplitter.chunk_overlap must be "
                f"non-negative and smaller than chunk_size ({chunk_size})"
            )

        super().__init__(cache=cache, policy=policy)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators
        self._splitter = _LangchainSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=list(separators) if separators is not None else None,
            keep_separator=True,
        )

    def config(self) -> Mapping[str, Any]:
        """This splitter's configuration, for cache-identity hashing."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separators": (
                list(self.separators) if self.separators is not None else None
            ),
        }

    def split(self, document: Document) -> Iterator[str]:
        """Split ``document.content`` on the paragraph/line/word/character hierarchy."""
        self.validate_input(document)
        yield from self._splitter.split_text(document.content)
