"""``CharacterTextSplitter``: split on one separator only, no fallback hierarchy."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from typing import Any, Literal

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.cache import CacheStore
from fastrag.models import Document
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.chunking.base import Chunker

try:
    from langchain_text_splitters import (
        CharacterTextSplitter as _CharacterTextSplitter,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _LANGCHAIN_AVAILABLE = False

#: The splitter class, or ``None`` when the optional "chunkers" extra is
#: missing; :meth:`CharacterTextSplitter.__init__` refuses to run without it.
_LangchainSplitter: Any = _CharacterTextSplitter if _LANGCHAIN_AVAILABLE else None

MISSING_EXTRA = 'CharacterTextSplitter needs the "chunkers" extra: pip install "fastrag-ai[chunkers]"'


class CharacterTextSplitter(Chunker):
    """Split on a single literal separator, with no separator hierarchy.

    Unlike :class:`~fastrag.ingestion.steps.chunking.recursive.RecursiveCharacterTextSplitter`,
    this never descends to a smaller separator when a piece is still too
    large after splitting on ``separator`` — the right choice when a source's
    structure already guarantees short-enough pieces at one boundary (e.g. a
    page's own section breaks), and the wrong one for prose of unpredictable
    length.

    .. code-block:: python

        from fastrag.ingestion.steps.chunking import CharacterTextSplitter

        chunker = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, cache=cache)
    """

    accepted_content_formats = frozenset(["text", "markdown", "html", "rst"])

    def __init__(
        self,
        *,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        separator: str = "\n\n",
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure the splitter, validating its bounds immediately.

        Args:
            chunk_size: Target chunk length in characters. Must be at least 1.
            chunk_overlap: Characters repeated between neighbours, so a
                sentence split across a boundary is still whole in one of
                them. Must be non-negative and smaller than ``chunk_size``.
            separator: The single literal string to split on.
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
                "CharacterTextSplitter.chunk_size must be at least 1"
            )
        if not 0 <= chunk_overlap < chunk_size:
            raise ConfigurationError(
                "CharacterTextSplitter.chunk_overlap must be "
                f"non-negative and smaller than chunk_size ({chunk_size})"
            )

        super().__init__(cache=cache, policy=policy)
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator
        self._splitter = _LangchainSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separator=separator,
            keep_separator=True,
        )

    def config(self) -> Mapping[str, Any]:
        """This splitter's configuration, for cache-identity hashing."""
        return {
            "chunk_size": self.chunk_size,
            "chunk_overlap": self.chunk_overlap,
            "separator": self.separator,
        }

    def split(self, document: Document) -> Iterator[str]:
        """Split ``document.content`` on ``self.separator`` only."""
        self.validate_input(document)
        yield from self._splitter.split_text(document.content)
