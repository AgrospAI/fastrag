"""``LangchainSemanticChunker``: LangChain Experimental's ``SemanticChunker`` over an SDK Embedder."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from typing import Any, Literal

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.cache import CacheStore
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.chunking.base import Chunker, SplitPiece
from fastrag.ingestion.steps.embedding.base import Embedder
from fastrag.models import Document

try:
    from langchain_core.embeddings import Embeddings as _Embeddings
    from langchain_experimental.text_splitter import (
        SemanticChunker as _SemanticChunkerImpl,
    )

    _LANGCHAIN_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _LANGCHAIN_AVAILABLE = False

#: The splitter class, or ``None`` when the optional "chunkers" extra is
#: missing; :meth:`LangchainSemanticChunker.__init__` refuses to run without it.
_LangchainSemanticChunker: Any = _SemanticChunkerImpl if _LANGCHAIN_AVAILABLE else None

MISSING_EXTRA = (
    'LangchainSemanticChunker needs the "chunkers" extra: '
    'pip install "fastrag-ai[chunkers]"'
)


if _LANGCHAIN_AVAILABLE:

    class _EmbedderAdapter(_Embeddings):
        """Expose an asynchronous SDK Embedder as synchronous LangChain Embeddings.

        LangChain's ``SemanticChunker.split_text`` is synchronous, so it runs in
        a worker thread; each ``embed_documents`` call it makes is scheduled back
        onto the event loop that owns ``embedder`` and awaited from that thread.
        """

        def __init__(
            self,
            embedder: Embedder,
            loop: asyncio.AbstractEventLoop,
            batch_size: int,
        ) -> None:
            self.embedder = embedder
            self.loop = loop
            self.batch_size = batch_size

        async def _embed(self, texts: Sequence[str]) -> list[list[float]]:
            vectors: list[list[float]] = []
            for start in range(0, len(texts), self.batch_size):
                vectors.extend(
                    await self.embedder.embed(texts[start : start + self.batch_size])
                )
            return vectors

        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            return asyncio.run_coroutine_threadsafe(
                self._embed(texts), self.loop
            ).result()

        def embed_query(self, text: str) -> list[float]:
            return self.embed_documents([text])[0]


class LangchainSemanticChunker(Chunker):
    """Split with LangChain Experimental's ``SemanticChunker``, embedding with an SDK Embedder.

    A thin wrapper: sentence grouping, distance thresholds and chunk assembly
    are all LangChain's, so output matches ``SemanticChunker`` exactly (it
    trims and rejoins sentences rather than preserving source whitespace —
    use :class:`~fastrag.ingestion.steps.chunking.EmbeddingSemanticChunker`
    when exact source slices matter). Only the embedding model is ours.

    .. code-block:: python

        from fastrag.ingestion.steps.chunking import LangchainSemanticChunker
        from fastrag.ingestion.steps.embedding import OpenAIEmbedder

        chunker = LangchainSemanticChunker(
            embeddings=OpenAIEmbedder(model="text-embedding-3-small", api_key=key)
        )
    """

    accepted_content_formats = frozenset(["text", "markdown"])

    def __init__(
        self,
        *,
        embeddings: Embedder,
        buffer_size: int = 1,
        breakpoint_threshold_type: Literal[
            "percentile", "standard_deviation", "interquartile", "gradient"
        ] = "percentile",
        breakpoint_threshold_amount: float | None = None,
        number_of_chunks: int | None = None,
        sentence_split_regex: str = r"(?<=[.?!])\s+",
        min_chunk_size: int | None = None,
        add_start_index: bool = False,
        batch_size: int = 64,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Configure the LangChain splitter; arguments mirror ``SemanticChunker``.

        Args:
            embeddings: The SDK embedder used for sentence vectors.
            buffer_size: Neighbouring sentences combined into each embedded window.
            breakpoint_threshold_type: How a distance becomes a cut.
            breakpoint_threshold_amount: Threshold parameter; ``None`` uses
                LangChain's default for ``breakpoint_threshold_type``.
            number_of_chunks: Target chunk count, overriding the threshold.
            sentence_split_regex: Regex sentences are split on.
            min_chunk_size: Minimum chunk length in characters.
            add_start_index: Record each chunk's source offset as
                ``metadata["start_index"]`` (``None`` when LangChain rejoined
                its sentences with whitespace absent from the source).
            batch_size: Maximum texts per :meth:`Embedder.embed` call.
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.

        Raises:
            fastrag.exceptions.ConfigurationError: The "chunkers" extra is not
                installed, or ``batch_size < 1``.
        """
        if _LangchainSemanticChunker is None:
            raise ConfigurationError(MISSING_EXTRA)
        if batch_size < 1:
            raise ConfigurationError(
                "LangchainSemanticChunker.batch_size must be at least 1"
            )
        super().__init__(cache=cache, policy=policy)
        self.embeddings = embeddings
        self.buffer_size = buffer_size
        self.breakpoint_threshold_type = breakpoint_threshold_type
        self.breakpoint_threshold_amount = breakpoint_threshold_amount
        self.number_of_chunks = number_of_chunks
        self.sentence_split_regex = sentence_split_regex
        self.min_chunk_size = min_chunk_size
        self.add_start_index = add_start_index
        self.batch_size = batch_size

    def config(self) -> Mapping[str, Any]:
        """Return splitter settings and the embedder's nonsecret identity."""
        return {
            "model": self.embeddings.model_id,
            "provider_config_hash": self.embeddings.config_hash,
            "buffer_size": self.buffer_size,
            "breakpoint_threshold_type": self.breakpoint_threshold_type,
            "breakpoint_threshold_amount": self.breakpoint_threshold_amount,
            "number_of_chunks": self.number_of_chunks,
            "sentence_split_regex": self.sentence_split_regex,
            "min_chunk_size": self.min_chunk_size,
            "add_start_index": self.add_start_index,
        }

    def split(self, document: Document) -> Iterator[SplitPiece]:
        """Require asynchronous model evaluation before yielding any pieces."""
        self.validate_input(document)
        raise NotImplementedError(
            "LangchainSemanticChunker uses asplit(); consume it async"
        )

    def _split_sync(self, text: str, adapter: Any) -> list[SplitPiece]:
        splitter = _LangchainSemanticChunker(
            embeddings=adapter,
            buffer_size=self.buffer_size,
            breakpoint_threshold_type=self.breakpoint_threshold_type,
            breakpoint_threshold_amount=self.breakpoint_threshold_amount,
            number_of_chunks=self.number_of_chunks,
            sentence_split_regex=self.sentence_split_regex,
            min_chunk_size=self.min_chunk_size,
        )
        chunks = splitter.split_text(text)
        if not self.add_start_index:
            return list(chunks)
        # LangChain's own start_index is a running sum of chunk lengths, which
        # drifts once inter-sentence whitespace is dropped; locate each chunk
        # in the source instead. A chunk rejoined with different whitespace
        # than its source has no exact offset, recorded as None.
        pieces: list[SplitPiece] = []
        cursor = 0
        for chunk in chunks:
            start = text.find(chunk, cursor)
            if start >= 0:
                cursor = start + len(chunk)
            pieces.append((chunk, {"start_index": start if start >= 0 else None}))
        return pieces

    async def asplit(self, document: Document) -> AsyncIterator[SplitPiece]:
        """Run LangChain's splitter in a thread, embedding on this event loop."""
        self.validate_input(document)
        if not document.content.strip():
            return
        adapter = _EmbedderAdapter(
            self.embeddings, asyncio.get_running_loop(), self.batch_size
        )
        for piece in await asyncio.to_thread(
            self._split_sync, document.content, adapter
        ):
            yield piece
