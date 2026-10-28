"""The ``Embedder`` family base.

``docs/specification/subclasses.md``: ``Embedder`` is the ``DocumentStep``
every concrete embedder (``Model2VecEmbedder``, ``OpenAIEmbedder``) extends.
Its one job beyond ``DocumentStep`` itself is attaching a dense vector to each
incoming document while coalescing the underlying model calls into batches of
up to ``policy.batch_size`` texts, rather than issuing one call per document.

``DocumentStep`` (task 003) drives caching, retries and per-item concurrency
around a single-item ``transform`` hook — there is no batch-shaped hook on the
base to plug into, and adding one would mean changing a foundation three
already-finished tasks (parsing, chunking) build on. So batching is a detail
private to this family: :meth:`Embedder.transform` still satisfies the
single-item hook (each cache decision, retry and ``ItemFailed`` stays exactly
what ``DocumentStep`` already gives every other step), but internally it hands
its text to a small per-instance batcher that coalesces concurrently pending
calls into one :meth:`Embedder.embed` call once ``policy.batch_size`` texts
have queued, or after a short window lets slower siblings catch up. Nothing
here re-implements caching or retries — a batch that fails fails every text
in it, and each of those documents' own ``DocumentStep``-level retry (if any)
resubmits into a fresh batch.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Literal

from fastrag.exceptions import EmbeddingError
from fastrag.ingestion._hashing import config_hash
from fastrag.ingestion.cache import CacheStore

from fastrag.ingestion.events import DocumentEmbedded
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import DocumentStep, ExecutionContext
from fastrag.models import ContentFormat, Document

#: How long a batch waits for concurrently-arriving siblings before it is
#: flushed with fewer than ``policy.batch_size`` texts. Short relative to any
#: real model call (network round trip or an encode pass), long enough for
#: the handful of upstream items ``policy.concurrency`` admits at once to all
#: reach the batcher.
_BATCH_WINDOW_SECONDS = 0.05

_PendingEmbed = tuple[str, "asyncio.Future[list[float]]"]


class Embedder(DocumentStep, ABC):
    r"""Attaches a dense vector to each incoming document.

    Binding an upstream (``embedder(chunked_documents)``, via the inherited
    ``__call__``) and
    evaluating it (``async for document in embedder`` /
    :meth:`~fastrag.ingestion.steps.base.PipelineStep.run`) never embeds
    eagerly — only consuming the resulting stream does.

    Caching (per document id and this embedder's own configuration) and
    retries are entirely :class:`~fastrag.ingestion.steps.base.DocumentStep`'s
    job, applied around :meth:`transform`; nothing here re-implements either.
    Only the underlying model call is batched, up to ``policy.batch_size``
    texts per call — see the module docstring.

    Writing one is a class plus three members::

        from collections.abc import Sequence

        from fastrag.ingestion.steps.embedding import Embedder


        class EchoEmbedder(Embedder):
            def __init__(self, *, width: int = 8, cache=False, policy=None):
                super().__init__(cache=cache, policy=policy)
                self.width = width

            def config(self):
                return {"width": self.width}

            @property
            def model_id(self) -> str:
                return f"echo@{self.width}"

            @property
            def dimension(self) -> int:
                return self.width

            async def embed(self, texts: Sequence[str]) -> list[list[float]]:
                return [[float(len(text))] * self.width for text in texts]

    Every event an embedder emits is labelled ``"Embedding"``, inherited by
    every concrete embedder (``Model2VecEmbedder``, ``OpenAIEmbedder``, ...)
    unless it declares an ``event_name`` of its own.
    """

    event_name = "Embedding"

    def __init__(
        self,
        *,
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
    ) -> None:
        """Initialize an ``Embedder``.

        Args:
            cache: Cache backend, or ``False``/``None`` — see
                :class:`~fastrag.ingestion.steps.base.PipelineStep`.
            policy: Execution policy overriding the pipeline default.
        """
        super().__init__(cache=cache, policy=policy)
        self._pending: list[_PendingEmbed] = []
        self._batch_lock = asyncio.Lock()

    @property
    def config_hash(self) -> str:
        """Invalidate embeddings when the shared input serialization changes."""
        from fastrag.ingestion.steps.chunking.structure import EMBEDDING_TEXT_VERSION

        return config_hash(
            type(self).__name__,
            {
                "provider": self.config(),
                "embedding_text_version": EMBEDDING_TEXT_VERSION,
            },
        )

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier of the model configuration.

        Changing it — a different model, or a pinned dimension — must change
        a subclass's :meth:`~fastrag.ingestion.steps.base.PipelineStep.config`
        (and therefore its ``config_hash``), so vectors of different models or
        widths never share a document-cache entry.
        """

    @property
    @abstractmethod
    def dimension(self) -> int:
        """The width of every returned vector, known at construction.

        Resolved eagerly — never behind a network call — so a caller can name
        a vector-store collection from it before anything is embedded.
        """

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed a batch of texts, in order.

        Called with at most ``policy.batch_size`` texts at a time — the
        batcher in this base class is what enforces that ceiling, not the
        caller. Must return exactly one vector per text, in the same order.
        """

    async def transform(
        self, item: Document, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Embed the versioned input, preserving document display content.

        The actual model call this document's text takes part in may be
        shared with other documents pending at the same time — see the
        module docstring — but from the caller's perspective this awaits one
        vector for one document, exactly like any other
        :class:`~fastrag.ingestion.steps.base.DocumentStep` transform.

        This method only runs on a cache miss for *this* embedder — a cached
        document is replayed by ``DocumentStep`` without calling it at all —
        so the yielded document's own ``cached`` is always reset to
        ``False`` here, regardless of whatever the upstream item (e.g. a
        chunk already served from the chunker's own cache) carried in.
        Emitting :class:`~fastrag.ingestion.events.DocumentEmbedded` is
        ``DocumentStep._on_produced``'s job,
        not this method's, precisely so the event fires the same way for a
        cache hit as for a fresh embed.

        Raises:
            fastrag.exceptions.EmbeddingError: ``item.metadata["content_format"]``
                is :attr:`~fastrag.models.ContentFormat.DOCLING` — a
                serialized ``DoclingDocument``, not text. A document with no
                recorded ``content_format`` is treated as
                :attr:`~fastrag.models.ContentFormat.TEXT` (the default for
                every document that predates ``content_format`` — see
                :class:`~fastrag.ingestion.filters.ContentFormatFilter`), so
                nothing that embedded before this check existed starts
                failing now.
        """
        from fastrag.ingestion.steps.chunking.structure import build_embedding_text

        content_format = item.metadata.get("content_format", ContentFormat.TEXT)
        if content_format == ContentFormat.DOCLING:
            raise EmbeddingError(
                f"{item.id} carries content_format=DOCLING — a serialized "
                "DoclingDocument, not embeddable text. Chunk it first with "
                "fastrag.ingestion.steps.chunking.DoclingChunker, which emits "
                "plain-text chunks an embedder can accept."
            )
        effective_policy = self.effective_policy(context)
        vector = await self._embed_one(
            build_embedding_text(item.content, item.metadata),
            effective_policy.batch_size,
        )
        yield replace(item, embedding=vector, cached=False)

    async def _on_produced(
        self, item: Document, produced: list[Document], context: ExecutionContext
    ) -> None:
        """Emit one :class:`~fastrag.ingestion.events.DocumentEmbedded` per embedded document.

        Called once per upstream document regardless of whether ``produced``
        came from the cache or a fresh :meth:`transform` call, so
        ``document.cached`` is trustworthy either way — see
        ``DocumentStep._on_produced``.
        """
        for document in produced:
            await context.emit(
                DocumentEmbedded(
                    document=document,
                    step=self.name,
                    event_name=self.event_name,
                )
            )

    async def _embed_one(self, text: str, batch_size: int) -> list[float]:
        """Queue ``text`` for embedding, flushing the batch it lands in.

        A call joins whatever batch is currently accumulating. It flushes
        immediately once ``batch_size`` texts have queued; otherwise it is
        the first arrival responsible for scheduling :func:`_delayed_flush`,
        which flushes whatever is still pending after
        :data:`_BATCH_WINDOW_SECONDS` so a batch smaller than ``batch_size``
        is never left waiting forever for siblings that will never arrive.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[list[float]] = loop.create_future()
        to_flush: list[_PendingEmbed] | None = None
        async with self._batch_lock:
            self._pending.append((text, future))
            if len(self._pending) >= batch_size:
                to_flush, self._pending = self._pending, []
            elif len(self._pending) == 1:
                asyncio.create_task(self._delayed_flush(batch_size))
        if to_flush is not None:
            await self._flush(to_flush)
        return await future

    async def _delayed_flush(self, batch_size: int) -> None:
        """Flush whatever is still pending after the batch window elapses."""
        await asyncio.sleep(_BATCH_WINDOW_SECONDS)
        to_flush: list[_PendingEmbed] | None = None
        async with self._batch_lock:
            if self._pending:
                to_flush, self._pending = self._pending, []
        if to_flush is not None:
            await self._flush(to_flush)
        # `batch_size` is unused once the window fires — flushing at this
        # point always takes whatever is pending, full or not — but is kept
        # as a parameter so a future revision can re-check the ceiling
        # without changing the call site.
        del batch_size

    async def _flush(self, batch: list[_PendingEmbed]) -> None:
        """Call :meth:`embed` once for ``batch`` and resolve each future.

        Runs with the batch lock already released, so other calls keep
        accumulating the next batch while this one's model call is in
        flight — batches overlap rather than serializing behind one another.
        A failure fails every text in this batch; each document's own
        ``DocumentStep``-level retry (if any) calls :meth:`transform` again,
        which resubmits into a fresh batch.
        """
        texts = [text for text, _ in batch]
        try:
            vectors = await self.embed(texts)
        except Exception as error:  # propagated to every waiter, not swallowed
            for _, future in batch:
                if not future.done():
                    future.set_exception(error)
            return
        if len(vectors) != len(batch):
            mismatch = EmbeddingError(
                f"{type(self).__name__}.embed returned {len(vectors)} vectors "
                f"for {len(batch)} texts"
            )
            for _, future in batch:
                if not future.done():
                    future.set_exception(mismatch)
            return
        for (_, future), vector in zip(batch, vectors, strict=True):
            if not future.done():
                future.set_result(vector)
