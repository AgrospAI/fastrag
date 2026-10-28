"""The ``VectorStore`` sink family base.

A ``VectorStore`` is the pipeline's storage backend, not a ``PipelineStep``
itself: a caller opens it once (``async with MilvusVectorStore(url=...) as
vector_store:``) and hands an upstream ``DocumentStep`` to
:meth:`~fastrag.vector_store.base.VectorStore.save`, which returns the
terminal ``PipelineStep[Document]`` that indexes when evaluated. Constructing
that step performs no I/O.

Batching mirrors ``fastrag.ingestion.steps.embedding.base.Embedder``: each
document is queued into a private micro-batcher that coalesces pending
documents into one existence check plus (at most) one
:meth:`~fastrag.vector_store.base.VectorStore.upsert` call once
``policy.batch_size`` documents have queued, or after a short window lets
slower siblings catch up. The store is the source of truth for "already
indexed": :meth:`~fastrag.vector_store.base.VectorStore.existing_ids` is
asked once per batch, present ids are skipped and reported ``cached=True``,
the rest are upserted and reported ``cached=False``. Under a policy with
``recreate_cache``, nothing counts as already indexed: every document is
upserted, overwriting what the store holds for its id, as a rebuild expects.

:meth:`~fastrag.vector_store.base.VectorStore.sync` wraps that same step and,
once the upstream is exhausted, deletes every stored document the run did not
yield, so the store ends up holding exactly the run's documents. It deletes
only after a clean run: a failed item's documents are missing from the run
without being gone from the source.

``VectorStore`` and its concrete stores live at the top level, in
:mod:`fastrag.vector_store`, rather than under :mod:`fastrag.ingestion`: a
vector store is queried at read time as well as written at ingest time, so it
is a storage backend the whole SDK depends on.
"""

from __future__ import annotations

import asyncio
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Mapping, Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import replace
from typing import Any, Self

from fastrag.exceptions import ConfigurationError, VectorStoreError
from fastrag.ingestion.cache.base import _public_item
from fastrag.ingestion.events import (
    DocumentEmbedded,
    DocumentSaved,
    DocumentsPruned,
    ItemFailed,
    PipelineEvent,
)
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import (
    DocumentStep,
    ExecutionContext,
    PipelineStep,
    RoutedStep,
    _with_retries,
    run_step,
    stream_batches,
)
from fastrag.models import Document

#: How long a batch waits for slower siblings before flushing short. Without
#: it, a run with fewer than ``batch_size`` documents would hang (default
#: ``concurrency=8 < batch_size=16``).
_BATCH_WINDOW_SECONDS = 0.05

#: A queued document, the future resolving whether the store already held
#: it, and whether to overwrite it if so.
_PendingSave = tuple[Document, "asyncio.Future[bool]", bool]

#: Ids per :meth:`VectorStore.delete` call when a sync prunes stale documents.
_DELETE_BATCH_SIZE = 1000


class VectorStore(AbstractAsyncContextManager["VectorStore"], ABC):
    """Sink backend for persisting processed and embedded documents.

    Opened once per run (``async with MilvusVectorStore(url=...) as
    vector_store:``) and handed the final upstream step via :meth:`save`,
    which returns a lazy terminal :class:`~fastrag.ingestion.steps.base.PipelineStep`
    — nothing is indexed until that step is evaluated.
    """

    def __init__(self, *, policy: Policy | None = None) -> None:
        """Store the default policy; no I/O happens until ``__aenter__``.

        Args:
            policy: Execution policy for the terminal step :meth:`save`
                returns — governs its concurrency, retries, and the size of
                each :meth:`upsert` batch.
        """
        self.policy = policy or Policy()

    async def __aenter__(self) -> Self:
        """Open backend resources. Overridden by concrete stores that need to."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Release backend resources. Overridden by concrete stores that need to."""
        return None

    @abstractmethod
    async def ensure_ready(self, dimension: int) -> None:
        """Create the backing collection for ``dimension``-wide vectors if absent.

        Idempotent and cheap to call again on a collection that already
        exists; called once, lazily, before the first upsert, with the vector
        width taken from the first document flowing through the terminal step.

        Args:
            dimension: Width of the vectors this run's documents carry.
        """
        ...

    @abstractmethod
    async def upsert(self, documents: Sequence[Document]) -> None:
        """Insert or replace ``documents`` by :attr:`~fastrag.models.Document.id`, in one call.

        Idempotent: rerunning with the same ids converges rather than
        duplicating rows. Called with at most ``self.policy.batch_size``
        documents at a time — the batcher in :meth:`save`'s returned step
        enforces that ceiling, not the caller — and possibly fewer, since
        documents already reported by :meth:`existing_ids` are excluded.

        Args:
            documents: The batch to upsert. Every document carries a
                non-``None`` :attr:`~fastrag.models.Document.embedding`
                — the terminal step never lets one through otherwise.
        """
        ...

    @abstractmethod
    async def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Return the subset of ``ids`` already present in the store.

        Called once per pending batch, never once per document, so the round
        trips it adds scale with the number of batches. An id returned here
        is treated as already indexed (skipped, reported ``cached=True``); an
        absent id is upserted and reported ``cached=False``. Skipping a
        present document is an optimisation, not a correctness requirement —
        a store unable to answer cheaply may upsert everything, but must then
        report which ids were genuinely present rather than guessing.

        Args:
            ids: Candidate document ids to check, at most
                ``self.policy.batch_size`` at a time.
        """
        ...

    @abstractmethod
    async def search(
        self, embedding: Sequence[float], *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Return at most ``k`` documents nearest ``embedding``, best first.

        The read-path counterpart to :meth:`upsert`, queried by
        :class:`~fastrag.retrieval.vector.VectorRetriever` and, through it,
        :class:`~fastrag.retrieval.reranking_retriever.RerankingRetriever`.
        Never opens the store — the caller (a retriever) holds an
        already-open one.

        Stamps two reserved metadata keys on every returned document,
        alongside its stored ``content``/``metadata`` (including
        ``source_uri``):

        - ``score``: the backend's own relevance score, on the backend's own
          scale — never comparable across backends or metrics.
        - ``rank``: 0-based position in the returned (best-first) list.

        The list is best-first regardless of the configured metric's raw sort
        direction, so position 0 is always the best match.

        Args:
            embedding: Query vector, the same width the store's collection
                was created with.
            k: Maximum number of documents to return. Fewer come back if the
                collection holds fewer than ``k`` matching documents — this
                is not an error.
            filter: Backend-native filter expression, applied server-side.
                ``None`` returns the ``k`` nearest documents unfiltered.
        """
        ...

    async def all_ids(self) -> set[str]:
        """Return the id of every document in the store.

        Needed only by :meth:`sync`, so a store that does not support it may
        leave this unimplemented. Returns an empty set when the backing
        collection does not exist yet.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sync()")

    async def delete(self, ids: Sequence[str]) -> None:
        """Remove the documents with ``ids``; ids the store does not hold are ignored.

        Needed only by :meth:`sync`, which calls it with at most 1000 ids at
        a time.
        """
        raise NotImplementedError(f"{type(self).__name__} does not support sync()")

    def sync(
        self, upstream: PipelineStep[Document], *, skip_existing: bool = False
    ) -> PipelineStep[Document]:
        """Like :meth:`save`, then delete every stored document ``upstream`` did not yield.

        After a clean run the store holds exactly the documents the run
        produced: documents whose source disappeared or changed (a changed
        chunk gets a new id) are removed. Deletion happens once the upstream
        is exhausted, so a run that raises or is stopped early deletes
        nothing. A run with any ``ItemFailed`` deletes nothing either: a
        failed item's documents are missing from the run, not from the
        source. Either way a
        :class:`~fastrag.ingestion.events.DocumentsPruned` event reports the
        outcome.

        The run must cover everything the store should keep — use one store
        (collection/index) per synced pipeline.

        Args:
            upstream: Final upstream step yielding embedded documents.
            skip_existing: Passed through to :meth:`save`.
        """
        return _SyncStep(self)(self.save(upstream, skip_existing=skip_existing))

    def save(
        self, upstream: PipelineStep[Document], *, skip_existing: bool = False
    ) -> PipelineStep[Document]:
        """Return a lazy terminal step that indexes ``upstream``'s documents.

        Constructing the returned step performs no I/O — evaluating it
        (``async for document in vector_store.save(upstream):`` or
        ``.run()``) is what upserts. Each document is queued into a batcher
        that checks :meth:`existing_ids` and flushes to :meth:`upsert` once
        ``self.policy.batch_size`` documents have queued, or after a short
        window; every document is yielded back once its batch is resolved,
        with :attr:`~fastrag.models.Document.cached` set to whether the
        store already held it, and
        :class:`~fastrag.ingestion.events.DocumentSaved` emitted for it.

        Args:
            upstream: Final upstream step yielding embedded documents.
            skip_existing: For an identity-preserving Embedder, check destination
                ids before embedding. Indexed items bypass vector retrieval and
                produce cached DocumentEmbedded and DocumentSaved events. Cached events
                and results may contain only identity/metadata with empty content
                and no vectors. Custom content filters disable header projection.
                Use a separate destination for each embedding configuration.

        Either way, ``recreate_cache`` (on the store's policy or the
        pipeline's) skips the existence check and upserts every document.
        """
        if skip_existing:
            from fastrag.ingestion.steps.embedding.base import Embedder

            if not isinstance(upstream, Embedder) or upstream._upstream is None:
                raise ConfigurationError("skip_existing requires a bound Embedder")
            return _IndexStep(self, upstream)(upstream)
        return _SaveStep(self)(upstream)


def _headers_safe(root: PipelineStep[Any]) -> bool:
    """Content-aware custom routing requires complete cached records."""
    pending = [root]
    seen: set[int] = set()
    while pending:
        step = pending.pop()
        if id(step) in seen:
            continue
        seen.add(id(step))
        if step.upstream is not None:
            pending.append(step.upstream)
        if isinstance(step, RoutedStep):
            if step.routing_requires_content:
                return False
            for predicate, child in step.routes:
                if not predicate.header_safe:
                    return False
                pending.append(child)
            if step._default_step is not None:
                pending.append(step._default_step)
    return True


class _EmbeddingWindow(DocumentStep):
    """Run a missing-document window through normal embedding lifecycle events."""

    def __init__(self, embedder: Any, items: list[Document]) -> None:
        super().__init__(cache=False, policy=embedder.policy)
        self.name = embedder.name
        self.event_name = embedder.event_name
        self._embedder = embedder
        self._items = items

    def config(self) -> Mapping[str, Any]:
        return {}

    async def transform(
        self, item: Any, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        raise AssertionError("embedding window evaluates the existing embedder")
        yield  # pragma: no cover

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Document]:
        async def inputs() -> AsyncIterator[Document]:
            for item in self._items:
                yield item

        async for item in self._embedder.evaluate_items(inputs(), context):
            yield item


class _IndexStep(DocumentStep):
    """Index-only orchestration: check text ids before retrieving vectors."""

    event_name = "Indexing"

    def __init__(self, store: VectorStore, embedder: Any) -> None:
        super().__init__(cache=False, policy=store.policy)
        self.name = f"{type(store).__name__}.save"
        self._store = store
        self._embedder = embedder

    def config(self) -> Mapping[str, Any]:
        """Indexing is never represented by a processing-cache bucket."""
        return {}

    async def transform(
        self, item: Any, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """The index-only path is driven in bounded windows by evaluate."""
        raise AssertionError("_IndexStep is evaluated as a stream")
        yield  # pragma: no cover

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Document]:
        """Skip present documents before the embedder's cache or provider call."""
        assert self._upstream is not None
        # Preserve the embedder in the declared graph for observers, but pull
        # text directly so destination hits still bypass embedding entirely.
        text_upstream = self._upstream.upstream
        assert text_upstream is not None
        policy = self.policy or context.policy
        overwrite = self.effective_policy(context).recreate_cache
        ready = False
        index_context = replace(
            context, cache_headers_only=_headers_safe(text_upstream)
        )
        async for batch in stream_batches(
            run_step(text_upstream, index_context), policy.cache_batch_size
        ):
            try:
                if not ready:
                    await _with_retries(
                        lambda: self._store.ensure_ready(self._embedder.dimension),
                        policy,
                    )
                    ready = True
                present = (
                    set()
                    if overwrite
                    else await _with_retries(
                        lambda: self._store.existing_ids([item.id for item in batch]),
                        policy,
                    )
                )
            except Exception as error:
                await self._failed_batch(batch, error, context, policy)
                continue
            missing: list[Document] = []
            for item in batch:
                if item.id in present:
                    saved = replace(_public_item(item), cached=True)
                    await context.emit(
                        DocumentEmbedded(
                            document=saved,
                            step=self._embedder.name,
                            event_name=self._embedder.event_name,
                        )
                    )
                    await context.emit(
                        DocumentSaved(
                            document=saved, step=self.name, event_name=self.event_name
                        )
                    )
                    yield saved
                else:
                    missing.append(item)
            if not missing:
                continue

            embedded = run_step(_EmbeddingWindow(self._embedder, missing), context)
            async for documents in stream_batches(embedded, policy.batch_size):
                try:
                    await _with_retries(lambda: self._store.upsert(documents), policy)
                except Exception as error:
                    await self._failed_batch(documents, error, context, policy)
                    continue
                for document in documents:
                    saved = replace(document, cached=False)
                    await context.emit(
                        DocumentSaved(
                            document=saved, step=self.name, event_name=self.event_name
                        )
                    )
                    yield saved

    async def _failed_batch(
        self,
        documents: Sequence[Document],
        error: Exception,
        context: ExecutionContext,
        policy: Policy,
    ) -> None:
        """Report every affected item and honor the sink's terminal error policy."""
        for document in documents:
            await context.emit(
                ItemFailed(
                    uri=document.id,
                    source_id=document.source_id,
                    error=error,
                    step=self.name,
                    event_name=self.event_name,
                )
            )
        if policy.on_error == "raise":
            raise error


class _SaveStep(DocumentStep):
    """The terminal ``PipelineStep[Document]`` :meth:`VectorStore.save` returns.

    Never caches (a sink has nothing to serve from a cache) and always runs
    under the owning :class:`VectorStore`'s own policy — except for
    ``recreate_cache``, which a pipeline-wide rebuild also sets for the sink.

    Declares ``event_name = "Indexing"``: unlike a parser/chunker/embedder,
    the sink family's base class is :class:`VectorStore` — not a
    ``PipelineStep`` at all — so the stage label is declared here, on the
    step :meth:`VectorStore.save` actually returns. Its :attr:`name` still
    names the concrete store (``MilvusVectorStore.save``), so several stores
    in one graph stay distinguishable while sharing the one stage.
    """

    event_name = "Indexing"

    def __init__(self, store: VectorStore) -> None:
        super().__init__(cache=False, policy=store.policy)
        self.name = f"{type(store).__name__}.save"
        self._store = store
        self._pending: list[_PendingSave] = []
        self._batch_lock = asyncio.Lock()
        self._ready_lock = asyncio.Lock()
        self._ready = False

    def config(self) -> Mapping[str, Any]:
        """No cache-identity config: this step never caches (see above)."""
        return {}

    async def transform(
        self, item: Document, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """Upsert one document (batched) and emit ``DocumentSaved``.

        The store is the source of truth for ``cached``: a document the store
        already holds is not re-upserted and is yielded with ``cached=True``;
        one it does not yet have is upserted and yielded with
        ``cached=False`` — regardless of what ``item.cached`` was coming in.

        Raises:
            fastrag.exceptions.VectorStoreError: ``item.embedding`` is
                ``None`` — an unembedded document is a step error, handled
                like any other by ``DocumentStep``'s evaluator, not silently
                skipped here.
        """
        if item.embedding is None:
            raise VectorStoreError(
                f"document {item.id} has no embedding; cannot index it"
            )
        effective_policy = self.policy or context.policy
        await self._ensure_ready(len(item.embedding))
        already_indexed = await self._save_one(
            item,
            effective_policy.batch_size,
            overwrite=self.effective_policy(context).recreate_cache,
        )
        saved = (
            item
            if item.cached == already_indexed
            else replace(item, cached=already_indexed)
        )
        await context.emit(
            DocumentSaved(document=saved, step=self.name, event_name=self.event_name)
        )
        yield saved

    async def _ensure_ready(self, dimension: int) -> None:
        """Call ``VectorStore.ensure_ready`` exactly once per instance."""
        if self._ready:
            return
        async with self._ready_lock:
            if not self._ready:
                await self._store.ensure_ready(dimension)
                self._ready = True

    async def _save_one(
        self, document: Document, batch_size: int, *, overwrite: bool
    ) -> bool:
        """Queue ``document`` for upsert, flushing the batch it lands in.

        Mirrors ``Embedder._embed_one``: joins the accumulating batch,
        flushing immediately once ``batch_size`` documents have queued, or
        scheduling :meth:`_delayed_flush` if it is the first arrival.

        Args:
            document: The document to index.
            batch_size: How many queued documents trigger a flush.
            overwrite: Upsert ``document`` even if the store already holds
                its id.

        Returns:
            Whether the store already held this document (skipped, not
            upserted) — resolved by :meth:`_flush` once its batch's
            existence check comes back. Always ``False`` with ``overwrite``.
        """
        loop = asyncio.get_running_loop()
        future: asyncio.Future[bool] = loop.create_future()
        to_flush: list[_PendingSave] | None = None
        async with self._batch_lock:
            self._pending.append((document, future, overwrite))
            if len(self._pending) >= batch_size:
                to_flush, self._pending = self._pending, []
            elif len(self._pending) == 1:
                asyncio.create_task(self._delayed_flush())
        if to_flush is not None:
            await self._flush(to_flush)
        return await future

    async def _delayed_flush(self) -> None:
        """Flush whatever is still pending after the batch window elapses."""
        await asyncio.sleep(_BATCH_WINDOW_SECONDS)
        to_flush: list[_PendingSave] | None = None
        async with self._batch_lock:
            if self._pending:
                to_flush, self._pending = self._pending, []
        if to_flush is not None:
            await self._flush(to_flush)

    async def _flush(self, batch: list[_PendingSave]) -> None:
        """Check existence once for ``batch``, upsert what's missing, resolve each future.

        One :meth:`VectorStore.existing_ids` call decides which of the batch
        is already indexed; those are excluded from the (at most one)
        :meth:`VectorStore.upsert` call that follows — skipped entirely if
        every document is already present. A failure from either call fails
        every document in the batch; each document's own ``DocumentStep``
        retry calls :meth:`transform` again, resubmitting into a fresh batch.
        Documents queued with ``overwrite`` skip the existence check and are
        always upserted.
        """
        checked = [document.id for document, _, overwrite in batch if not overwrite]
        try:
            present = await self._store.existing_ids(checked) if checked else set()
            missing = [
                document for document, _, _ in batch if document.id not in present
            ]
            if missing:
                await self._store.upsert(missing)
        except Exception as error:  # propagated to every waiter, not swallowed
            for _, future, _ in batch:
                if not future.done():
                    future.set_exception(error)
            return
        for document, future, _ in batch:
            if not future.done():
                future.set_result(document.id in present)


class _SyncStep(DocumentStep):
    """The terminal step :meth:`VectorStore.sync` returns: save, then prune.

    Bound to the step :meth:`VectorStore.save` returns. Records the id of
    every document that step yields, and counts the run's ``ItemFailed``
    events by evaluating it under a context whose sink sees each event on its
    way to the pipeline's own. Shares the ``Indexing`` stage with the save
    step; its :attr:`name` is ``MilvusVectorStore.sync``.
    """

    event_name = "Indexing"

    def __init__(self, store: VectorStore) -> None:
        super().__init__(cache=False, policy=store.policy)
        self.name = f"{type(store).__name__}.sync"
        self._store = store

    def config(self) -> Mapping[str, Any]:
        """Syncing is never represented by a processing-cache bucket."""
        return {}

    async def transform(
        self, item: Any, context: ExecutionContext
    ) -> AsyncIterator[Document]:
        """The sync step only wraps the save step's stream; see evaluate."""
        raise AssertionError("_SyncStep is evaluated as a stream")
        yield  # pragma: no cover

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Document]:
        """Yield every saved document, then delete the stored ones not yielded."""
        assert self._upstream is not None
        sink = context.event_sink
        failures = 0

        async def watch(event: PipelineEvent) -> None:
            nonlocal failures
            if isinstance(event, ItemFailed):
                failures += 1
            if sink is not None:
                await sink(event)

        kept: set[str] = set()
        watched = replace(context, event_sink=watch)
        async for document in run_step(self._upstream, watched):
            kept.add(document.id)
            yield document

        policy = self.policy or context.policy
        stale = sorted(await _with_retries(self._store.all_ids, policy) - kept)
        if not failures:
            for start in range(0, len(stale), _DELETE_BATCH_SIZE):
                ids = stale[start : start + _DELETE_BATCH_SIZE]
                await _with_retries(lambda: self._store.delete(ids), policy)
        await context.emit(
            DocumentsPruned(
                step=self.name,
                stale=len(stale),
                deleted=0 if failures else len(stale),
                failures=failures,
                event_name=self.event_name,
            )
        )
