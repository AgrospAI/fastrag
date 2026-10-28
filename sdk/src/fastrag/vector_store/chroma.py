"""``ChromaVectorStore``, via chromadb's async HTTP client or an embedded store.

A :class:`~fastrag.vector_store.base.VectorStore` backed by Chroma. Chroma has
no BM25 ranking (its document filters are substring/regex matches, not
relevance scores), so this store does not implement
:class:`~fastrag.lexical_store.base.LexicalStore`: pair it with a separate
lexical store for hybrid retrieval.

Every write and query passes vectors, so the collection's default embedding
function is never invoked (and its model never downloaded).

Chroma metadata values must be scalars, so a document's (possibly nested)
metadata is kept as one JSON string; ``source_id``, ``source_uri`` and
``step_config_hash`` are stored as their own metadata keys so ``where``
filters can match them.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import Mapping, Sequence
from typing import Any, Self

import httpx

from fastrag.exceptions import (
    ConfigurationError,
    TransientVectorStoreError,
    VectorStoreError,
)
from fastrag.ingestion.policy import Policy
from fastrag.models import Document
from fastrag.vector_store.base import VectorStore

try:
    import chromadb as _chromadb_impl
    from chromadb.errors import InternalError as _InternalErrorImpl
    from chromadb.errors import NotFoundError as _NotFoundErrorImpl
    from chromadb.errors import RateLimitError as _RateLimitErrorImpl

    _CHROMA_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _CHROMA_AVAILABLE = False

#: The chromadb module, or ``None`` when the optional "chroma" extra is
#: missing; :meth:`ChromaVectorStore.__init__` refuses to run without it.
_chromadb: Any = _chromadb_impl if _CHROMA_AVAILABLE else None
#: chromadb errors worth a retry: the server is overloaded or failed
#: internally, rather than rejecting the request itself.
_TRANSIENT_ERRORS: tuple[type[Exception], ...] = (
    (_InternalErrorImpl, _RateLimitErrorImpl) if _CHROMA_AVAILABLE else ()
)
#: The chromadb error for a collection that does not exist.
_NOT_FOUND_ERRORS: tuple[type[Exception], ...] = (
    (_NotFoundErrorImpl,) if _CHROMA_AVAILABLE else ()
)

MISSING_EXTRA = (
    'ChromaVectorStore needs the "chroma" extra: pip install "fastrag-ai[chroma]"'
)

#: Collection name used when the caller does not name one.
DEFAULT_COLLECTION = "fastrag_documents"

#: Chroma's HNSW distance functions.
METRICS = frozenset({"cosine", "l2", "ip"})

#: Ids per ``get`` page in :meth:`ChromaVectorStore.all_ids`.
ID_PAGE_SIZE = 10000


def _is_transient(error: Exception) -> bool:
    """Return True when ``error`` is a lost connection or an overloaded server."""
    return isinstance(error, (httpx.TransportError, *_TRANSIENT_ERRORS))


class ChromaVectorStore(VectorStore):
    """Vectors in a Chroma server, or in an embedded on-disk Chroma database.

    Give ``url`` to talk to a Chroma server through its async HTTP client, or
    ``path`` to run Chroma in-process against a local directory (its client is
    synchronous, so calls run in a worker thread). Construction never touches
    the network or the disk — the client is built in ``__aenter__``.

    .. code-block:: python

        async with ChromaVectorStore(url="http://localhost:8000") as vector_store:
            async for document in vector_store.save(embedded_documents):
                ...

    ``filter`` is a JSON-encoded Chroma ``where`` object, e.g.
    ``'{"source_uri": "https://example.org/docs"}'``. ``score`` is Chroma's
    own distance, so **lower is better**; results are still best first.

    Args:
        url: Server endpoint, e.g. ``http://localhost:8000``. Exclusive with
            ``path``.
        path: Directory of an embedded persistent database. Exclusive with
            ``url``.
        collection: Name of the Chroma collection. Created on demand.
        headers: Extra HTTP headers, e.g. an auth token, sent with ``url``.
        tenant: Chroma tenant the collection lives in.
        database: Chroma database the collection lives in.
        metric_type: HNSW distance for a new collection: ``cosine``, ``l2`` or
            ``ip``. An existing collection must already use it.
        policy: Execution policy for the terminal step :meth:`save` returns.
    """

    def __init__(
        self,
        *,
        url: str | None = None,
        path: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        headers: Mapping[str, str] | None = None,
        tenant: str = "default_tenant",
        database: str = "default_database",
        metric_type: str = "cosine",
        policy: Policy | None = None,
    ) -> None:
        """Validate the optional extra and store configuration; no I/O yet."""
        super().__init__(policy=policy)
        if _chromadb is None:
            raise ConfigurationError(MISSING_EXTRA)
        if (url is None) == (path is None):
            raise ConfigurationError("ChromaVectorStore needs exactly one of url, path")
        if url is not None:
            parsed = httpx.URL(url)
            if parsed.scheme not in {"http", "https"} or not parsed.host:
                raise ConfigurationError("Chroma url must be an HTTP(S) endpoint")
        if metric_type not in METRICS:
            raise ConfigurationError(
                f"Chroma metric_type must be one of {sorted(METRICS)}"
            )
        self.url = url
        self.path = path
        self.collection = collection
        self.headers = dict(headers or {})
        self.tenant = tenant
        self.database = database
        self.metric_type = metric_type
        self._client: Any = None
        self._collection: Any = None

    async def __aenter__(self) -> Self:
        """Build the client. The HTTP client checks the server is reachable."""
        if self.url is not None:
            parsed = httpx.URL(self.url)
            self._client = await self._call(
                "connect",
                _chromadb.AsyncHttpClient,
                host=parsed.host,
                port=parsed.port or (443 if parsed.scheme == "https" else 80),
                ssl=parsed.scheme == "https",
                headers=self.headers or None,
                tenant=self.tenant,
                database=self.database,
            )
        else:
            self._client = await self._call(
                "connect",
                _chromadb.PersistentClient,
                path=self.path,
                tenant=self.tenant,
                database=self.database,
            )
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Drop the client and collection handle. Safe if never opened."""
        self._client = None
        self._collection = None

    async def _call(self, operation: str, method: Any, /, **kwargs: Any) -> Any:
        """Run one chromadb call, translating its failures into ``VectorStoreError``.

        The async HTTP client's methods are awaited; the embedded client's
        are synchronous and run in a worker thread so they never block the
        event loop.
        """
        try:
            if asyncio.iscoroutinefunction(method):
                return await method(**kwargs)
            return await asyncio.to_thread(method, **kwargs)
        except Exception as error:  # any chromadb failure is a store failure
            message = f"Chroma {operation} failed: {error}"
            if _is_transient(error):
                raise TransientVectorStoreError(message) from error
            raise VectorStoreError(message) from error

    def _require_client(self) -> Any:
        """The open client, or a clear error instead of an ``AttributeError``."""
        if self._client is None:
            raise VectorStoreError(
                "ChromaVectorStore is not open; use "
                "`async with ChromaVectorStore(...) as vector_store:`"
            )
        return self._client

    async def _require_collection(self) -> Any:
        """The collection handle, fetched (not created) on first use."""
        if self._collection is None:
            self._collection = await self._call(
                "get_collection",
                self._require_client().get_collection,
                name=self.collection,
            )
        return self._collection

    async def ensure_ready(self, dimension: int) -> None:
        """Create ``collection`` if absent, or validate its metric and width.

        Chroma fixes a collection's width on its first insert rather than at
        creation, so an existing collection is checked against one stored
        vector; an empty one accepts any width.
        """
        if dimension < 1:
            raise ConfigurationError("vector dimension must be at least 1")
        collection = await self._call(
            "get_or_create_collection",
            self._require_client().get_or_create_collection,
            name=self.collection,
            configuration={"hnsw": {"space": self.metric_type}},
        )
        space = (collection.configuration or {}).get("hnsw", {}) or {}
        if space.get("space", self.metric_type) != self.metric_type:
            raise ConfigurationError(
                "Chroma collection distance metric differs; use a new collection"
            )
        sample = await self._call(
            "get", collection.get, limit=1, include=["embeddings"]
        )
        embeddings = sample.get("embeddings")
        if (
            embeddings is not None
            and len(embeddings)
            and len(embeddings[0]) != dimension
        ):
            raise ConfigurationError(
                "Chroma vector dimension differs; use a new collection"
            )
        self._collection = collection

    async def upsert(self, documents: Sequence[Document]) -> None:
        """Insert or replace ``documents`` by :attr:`~fastrag.models.Document.id`, in one call."""
        if not documents:
            return
        ids: list[str] = []
        embeddings: list[list[float]] = []
        contents: list[str] = []
        metadatas: list[dict[str, Any]] = []
        for document in documents:
            if document.embedding is None:
                raise VectorStoreError(
                    f"document {document.id} has no embedding; cannot index it"
                )
            try:
                metadata_json = json.dumps(dict(document.metadata), allow_nan=False)
            except (TypeError, ValueError) as error:
                raise VectorStoreError(
                    f"document {document.id}: cannot serialize metadata JSON: {error}"
                ) from error
            ids.append(document.id)
            embeddings.append(list(document.embedding))
            contents.append(document.content)
            metadatas.append(
                {
                    "source_id": document.source_id,
                    "source_uri": str(
                        document.metadata.get("source_uri", document.source_id)
                    ),
                    "step_config_hash": document.step_config_hash,
                    "metadata": metadata_json,
                }
            )
        collection = await self._require_collection()
        await self._call(
            "upsert",
            collection.upsert,
            ids=ids,
            embeddings=embeddings,
            documents=contents,
            metadatas=metadatas,
        )

    async def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Fetch the whole batch by id in one ``get``, returning ids only."""
        if not ids:
            return set()
        collection = await self._require_collection()
        result = await self._call("get", collection.get, ids=list(ids), include=[])
        return set(result["ids"])

    async def all_ids(self) -> set[str]:
        """Page through every id with ``get``; empty if the collection does not exist."""
        try:
            collection = await self._require_collection()
        except VectorStoreError as error:
            if isinstance(error.__cause__, _NOT_FOUND_ERRORS):
                return set()
            raise
        ids: set[str] = set()
        while True:
            page = await self._call(
                "get",
                collection.get,
                include=[],
                limit=ID_PAGE_SIZE,
                offset=len(ids),
            )
            ids.update(page["ids"])
            if len(page["ids"]) < ID_PAGE_SIZE:
                return ids

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete ``ids`` in one call; Chroma ignores ids it does not hold."""
        if ids:
            collection = await self._require_collection()
            await self._call("delete", collection.delete, ids=list(ids))

    async def search(
        self, embedding: Sequence[float], *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Return at most ``k`` documents nearest ``embedding``, best first.

        ``score`` is Chroma's distance for the collection's metric, where
        lower means closer; ``rank`` is the position in Chroma's already
        nearest-first order.
        """
        if k < 1:
            raise ConfigurationError("search k must be at least 1")
        if not embedding:
            raise ConfigurationError("query embedding must not be empty")
        collection = await self._require_collection()
        results = await self._call(
            "query",
            collection.query,
            query_embeddings=[list(embedding)],
            n_results=k,
            where=_parse_filter(filter) if filter is not None else None,
            include=["documents", "metadatas", "distances"],
        )
        ids = results["ids"][0] if results["ids"] else []
        contents = (results.get("documents") or [[]])[0]
        metadatas = (results.get("metadatas") or [[]])[0]
        distances = (results.get("distances") or [[]])[0]
        documents: list[Document] = []
        for rank, (id_, content, stored, distance) in enumerate(
            zip(ids, contents, metadatas, distances, strict=True)
        ):
            stored = stored or {}
            metadata = json.loads(stored.get("metadata") or "{}")
            metadata.setdefault("source_uri", stored.get("source_uri", ""))
            metadata.update(score=float(distance), rank=rank)
            documents.append(
                Document(
                    id=id_,
                    source_id=str(stored.get("source_id", "")),
                    content=content or "",
                    metadata=metadata,
                    step_config_hash=str(stored.get("step_config_hash", "")),
                )
            )
        return documents


def _parse_filter(filter: str) -> dict[str, Any]:
    """Decode a backend-native ``where`` filter before sending requests."""
    try:
        value = json.loads(filter)
    except ValueError as error:
        raise ConfigurationError("Chroma filter must be a JSON where object") from error
    if not isinstance(value, dict) or not value:
        raise ConfigurationError("Chroma filter must be a nonempty where object")
    return value
