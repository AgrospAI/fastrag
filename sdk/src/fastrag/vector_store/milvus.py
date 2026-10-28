"""``MilvusVectorStore``, via pymilvus' async client.

A :class:`~fastrag.vector_store.base.VectorStore` backed by Milvus. It needs
only what the sink, sync and the read path use — open, ensure the collection,
check which ids already exist, upsert, list and delete ids, search, close.
Filtering is not ported: :meth:`MilvusVectorStore.search` takes a
backend-native filter string directly, per
:meth:`~fastrag.vector_store.base.VectorStore.search`'s own signature.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, Self

from fastrag.exceptions import (
    ConfigurationError,
    TransientVectorStoreError,
    VectorStoreError,
)
from fastrag.ingestion.policy import Policy
from fastrag.models import Document
from fastrag.lexical_store import LexicalStore
from fastrag.vector_store.base import VectorStore

if TYPE_CHECKING:
    from pymilvus import AsyncMilvusClient

try:
    from pymilvus import AsyncMilvusClient as _AsyncMilvusClientImpl
    from pymilvus import DataType as _DataTypeImpl
    from pymilvus import Function as _FunctionImpl
    from pymilvus import FunctionType as _FunctionTypeImpl
    from pymilvus import MilvusException as _MilvusExceptionImpl
    from pymilvus import Status as _StatusImpl

    _MILVUS_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised by installs without the extra
    _MILVUS_AVAILABLE = False

#: The client class and schema enum, or ``None`` when the optional "milvus"
#: extra is missing; :meth:`MilvusVectorStore.__init__` refuses to run
#: without them.
_AsyncMilvusClient: Any = _AsyncMilvusClientImpl if _MILVUS_AVAILABLE else None
_Function: Any = _FunctionImpl if _MILVUS_AVAILABLE else None
_FunctionType: Any = _FunctionTypeImpl if _MILVUS_AVAILABLE else None
_DataType: Any = _DataTypeImpl if _MILVUS_AVAILABLE else None
#: pymilvus's own exception type and status-code enum, used only by
#: :func:`_is_transient` to tell a lost connection from a real failure —
#: never constructed here, so a placeholder ``None`` is fine when the
#: "milvus" extra is missing (``_call`` is unreachable in that case, since
#: ``__init__`` already refused to run).
_MilvusException: Any = _MilvusExceptionImpl if _MILVUS_AVAILABLE else None
_Status: Any = _StatusImpl if _MILVUS_AVAILABLE else None

MISSING_EXTRA = (
    'MilvusVectorStore needs the "milvus" extra: pip install "fastrag-ai[milvus]"'
)

#: Length caps of the VARCHAR columns, checked locally so one oversized
#: document fails with a message naming the field instead of a Milvus batch
#: rejection.
ID_MAX_LENGTH = 64  # Document.id is 32 hex chars; headroom for a future format
CONTENT_MAX_LENGTH = 65535
SOURCE_URI_MAX_LENGTH = 2048
METADATA_MAX_BYTES = 65536

#: Collection name used when the caller does not name one.
DEFAULT_COLLECTION = "fastrag_documents"

#: Ids per ``query`` page in :meth:`MilvusVectorStore.all_ids`, below Milvus's
#: 16384-row cap on one query's results.
ID_PAGE_SIZE = 10000


def _is_transient(error: Exception) -> bool:
    """Return True when ``error`` is a lost or refused connection, hence worth a retry.

    Mirrors :func:`fastrag._sql.is_transient`'s shape for pymilvus's own
    exception type: whether a call fails during :meth:`MilvusVectorStore.ensure_ready`
    (first contact) or mid-run (a dropped connection during ``upsert`` or
    ``search``), pymilvus surfaces it the same way — a channel that never
    became ready, wrapped by pymilvus itself as a ``MilvusException`` carrying
    ``Status.CONNECT_FAILED`` (verified empirically: pointing the client at an
    unreachable host raises exactly this). A schema conflict, a bad filter
    string, or an oversized batch raises a ``MilvusException`` with a
    different code, and retrying it would only fail the same way again.
    """
    if _MilvusException is None or not isinstance(error, _MilvusException):
        return False
    # `_MilvusException` is typed `Any` (it's an optional-extra import), so
    # the `isinstance` above cannot narrow `error` for the type checker;
    # `getattr` reads `.code` without asserting a static type `error` doesn't
    # statically have.
    return getattr(error, "code", None) == _Status.CONNECT_FAILED


class MilvusVectorStore(VectorStore, LexicalStore):
    """Vectors in a Milvus or Milvus-Lite server.

    Choose this when the corpus outgrows in-process search, or when several
    processes index or query the same data. Construction never touches the
    network — the ``pymilvus`` client is built in ``__aenter__`` and
    closed in ``__aexit__``.

    .. code-block:: python

        async with MilvusVectorStore(url=MILVUS_URI) as vector_store:
            saved_documents = vector_store.save(embedded_documents)
            async for document in saved_documents:
                ...

    Args:
        url: Server endpoint, e.g. ``http://localhost:19530``.
        collection: Name of the Milvus collection documents are upserted
            into. Created automatically, on demand, if absent.
        user: Username, combined with ``password`` into the ``token`` Milvus
            authenticates with. ``None`` connects without credentials.
        password: Password paired with ``user``.
        metric_type: Distance metric for the vector index. ``COSINE`` suits
            embedding models whose vectors are not length-normalized; ``L2``
            is the other common choice.
        enable_bm25: Create an analyzed content field, BM25 function and sparse
            index. Requires Milvus 2.5+. Existing dense-only collections must
            be reindexed into a new collection to use this option.
        timeout: Seconds per client operation.
        policy: Execution policy for the terminal step :meth:`save` returns.
    """

    def __init__(
        self,
        *,
        url: str,
        collection: str = DEFAULT_COLLECTION,
        user: str | None = None,
        password: str | None = None,
        metric_type: str = "COSINE",
        enable_bm25: bool = False,
        timeout: float = 30.0,
        policy: Policy | None = None,
    ) -> None:
        """Validate the optional extra and store configuration; no I/O yet."""
        super().__init__(policy=policy)
        if _AsyncMilvusClient is None:
            raise ConfigurationError(MISSING_EXTRA)
        self.url = url
        self.collection = collection
        self.user = user
        self.password = password
        self.metric_type = metric_type
        self.enable_bm25 = enable_bm25
        self.timeout = timeout
        self._client: AsyncMilvusClient | None = None

    async def __aenter__(self) -> Self:
        """Open the pymilvus async client. The connection itself is lazy."""
        kwargs: dict[str, Any] = {"uri": self.url, "timeout": self.timeout}
        if self.user is not None:
            kwargs["token"] = f"{self.user}:{self.password}"
        self._client = _AsyncMilvusClient(**kwargs)
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Close the client and release its connection. Safe if never opened."""
        if self._client is not None:
            await self._client.close()
            self._client = None

    def _require_client(self) -> AsyncMilvusClient:
        """The open client, or a clear error instead of an ``AttributeError``."""
        if self._client is None:
            raise VectorStoreError(
                "MilvusVectorStore is not open; use "
                "`async with MilvusVectorStore(...) as vector_store:`"
            )
        return self._client

    async def _call(self, operation: str, coro: Any) -> Any:
        """Run a pymilvus call, translating its failures into ``VectorStoreError``.

        A pymilvus ``MilvusException`` is not a Fastrag type; wrapping it
        here keeps the typed hierarchy intact. The single place that decides
        *which* ``VectorStoreError`` subclass a failure becomes — see
        :func:`_is_transient` — so every caller of ``_call`` (``ensure_ready``,
        ``upsert``, ``existing_ids``, ``search``) gets the same classification
        for free, with no per-method ``except`` clause of its own.
        """
        try:
            return await coro
        except Exception as error:  # any pymilvus failure is a store failure
            message = f"{operation} failed: {error}"
            if _is_transient(error):
                raise TransientVectorStoreError(message) from error
            raise VectorStoreError(message) from error

    async def ensure_ready(self, dimension: int) -> None:
        """Create ``collection`` for ``dimension``-wide vectors if absent.

        Idempotent: safe to call on every run, including one where the
        collection already exists. The schema mirrors
        :class:`~fastrag.models.Document`: id and provenance are columns so
        they filter without reading the JSON, and the content and metadata
        live alongside the vector.
        """
        client = self._require_client()
        if await self._call(
            "has_collection", client.has_collection(collection_name=self.collection)
        ):
            description = await self._call(
                "describe_collection",
                client.describe_collection(collection_name=self.collection),
            )
            fields = {field["name"]: field for field in description["fields"]}
            if fields.get("vector", {}).get("params", {}).get("dim") != dimension:
                raise ConfigurationError(
                    "Milvus vector dimension differs; use a new collection"
                )
            if self.enable_bm25 and "sparse" not in fields:
                raise ConfigurationError(
                    "Milvus collection has no BM25 schema; reindex into a new collection"
                )
            await self._call(
                "load_collection",
                client.load_collection(collection_name=self.collection),
            )
            return

        data_type = _DataType
        if data_type is None:  # unreachable: __init__ refused without the extra
            raise ConfigurationError(MISSING_EXTRA)

        schema = client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(
            field_name="id",
            datatype=data_type.VARCHAR,
            max_length=ID_MAX_LENGTH,
            is_primary=True,
        )
        schema.add_field(
            field_name="content",
            datatype=data_type.VARCHAR,
            max_length=CONTENT_MAX_LENGTH,
            enable_analyzer=self.enable_bm25,
        )
        schema.add_field(
            field_name="source_uri",
            datatype=data_type.VARCHAR,
            max_length=SOURCE_URI_MAX_LENGTH,
        )
        schema.add_field(field_name="metadata", datatype=data_type.JSON)
        schema.add_field(
            field_name="vector",
            datatype=data_type.FLOAT_VECTOR,
            dim=dimension,
        )

        if self.enable_bm25:
            schema.add_field(
                field_name="sparse", datatype=data_type.SPARSE_FLOAT_VECTOR
            )
            schema.add_function(
                _Function(
                    name="content_bm25",
                    input_field_names=["content"],
                    output_field_names=["sparse"],
                    function_type=_FunctionType.BM25,
                )
            )

        index_params = client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            index_type="AUTOINDEX",
            metric_type=self.metric_type,
        )

        if self.enable_bm25:
            index_params.add_index(
                field_name="sparse",
                index_type="SPARSE_INVERTED_INDEX",
                metric_type="BM25",
                params={"inverted_index_algo": "DAAT_MAXSCORE"},
            )

        await self._call(
            "create_collection",
            client.create_collection(
                collection_name=self.collection,
                schema=schema,
                index_params=index_params,
            ),
        )
        await self._call(
            "load_collection", client.load_collection(collection_name=self.collection)
        )

    async def upsert(self, documents: Sequence[Document]) -> None:
        """Insert or replace ``documents`` by :attr:`~fastrag.models.Document.id`, in one request.

        Raises:
            VectorStoreError: A document is missing its embedding (checked
                again defensively — the terminal step should already have
                rejected it), its UTF-8 content exceeds the column cap, or its
                serialized metadata exceeds the JSON cap.
        """
        data: list[dict[str, Any]] = []
        for document in documents:
            if document.embedding is None:
                raise VectorStoreError(
                    f"document {document.id} has no embedding; cannot index it"
                )
            content = document.content
            if len(content.encode("utf-8")) > CONTENT_MAX_LENGTH:
                raise VectorStoreError(
                    f"document {document.id}: content exceeds the "
                    f"{CONTENT_MAX_LENGTH}-byte VARCHAR cap of collection "
                    f"{self.collection!r}"
                )
            # Use the client's encoder: JSON's default ASCII escaping/spaces
            # would overestimate UTF-8 and misclassify valid near-limit values.
            from pymilvus.client.entity_helper import convert_to_json

            try:
                metadata_bytes = len(convert_to_json(dict(document.metadata)))
            except Exception as error:
                raise VectorStoreError(
                    f"document {document.id}: cannot serialize metadata JSON: {error}"
                ) from error
            if metadata_bytes > METADATA_MAX_BYTES:
                raise VectorStoreError(
                    f"document {document.id}: metadata JSON is {metadata_bytes} bytes; "
                    f"exceeds {METADATA_MAX_BYTES}-byte cap of collection {self.collection!r}; "
                    "reduce caller metadata or use smaller structural chunks"
                )
            source_uri = str(document.metadata.get("source_uri", document.source_id))
            data.append(
                {
                    "id": document.id,
                    "content": content,
                    "source_uri": source_uri,
                    "metadata": dict(document.metadata),
                    "vector": list(document.embedding),
                }
            )
        if data:
            await self._call(
                "upsert",
                self._require_client().upsert(
                    collection_name=self.collection, data=data
                ),
            )

    async def search(
        self, embedding: Sequence[float], *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Return at most ``k`` documents nearest ``embedding``, best first.

        One ``search`` call, requesting ``search_params={"metric_type":
        self.metric_type}`` explicitly so the query is scored the same way
        the collection's own index was built (:meth:`ensure_ready`). pymilvus
        already returns hits best-first for whichever metric that is, so no
        reordering is needed: ``score`` is each hit's own ``distance`` field,
        and ``rank`` its position in that already-correct order.

        Queries with ``consistency_level="Strong"`` for read-your-writes:
        pymilvus's client-wide default, ``"Bounded"``, may not yet reflect
        an :meth:`upsert` this same process just issued, and a caller
        retrieving right after indexing must see it.
        """
        if k < 1:
            raise ConfigurationError("search k must be at least 1")
        client = self._require_client()
        results = await self._call(
            "search",
            client.search(
                collection_name=self.collection,
                data=[list(embedding)],
                anns_field="vector",
                limit=k,
                filter=filter or "",
                output_fields=["content", "source_uri", "metadata"],
                search_params={"metric_type": self.metric_type},
                consistency_level="Strong",
            ),
        )
        return self._documents(results)

    async def lexical_search(
        self, query: str, *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Rank text using Milvus's server-side BM25 function and sparse index."""
        if not self.enable_bm25:
            raise ConfigurationError("Milvus lexical search requires enable_bm25=True")
        if k < 1:
            raise ConfigurationError("search k must be at least 1")
        results = await self._call(
            "lexical_search",
            self._require_client().search(
                collection_name=self.collection,
                data=[query],
                anns_field="sparse",
                limit=k,
                filter=filter or "",
                output_fields=["content", "source_uri", "metadata"],
                search_params={"metric_type": "BM25"},
                consistency_level="Strong",
            ),
        )
        return self._documents(results)

    @staticmethod
    def _documents(results: Any) -> list[Document]:
        """Convert either dense or sparse hits into the common document contract."""
        hits = results[0] if results else []
        documents: list[Document] = []
        for rank, hit in enumerate(hits):
            entity = hit.get("entity", {})
            metadata = dict(entity.get("metadata", {}) or {})
            metadata.setdefault("source_uri", entity.get("source_uri", ""))
            metadata["score"] = float(hit.get("distance", 0.0))
            metadata["rank"] = rank
            documents.append(
                Document(
                    id=str(hit.get("id")),
                    source_id="",
                    content=entity.get("content", ""),
                    metadata=metadata,
                )
            )
        return documents

    async def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Query ``collection`` once for which of ``ids`` are already indexed.

        One ``query`` call for the whole batch, filtering on the primary key
        (``id in [...]``) — never one call per id — so this scales with the
        number of batches :meth:`~fastrag.vector_store.base.VectorStore.save`'s
        terminal step flushes, not with the number of documents in them.

        Queries with ``consistency_level="Strong"`` for read-your-writes:
        pymilvus's client-wide default, ``"Bounded"``, may not yet reflect
        an ``upsert`` this same process just issued — a staleness window
        here is not an acceptable trade for the latency it would save.
        """
        if not ids:
            return set()
        client = self._require_client()
        rows = await self._call(
            "query",
            client.query(
                collection_name=self.collection,
                filter=f"id in {json.dumps(list(ids))}",
                output_fields=["id"],
                consistency_level="Strong",
            ),
        )
        return {row["id"] for row in rows}

    async def all_ids(self) -> set[str]:
        """Page through every id in ``collection``, in id order.

        ``AsyncMilvusClient`` has no ``query_iterator``, so this pages the way
        pymilvus's own iterator does: ``iterator`` asks the server for each
        page sorted by primary key, and the next page filters on ids after
        the last one seen. ``offset`` paging would stop at Milvus's 16384-row
        result window.
        """
        client = self._require_client()
        if not await self._call(
            "has_collection", client.has_collection(collection_name=self.collection)
        ):
            return set()
        ids: set[str] = set()
        last: str | None = None
        while True:
            rows = await self._call(
                "query",
                client.query(
                    collection_name=self.collection,
                    filter='id != ""' if last is None else f"id > {json.dumps(last)}",
                    output_fields=["id"],
                    limit=ID_PAGE_SIZE,
                    consistency_level="Strong",
                    iterator="True",
                    reduce_stop_for_best="True",
                ),
            )
            ids.update(row["id"] for row in rows)
            if len(rows) < ID_PAGE_SIZE:
                return ids
            last = max(row["id"] for row in rows)

    async def delete(self, ids: Sequence[str]) -> None:
        """Delete ``ids`` by primary key in one request."""
        if ids:
            await self._call(
                "delete",
                self._require_client().delete(
                    collection_name=self.collection, ids=list(ids)
                ),
            )
