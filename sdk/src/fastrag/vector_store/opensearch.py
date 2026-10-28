"""Async OpenSearch storage with BM25 and dense k-NN search over one index.

Uses OpenSearch's REST API through the SDK's existing httpx dependency.
Lucene HNSW requires the k-NN plugin; OpenSearch 2.12+ supports the
4096-dimensional embeddings used by the chatbot example. Fusion happens in
the SDK, so no server search pipeline or fusion plugin is required.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Any, Self

import httpx

from fastrag.exceptions import (
    ConfigurationError,
    TransientVectorStoreError,
    VectorStoreError,
)
from fastrag.ingestion.policy import Policy
from fastrag.models import Document
from fastrag.lexical_store import LexicalStore
from fastrag.vector_store.base import VectorStore

#: Hits per ``_search`` page in :meth:`OpenSearchVectorStore.all_ids`, the
#: default ``index.max_result_window``.
ID_PAGE_SIZE = 10000


class OpenSearchVectorStore(VectorStore, LexicalStore):
    """Persist and search documents in an OpenSearch index.

    ``filter`` is a JSON-encoded Query DSL object, e.g.
    ``'{"term": {"source_uri": "https://example.org/docs"}}'``. Metadata is
    retained in ``_source`` without dynamic indexing. HTTPS uses username and
    password authentication with TLS certificate verification disabled.
    """

    def __init__(
        self,
        *,
        url: str,
        index: str = "fastrag_documents",
        user: str | None = None,
        password: str | None = None,
        timeout: float = 30.0,
        policy: Policy | None = None,
    ) -> None:
        """Validate configuration without opening connections."""
        super().__init__(policy=policy)
        parsed = httpx.URL(url)
        if parsed.scheme not in {"http", "https"} or not parsed.host:
            raise ConfigurationError("OpenSearch url must be an HTTP(S) endpoint")
        if (
            not re.fullmatch(r"[a-z0-9][a-z0-9_.-]*", index)
            or len(index.encode()) > 255
        ):
            raise ConfigurationError("OpenSearch index must be a valid lowercase name")
        if (user is None) != (password is None):
            raise ConfigurationError("OpenSearch needs both user and password")
        if timeout <= 0:
            raise ConfigurationError("OpenSearch timeout must be positive")
        self.url = url.rstrip("/") + "/"
        self.index = index
        self.user = user
        self.password = password
        self.timeout = timeout
        self._client: httpx.AsyncClient | None = None
        self._dimension: int | None = None

    async def __aenter__(self) -> Self:
        """Open the shared HTTP connection pool."""
        self._client = httpx.AsyncClient(
            base_url=self.url,
            auth=(self.user, self.password or "") if self.user is not None else None,
            verify=False,
            timeout=self.timeout,
        )
        return self

    async def __aexit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        """Release the HTTP client, even after a failed indexing run."""
        if self._client is not None:
            await self._client.aclose()
            self._client = None
        self._dimension = None

    async def _request(self, method: str, path: str, **kwargs: Any) -> httpx.Response:
        if self._client is None:
            raise VectorStoreError("OpenSearchVectorStore is not open; use async with")
        try:
            response = await self._client.request(method, path, **kwargs)
            response.raise_for_status()
            return response
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            error_type = (
                TransientVectorStoreError
                if status in {408, 425, 429} or status >= 500
                else VectorStoreError
            )
            raise error_type(
                f"OpenSearch {method} {path} failed with HTTP {status}"
            ) from error
        except httpx.HTTPError as error:
            raise TransientVectorStoreError(
                f"OpenSearch {method} {path} failed: {type(error).__name__}"
            ) from error

    async def ensure_ready(self, dimension: int) -> None:
        """Create the index or validate its existing vector and text mappings."""
        if self._dimension == dimension:
            return
        if dimension < 1:
            raise ConfigurationError("vector dimension must be at least 1")
        try:
            response = await self._request("GET", f"{self.index}/_mapping")
        except VectorStoreError as error:
            cause = error.__cause__
            if (
                not isinstance(cause, httpx.HTTPStatusError)
                or cause.response.status_code != 404
            ):
                raise
            body = {
                "settings": {"index": {"knn": True}},
                "mappings": {
                    "dynamic": "strict",
                    "properties": {
                        "id": {"type": "keyword"},
                        "source_id": {"type": "keyword"},
                        "source_uri": {"type": "keyword"},
                        "step_config_hash": {"type": "keyword"},
                        "content": {"type": "text", "similarity": "BM25"},
                        "metadata": {"type": "object", "enabled": False},
                        "vector": {
                            "type": "knn_vector",
                            "dimension": dimension,
                            "method": {
                                "name": "hnsw",
                                "engine": "lucene",
                                "space_type": "cosinesimil",
                            },
                        },
                    },
                },
            }
            try:
                await self._request("PUT", self.index, json=body)
            except VectorStoreError as create_error:
                cause = create_error.__cause__
                if not isinstance(cause, httpx.HTTPStatusError):
                    raise
                payload = cause.response.json().get("error", {})
                if (
                    not isinstance(payload, dict)
                    or payload.get("type") != "resource_already_exists_exception"
                ):
                    raise
            response = await self._request("GET", f"{self.index}/_mapping")
        properties = response.json()[self.index]["mappings"]["properties"]
        vector = properties.get("vector", {})
        if vector.get("type") != "knn_vector" or vector.get("dimension") != dimension:
            raise ConfigurationError(
                "OpenSearch index vector dimension or type differs; use a new index"
            )
        if properties.get("content", {}).get("type") != "text":
            raise ConfigurationError(
                "OpenSearch index needs a content text field for BM25"
            )
        self._dimension = dimension

    async def existing_ids(self, ids: Sequence[str]) -> set[str]:
        """Check the entire batch in one real-time multi-get request."""
        if not ids:
            return set()
        response = await self._request(
            "POST",
            f"{self.index}/_mget",
            json={"ids": list(ids)},
            params={"_source": "false"},
        )
        rows = response.json()["docs"]
        if any("error" in row for row in rows):
            raise VectorStoreError("OpenSearch multi-get returned an item error")
        return {row["_id"] for row in rows if row.get("found")}

    async def all_ids(self) -> set[str]:
        """Page through every id sorted on the ``id`` keyword, with ``search_after``.

        Returns an empty set when the index does not exist yet.
        """
        ids: set[str] = set()
        body: dict[str, Any] = {
            "size": ID_PAGE_SIZE,
            "_source": False,
            "sort": [{"id": "asc"}],
            "query": {"match_all": {}},
        }
        while True:
            try:
                response = await self._request(
                    "POST", f"{self.index}/_search", json=body
                )
            except VectorStoreError as error:
                cause = error.__cause__
                if (
                    isinstance(cause, httpx.HTTPStatusError)
                    and cause.response.status_code == 404
                ):
                    return ids
                raise
            payload = response.json()
            if payload.get("timed_out") or payload.get("_shards", {}).get("failed", 0):
                raise TransientVectorStoreError(
                    "OpenSearch id listing returned incomplete results"
                )
            hits = payload["hits"]["hits"]
            ids.update(hit["_id"] for hit in hits)
            if len(hits) < ID_PAGE_SIZE:
                return ids
            body["search_after"] = hits[-1]["sort"]

    async def delete(self, ids: Sequence[str]) -> None:
        """Bulk delete by id and refresh so searches immediately stop seeing them."""
        if not ids:
            return
        lines = [
            json.dumps({"delete": {"_index": self.index, "_id": id_}}) for id_ in ids
        ]
        response = await self._request(
            "POST",
            "_bulk",
            content="\n".join(lines) + "\n",
            headers={"content-type": "application/x-ndjson"},
            params={"refresh": "wait_for"},
        )
        payload = response.json()
        # Deleting an absent id reports "not_found" without an error.
        statuses = [
            item["delete"]["status"]
            for item in payload["items"]
            if item["delete"].get("error")
        ]
        if statuses:
            error_type = (
                TransientVectorStoreError
                if all(s == 429 or s >= 500 for s in statuses)
                else VectorStoreError
            )
            raise error_type(
                f"OpenSearch bulk delete failed for {len(statuses)} documents: HTTP {statuses}"
            )

    async def upsert(self, documents: Sequence[Document]) -> None:
        """Bulk index by id and refresh so searches immediately see writes."""
        if not documents:
            return
        lines: list[str] = []
        for document in documents:
            if not document.embedding:
                raise VectorStoreError(f"document {document.id} has no embedding")
            if (
                self._dimension is not None
                and len(document.embedding) != self._dimension
            ):
                raise VectorStoreError(
                    f"document {document.id} has the wrong embedding dimension"
                )
            lines.extend(
                [
                    json.dumps({"index": {"_index": self.index, "_id": document.id}}),
                    json.dumps(
                        {
                            "id": document.id,
                            "source_id": document.source_id,
                            "source_uri": str(
                                document.metadata.get("source_uri", document.source_id)
                            ),
                            "step_config_hash": document.step_config_hash,
                            "content": document.content,
                            "metadata": document.metadata,
                            "vector": document.embedding,
                        },
                        allow_nan=False,
                    ),
                ]
            )
        response = await self._request(
            "POST",
            "_bulk",
            content="\n".join(lines) + "\n",
            headers={"content-type": "application/x-ndjson"},
            params={"refresh": "wait_for"},
        )
        payload = response.json()
        if payload.get("errors"):
            statuses = [
                item["index"]["status"]
                for item in payload["items"]
                if item["index"].get("error")
            ]
            error_type = (
                TransientVectorStoreError
                if all(s == 429 or s >= 500 for s in statuses)
                else VectorStoreError
            )
            raise error_type(
                f"OpenSearch bulk indexing failed for {len(statuses)} documents: HTTP {statuses}"
            )

    async def search(
        self, embedding: Sequence[float], *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Search dense vectors with cosine similarity and optional native filter."""
        if not embedding:
            raise ConfigurationError("query embedding must not be empty")
        options: dict[str, Any] = {"vector": list(embedding), "k": k}
        if filter is not None:
            options["filter"] = _parse_filter(filter)
        return await self._search({"knn": {"vector": options}}, k=k)

    async def lexical_search(
        self, query: str, *, k: int, filter: str | None = None
    ) -> list[Document]:
        """Search the analyzed content field using BM25."""
        query_dsl: dict[str, Any] = {"match": {"content": query}}
        if filter is not None:
            query_dsl = {
                "bool": {"must": [query_dsl], "filter": [_parse_filter(filter)]}
            }
        return await self._search(query_dsl, k=k)

    async def _search(self, query: dict[str, Any], *, k: int) -> list[Document]:
        if k < 1:
            raise ConfigurationError("search k must be at least 1")
        response = await self._request(
            "POST",
            f"{self.index}/_search",
            json={"size": k, "query": query, "_source": {"excludes": ["vector"]}},
        )
        payload = response.json()
        if payload.get("timed_out") or payload.get("_shards", {}).get("failed", 0):
            raise TransientVectorStoreError(
                "OpenSearch search returned incomplete results"
            )
        documents = []
        for rank, hit in enumerate(payload["hits"]["hits"]):
            source = hit["_source"]
            metadata = dict(source.get("metadata", {}))
            metadata.setdefault("source_uri", source.get("source_uri", ""))
            metadata.update(score=float(hit["_score"]), rank=rank)
            documents.append(
                Document(
                    id=hit["_id"],
                    source_id=source.get("source_id", ""),
                    content=source["content"],
                    metadata=metadata,
                    step_config_hash=source.get("step_config_hash", ""),
                )
            )
        return documents


def _parse_filter(filter: str) -> dict[str, Any]:
    """Decode a backend-native Query DSL filter before sending requests."""
    try:
        value = json.loads(filter)
    except ValueError as error:
        raise ConfigurationError(
            "OpenSearch filter must be a JSON Query DSL object"
        ) from error
    if not isinstance(value, dict) or not value:
        raise ConfigurationError(
            "OpenSearch filter must be a nonempty Query DSL object"
        )
    return value
