"""``CohereReranker``: reranking against any endpoint speaking the Cohere protocol."""

from __future__ import annotations

import asyncio
from copy import deepcopy
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

import httpx

from fastrag.exceptions import ConfigurationError, RerankError, TransientRerankError
from fastrag.models import Document
from fastrag.retrieval.reranking.base import Reranker

#: Status codes worth retrying: rate limiting, and anything the server calls
#: its own fault.
_TRANSIENT_STATUS = frozenset({408, 425, 429, 500, 502, 503, 504})


class CohereReranker(Reranker):
    """Rerank against any endpoint speaking the Cohere rerank protocol.

    Posts JSON straight to ``{base_url}/rerank`` with ``httpx`` rather than
    importing the ``cohere`` SDK, so reranking needs no extra. LiteLLM, Jina,
    TEI, Infinity, and vLLM all speak the same ``{model, query, documents,
    top_n}`` request and ``results: [{index, relevance_score}]`` response —
    only the model name and ``base_url`` change. Returns
    :class:`~fastrag.retrieval.reranking.base.Reranker`: same HTTP contract,
    same error translation, adapted to return :class:`~fastrag.models.Document`
    (with ``score``/``rank`` metadata).

    .. code-block:: python

        import os

        from fastrag.retrieval import CohereReranker

        reranker = CohereReranker(
            model="rerank-v3.5",
            base_url=os.environ["RERANK_BASE_URL"],
            api_key=os.environ["COHERE_API_KEY"],
        )

    Args:
        model: Model name, e.g. ``bge-reranker-v2-m3`` or ``rerank-v3.5``.
        base_url: Endpoint base URL. Requests go to ``{base_url}/rerank`` —
            Cohere's own API needs the ``/v1`` already carried in the
            default; a server that mounts the route elsewhere needs the full
            prefix here.
        api_key: Bearer token.
        extra_body: Additional JSON request fields; matching fields override defaults.
        timeout: Seconds per call.
    """

    def __init__(
        self,
        *,
        model: str,
        base_url: str = "https://api.cohere.com/v1",
        api_key: str,
        timeout: float = 30.0,
        extra_body: Mapping[str, Any] | None = None,
    ) -> None:
        """Validate configuration; no I/O happens until :meth:`rerank`."""
        if not model:
            raise ConfigurationError("CohereReranker.model must not be empty")
        if not api_key:
            raise ConfigurationError("CohereReranker.api_key must not be empty")
        if timeout <= 0:
            raise ConfigurationError("CohereReranker.timeout must be positive")
        self.model = model
        self.base_url = base_url
        self.api_key = api_key
        self.timeout = timeout
        self.extra_body = deepcopy(dict(extra_body or {}))
        self._client: httpx.AsyncClient | None = None

    @property
    def model_id(self) -> str:
        """The configured model name."""
        return self.model

    async def rerank(
        self, query: str, documents: Sequence[Document], *, top_k: int
    ) -> list[Document]:
        """Score ``documents`` against ``query`` through the rerank endpoint.

        No provider exception escapes: everything is translated into the
        Fastrag hierarchy, with timeouts, rate limits, and server errors
        marked transient so a caller wrapping this in retries gets correct
        behaviour for free.

        Raises:
            fastrag.exceptions.ConfigurationError: ``top_k`` is less than 1.
            fastrag.exceptions.RerankError: The request failed non-transiently,
                or the response was malformed.
            fastrag.exceptions.TransientRerankError: The request failed for a
                reason that may succeed on retry.
        """
        if top_k < 1:
            raise ConfigurationError("CohereReranker.rerank top_k must be at least 1")
        if not documents:
            return []

        payload: dict[str, Any] = {
            "model": self.model,
            "query": query,
            "documents": [document.content for document in documents],
            "top_n": top_k,
            **self.extra_body,
        }

        client = self._http_client()
        url = f"{self.base_url.rstrip('/')}/rerank"
        try:
            async with asyncio.timeout(self.timeout):
                response = await client.post(url, json=payload)
                response.raise_for_status()
        except httpx.HTTPStatusError as error:
            status = error.response.status_code
            message = f"rerank failed with HTTP {status}"
            if status in _TRANSIENT_STATUS:
                raise TransientRerankError(message) from error
            raise RerankError(message) from error
        except httpx.HTTPError as error:
            raise TransientRerankError(f"rerank failed: {error}") from error

        return _to_documents(response.json(), documents, top_k)

    async def close(self) -> None:
        """Close the shared HTTP client. Safe to call more than once."""
        if self._client is not None and not self._client.is_closed:
            await self._client.aclose()
        self._client = None

    def _http_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(
                headers={"authorization": f"Bearer {self.api_key}"},
            )
        return self._client


def _to_documents(
    payload: Any, documents: Sequence[Document], top_k: int
) -> list[Document]:
    """Map a rerank response back onto the input documents, by index.

    Sorts descending by ``relevance_score`` and trims to ``top_k``
    defensively, in case the server did not honour ``top_n``. A missing key,
    a non-numeric score, or an out-of-range ``index`` raises
    :class:`~fastrag.exceptions.RerankError` naming what was wrong, never an
    ``IndexError`` or a ``KeyError`` from inside the SDK.
    """
    if not isinstance(payload, dict) or "results" not in payload:
        raise RerankError("rerank response missing 'results'")
    results = payload["results"]
    if not isinstance(results, list):
        raise RerankError("rerank response 'results' must be a list")

    parsed: list[tuple[int, float]] = []
    for result in results:
        if not isinstance(result, dict):
            raise RerankError(
                f"rerank response result must be an object, got {result!r}"
            )
        index = result.get("index")
        score = result.get("relevance_score")
        if not isinstance(index, int) or not (0 <= index < len(documents)):
            raise RerankError(f"rerank response index out of range: {index!r}")
        if not isinstance(score, (int, float)) or isinstance(score, bool):
            raise RerankError(
                f"rerank response relevance_score is not numeric: {score!r}"
            )
        parsed.append((index, float(score)))

    parsed.sort(key=lambda pair: pair[1], reverse=True)
    return [
        replace(
            documents[index],
            metadata={**documents[index].metadata, "score": score, "rank": rank},
        )
        for rank, (index, score) in enumerate(parsed[:top_k])
    ]
