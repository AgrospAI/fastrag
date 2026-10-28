"""Explicit widths reach actual clients and returned shapes obey the contract."""

import json

import httpx
import pytest

from fastrag.exceptions import (
    ConfigurationError,
    EmbeddingError,
    TransientEmbeddingError,
)
from fastrag.ingestion.steps.embedding import OpenAIEmbedder


def test_explicit_dimensions_reach_constructor_and_identity():
    embedder = OpenAIEmbedder(model="fixture", api_key="fixture", dimensions=17)
    assert embedder._client.dimensions == 17
    assert embedder._client._invocation_params["dimensions"] == 17
    assert embedder.dimension == 17
    assert embedder.model_id == "fixture@17"
    assert embedder.config()["dimensions"] == 17
    assert (
        embedder.config_hash
        != OpenAIEmbedder(model="fixture", api_key="fixture", dimensions=18).config_hash
    )


@pytest.mark.parametrize("dimensions", [0, -1, True, 2.5, "8"])
def test_dimensions_require_positive_integer(dimensions):
    with pytest.raises(ConfigurationError, match="positive integer"):
        OpenAIEmbedder(model="fixture", api_key="fixture", dimensions=dimensions)


@pytest.mark.parametrize("explicit", [False, True])
async def test_real_http_width_request_and_batch_alignment(explicit):
    openai = pytest.importorskip("openai")
    payloads = []
    width = 17 if explicit else 1536

    def handle(request):
        body = json.loads(request.content)
        payloads.append(body)
        # Deliberately use a different default to expose missing explicit dimensions.
        actual_width = body.get("dimensions", 1536)
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": [float(i + 1)] * actual_width}
                    for i in range(len(body["input"]))
                ]
            },
        )

    embedder = OpenAIEmbedder(
        model="fixture" if explicit else "text-embedding-ada-002",
        dimensions=width if explicit else None,
        api_key="fixture",
        base_url="https://provider.test/v1",
        batch_size=2,
        extra_body={"provider_options": {"task": "retrieval"}},
    )
    async with openai.AsyncOpenAI(
        api_key="fixture",
        base_url="https://provider.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as provider:
        embedder._client.async_client = provider.embeddings
        vectors = await embedder.embed(["alpha", "beta", "gamma"])
    assert len(vectors) == 3 and all(len(vector) == width for vector in vectors)
    assert [vector[0] for vector in vectors] == [1.0, 2.0, 1.0]
    assert [body["input"] for body in payloads] == [["alpha", "beta"], ["gamma"]]
    assert all(body["provider_options"] == {"task": "retrieval"} for body in payloads)
    assert all(
        body.get("dimensions") == (17 if explicit else None) for body in payloads
    )


@pytest.mark.parametrize(
    "vectors,message",
    [
        ([[1.0] * 8], "width.*8.*17"),
        ([[1.0] * 17, [1.0] * 17], "count.*2.*1"),
    ],
)
async def test_real_http_rejects_shape_mismatch(vectors, message):
    openai = pytest.importorskip("openai")

    def handle(request):
        return httpx.Response(
            200,
            json={
                "data": [
                    {"index": i, "embedding": vector}
                    for i, vector in enumerate(vectors)
                ]
            },
        )

    embedder = OpenAIEmbedder(
        model="fixture",
        api_key="fixture",
        dimensions=17,
        base_url="https://provider.test/v1",
    )
    async with openai.AsyncOpenAI(
        api_key="fixture",
        base_url="https://provider.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as provider:
        embedder._client.async_client = provider.embeddings
        with pytest.raises(EmbeddingError, match=message) as error:
            await embedder.embed(["alpha"])
    assert not isinstance(error.value, TransientEmbeddingError)


async def test_unsupported_dimension_request_is_not_silently_dropped():
    openai = pytest.importorskip("openai")
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            400,
            json={
                "error": {
                    "message": "dimensions unsupported",
                    "type": "invalid_request_error",
                }
            },
        )

    embedder = OpenAIEmbedder(
        model="fixed-provider",
        api_key="fixture",
        dimensions=17,
        base_url="https://provider.test/v1",
    )
    async with openai.AsyncOpenAI(
        api_key="fixture",
        base_url="https://provider.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as provider:
        embedder._client.async_client = provider.embeddings
        with pytest.raises(EmbeddingError, match="dimensions unsupported") as error:
            await embedder.embed(["alpha"])
    assert not isinstance(error.value, TransientEmbeddingError)
    assert len(payloads) == 1 and payloads[0]["dimensions"] == 17


def test_fixed_explicit_request_invalidates_legacy_cache_identity():
    from fastrag.ingestion._hashing import config_hash
    from fastrag.ingestion.steps.chunking.structure import EMBEDDING_TEXT_VERSION

    embedder = OpenAIEmbedder(model="fixture", api_key="fixture", dimensions=17)
    legacy = config_hash(
        "OpenAIEmbedder",
        {
            "provider": {"model": "fixture", "base_url": None, "dimensions": 17},
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
        },
    )
    assert embedder.config_hash != legacy
    # Inferred dimensions have unchanged request semantics and retain their identity.
    inferred = OpenAIEmbedder(model="text-embedding-ada-002", api_key="fixture")
    assert inferred.config_hash == config_hash(
        "OpenAIEmbedder",
        {
            "provider": {
                "model": "text-embedding-ada-002",
                "base_url": None,
                "dimensions": None,
            },
            "embedding_text_version": EMBEDDING_TEXT_VERSION,
        },
    )


@pytest.mark.parametrize("headers", [False, True])
async def test_corrupt_cached_vectors_rejected_before_yield(cache, headers):
    from fastrag.ingestion.cache import document_cache_id
    from fastrag.ingestion.policy import Policy
    from fastrag.ingestion.steps.base import ExecutionContext
    from fastrag.models import Document

    embedder = OpenAIEmbedder(
        model="fixture", api_key="fixture", dimensions=17, cache=cache
    )
    document = Document("dimension-parent", "source", "alpha")
    corrupted = Document("dimension-parent", "source", "alpha", embedding=[1.0] * 8)
    key = document_cache_id(embedder.name, document.id, embedder.config_hash)
    await cache.put_document_entries(key, [corrupted])

    async def stream():
        yield document

    context = ExecutionContext(
        "dimension-replay",
        Policy(on_error="raise"),
        cache=cache,
        cache_headers_only=headers,
    )
    with pytest.raises(EmbeddingError, match="width.*8.*17"):
        _ = [doc async for doc in embedder.evaluate_items(stream(), context)]
