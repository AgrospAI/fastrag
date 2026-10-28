"""Verify extra model parameters reach JSON bodies through real clients."""

import json

import httpx
import pytest

from fastrag.ingestion.steps.embedding import OpenAIEmbedder
from fastrag.ingestion.steps.parsing import DocusaurusParser, OpenAIVisionParser
from fastrag.models import Document
from fastrag.query.llm import Message, OpenAIChatModel
from fastrag.retrieval.reranking import CohereReranker


async def test_vision_extra_body_and_cache_identity(monkeypatch):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": "Diagram"}}]}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    config = dict(model="Qwen3.8", api_key="secret", base_url="https://model.test/v1")
    extra = {"temperature": 0.5, "chat_template_kwargs": {"enable_thinking": False}}
    vision = OpenAIVisionParser(**config, extra_body=extra)
    extra["chat_template_kwargs"]["enable_thinking"] = True
    assert await vision.describe(b"png", "image/png") == "Diagram"
    assert payloads[0]["temperature"] == 0.5
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in payloads[0]
    assert payloads[0]["messages"][0]["content"][1]["type"] == "image_url"
    default = OpenAIVisionParser(**config)
    assert vision.config_hash != default.config_hash
    assert (
        DocusaurusParser(vision=vision).config_hash
        != DocusaurusParser(vision=default).config_hash
    )
    assert (
        default.config_hash == OpenAIVisionParser(**config, extra_body={}).config_hash
    )


@pytest.mark.parametrize("stream", [False, True])
async def test_chat_extra_body(stream):
    openai = pytest.importorskip("openai")
    pytest.importorskip("langchain_openai")
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        if stream:
            chunk = {
                "id": "test",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": "Qwen3.8",
                "choices": [{"index": 0, "delta": {"content": "Hello"}}],
            }
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n",
            )
        return httpx.Response(
            200,
            json={
                "id": "test",
                "object": "chat.completion",
                "created": 0,
                "model": "Qwen3.8",
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "Hello"},
                        "finish_reason": "stop",
                    }
                ],
            },
        )

    llm = OpenAIChatModel(
        model="Qwen3.8",
        api_key="secret",
        base_url="https://model.test/v1",
        temperature=1.0,
        extra_body={
            "temperature": 0.0,
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
    async with openai.AsyncOpenAI(
        api_key="secret",
        base_url="https://model.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as provider:
        llm._client.async_client = provider.chat.completions
        messages = [Message(role="user", content="Hi")]
        if stream:
            assert "".join([part async for part in llm.stream(messages)]) == "Hello"
        else:
            assert await llm.invoke(messages) == "Hello"
    assert payloads[0]["temperature"] == 0.0
    assert payloads[0]["chat_template_kwargs"] == {"enable_thinking": False}
    assert "extra_body" not in payloads[0]


async def test_embedding_extra_body_and_cache_identity():
    openai = pytest.importorskip("openai")
    pytest.importorskip("langchain_openai")
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={"data": [{"index": 0, "embedding": [0.1, 0.2]}]},
        )

    config = dict(
        model="embedding-test",
        api_key="secret",
        base_url="https://model.test/v1",
        dimensions=2,
    )
    extra = {"provider_options": {"task": "retrieval"}}
    embedder = OpenAIEmbedder(**config, extra_body=extra)
    extra["provider_options"]["task"] = "changed"
    async with openai.AsyncOpenAI(
        api_key="secret",
        base_url="https://model.test/v1",
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    ) as provider:
        embedder._client.async_client = provider.embeddings
        assert await embedder.embed(["Hello"]) == [[0.1, 0.2]]
    assert payloads[0]["provider_options"] == {"task": "retrieval"}
    assert payloads[0]["input"] == ["Hello"]
    assert payloads[0]["dimensions"] == 2
    assert "extra_body" not in payloads[0]
    assert embedder.config_hash != OpenAIEmbedder(**config).config_hash
    assert (
        OpenAIEmbedder(**config).config_hash
        == OpenAIEmbedder(**config, extra_body={}).config_hash
    )


async def test_reranker_extra_body(monkeypatch):
    payloads = []

    def handle(request):
        payloads.append(json.loads(request.content))
        return httpx.Response(
            200, json={"results": [{"index": 0, "relevance_score": 0.9}]}
        )

    client = httpx.AsyncClient
    monkeypatch.setattr(
        httpx,
        "AsyncClient",
        lambda **kw: client(transport=httpx.MockTransport(handle), **kw),
    )
    reranker = CohereReranker(
        model="rerank-test",
        api_key="secret",
        extra_body={"return_documents": False, "top_n": 2},
    )
    document = Document(id="test", source_id="source", content="Hello")
    try:
        result = await reranker.rerank("Hello", [document], top_k=1)
    finally:
        await reranker.close()
    assert result[0].id == document.id
    assert payloads[0]["return_documents"] is False
    assert payloads[0]["top_n"] == 2
    assert payloads[0]["documents"] == ["Hello"]
    assert "extra_body" not in payloads[0]
