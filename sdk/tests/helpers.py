"""Fakes and factories shared across test modules.

Tests import from here rather than from each other, so a test module can be
edited or deleted without breaking unrelated ones.
"""

import json
import random
from unittest.mock import AsyncMock, Mock

from docling_core.transforms.chunker.tokenizer.base import BaseTokenizer

from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.chunking import ChatBoundaryRefiner
from fastrag.ingestion.steps.chunking.structure import build_embedding_text
from fastrag.ingestion.steps.embedding import Embedder
from fastrag.models import Document
from fastrag.normalization import NormalizerRegistry
from fastrag.query.llm import ChatModel


def source(content=b"body", mime="text/plain", uri="file:///body.txt"):
    return Source(uri, uri, content, mime, "hash")


def document(fmt="text", content="body"):
    return Document("document", "source", content, metadata={"content_format": fmt})


async def stream(*items):
    for item in items:
        yield item


class FixedEmbedder(Embedder):
    def __init__(self, vectors, identity="fixed"):
        super().__init__(cache=False)
        self.vectors = vectors
        self.identity = identity
        self.calls = []
        self.position = 0

    @property
    def model_id(self):
        return "fixed"

    @property
    def dimension(self):
        return 2

    def config(self):
        return {"identity": self.identity}

    async def embed(self, texts):
        self.calls.append(list(texts))
        if isinstance(self.vectors, BaseException):
            raise self.vectors
        result = self.vectors[self.position : self.position + len(texts)]
        self.position += len(texts)
        return result


def doc(text="Apples grow. Pears grow. Stars shine. Planets orbit."):
    return Document("parent", "source", text, metadata={"title": "Guide"})


class EventPipeline:
    def __init__(self, events, error=None):
        self.events = events
        self.error = error

    async def __aiter__(self):
        for event in self.events:
            yield event
        if self.error:
            raise self.error


class WordTokenizer(BaseTokenizer):
    """A local, network-free stand-in for a real Hugging Face tokenizer.

    ``HybridChunker``/``LineBasedTokenChunker`` both default to pulling a
    real one on first use -- this covers wiring a caller-supplied chunker
    into ``DoclingChunker`` without ever downloading anything.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())

    def get_max_tokens(self) -> int:
        return 128

    def get_tokenizer(self) -> None:
        return None


def model(response='{"chunk_starts": [0, 2]}'):
    llm = Mock(spec=ChatModel)
    llm.model_id = "semantic-test"
    llm.invoke = AsyncMock(return_value=response)
    return llm


class Characters:
    identity = "test:unicode-characters:v1"

    def count(self, text):
        return len(text)

    def boundaries(self, text):
        return range(len(text) + 1)


def assert_coverage(document, pieces):
    normalized = NormalizerRegistry().normalize(document)
    actual = sorted(
        (i["start_offset"], i["end_offset"])
        for _, m in pieces
        for i in m["retained_intervals"]
    )
    expected = [(u.span.start_offset, u.span.end_offset) for u in normalized.units]
    # Piece intervals may subdivide coverage units; compare source occupancy exactly.
    assert [p for a, b in actual for p in range(a, b)] == [
        p for a, b in expected for p in range(a, b)
    ]
    for content, metadata in pieces:
        assert metadata["embedding_token_count"] == len(
            build_embedding_text(content, metadata)
        )
        assert metadata["embedding_token_count"] <= 120 or metadata["overflow"]
        json.dumps(metadata)


class FakeChat(ChatModel):
    def __init__(self, responses):
        self.responses = iter(responses)
        self.calls = 0
        self.active = self.peak = 0

    @property
    def model_id(self):
        return "fake"

    async def invoke(self, messages):
        self.calls += 1
        self.active += 1
        self.peak = max(self.peak, self.active)
        try:
            response = next(self.responses)
            if callable(response):
                return await response(messages)
            if isinstance(response, BaseException):
                raise response
            return response
        finally:
            self.active -= 1

    async def stream(self, messages):
        yield await self.invoke(messages)


def refiner(model, **options):
    return ChatBoundaryRefiner(
        llm=model,
        allow_external_transmission=True,
        identity="fake:endpoint:v1",
        **options,
    )


def generated_source(seed, fmt):
    rng = random.Random(seed)
    words = ["orchard", "界", "😀", "café", "water", "soil"]
    paragraphs = [
        " ".join(rng.choices(words, k=rng.randint(12, 45))) + ". Next sentence."
        for _ in range(5)
    ]
    if fmt == "markdown":
        return (
            "# Guide\n### Same\n"
            + "\n\n".join(paragraphs[:3])
            + "\n\n- parent\n  - nested\n\n```python\ndef harvest():\n    return 7\n```\n"
            + "\n| Crop | Rate |\n| --- | --- |\n| pear | 7 |\n"
            + "\n### Same\n"
            + "\n\n".join(paragraphs[3:])
        )
    if fmt == "rst":
        return (
            "Guide\n=====\n\nSame\n----\n\n"
            + "\n\n".join(paragraphs[:3])
            + "\n\n* parent\n\n  * nested\n\n.. code-block:: python\n\n"
            + "    def harvest():\n        return 7\n\n"
            + "+------+-------+\n| Crop | Rate |\n+======+=======+\n| pear | 7     |\n+------+-------+\n"
            + "\nSame\n----\n\n"
            + "\n\n".join(paragraphs[3:])
        )
    return (
        "<h1>Guide</h1><h3>Same</h3>"
        + "".join(f"<p>{p}</p>" for p in paragraphs[:3])
        + '<ol start="3"><li>parent<ul><li>nested</li></ul></li></ol>'
        + '<pre><code class="language-python">def harvest():\n    return 7\n</code></pre>'
        + "<table><tr><th>Crop</th><th>Rate</th></tr><tr><td>pear</td><td>7</td></tr></table>"
        + "<h3>Same</h3>"
        + "".join(f"<p>{p}</p>" for p in paragraphs[3:])
        + "<script>excluded poison</script>"
    )


class RecordingEmbedder(Embedder):
    def __init__(self, **kw):
        super().__init__(**kw)
        self.texts = []

    def config(self):
        return {"model": "fake"}

    model_id = "fake"
    dimension = 1

    async def embed(self, texts):
        self.texts.extend(texts)
        return [[float(len(t))] for t in texts]
