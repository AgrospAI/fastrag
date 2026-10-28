"""Opt-in chat refinement uses safe offsets and preserves structural invariants."""

import asyncio
import json

import pytest
from helpers import Characters, FakeChat, assert_coverage, refiner

from fastrag.ingestion.steps.chunking import ChatBoundaryRefiner, StructureAwareChunker
from fastrag.models import Document


def doc(text="First sentence. Second sentence. Third sentence."):
    return Document(
        id="doc",
        source_id="source",
        content=text,
        metadata={"content_format": "markdown"},
    )


def engine(strategy=None, **options):
    return StructureAwareChunker(
        tokenizer=Characters(),
        target_tokens=80,
        max_tokens=120,
        min_preferred_tokens=20,
        semantic_refiner=strategy,
        **options,
    )


async def collect(chunker, document):
    return [pair async for pair in chunker.asplit(document)]


def test_consent_and_async_required():
    model = FakeChat([])
    with pytest.raises(ValueError):
        ChatBoundaryRefiner(llm=model, identity="fake")
    with pytest.raises(ValueError):
        engine(semantic_mode="llm")
    with pytest.raises(NotImplementedError):
        list(engine(refiner(model), semantic_mode="llm").split(doc()))
    assert model.calls == 0


@pytest.mark.asyncio
async def test_defaults_no_calls_and_valid_boundaries_preserve_source():
    model = FakeChat(['{"boundaries": [16]}'])
    strategy = refiner(model)
    document = doc()
    for mode in ("disabled", "heuristic"):
        await collect(engine(strategy, semantic_mode=mode), document)
    assert model.calls == 0
    reports = []
    pieces = await collect(
        engine(strategy, semantic_mode="llm", diagnostics=reports.append), document
    )
    assert len(pieces) == 2
    assert pieces[0][0] == "First sentence."
    assert_coverage(document, pieces)
    assert reports[-1].model_calls == 1
    assert reports[-1].model_fallbacks == ()
    assert all("model_calls" not in metadata for _, metadata in pieces)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response",
    [
        "garbage",
        '{"boundaries": [1], "boundaries": [16]}',
        '{"boundaries": [true]}',
        '{"boundaries": [1]}',
        '{"boundaries": [16,16]}',
        '{"boundaries": [-1]}',
        '{"boundaries": [999]}',
        '{"boundaries": [16], "text":"rewritten"}',
        RuntimeError("provider"),
    ],
)
async def test_errors_fallback_is_exact_and_isolated(response):
    model = FakeChat([response, '{"boundaries": [16]}'])
    reports = []
    chunker = engine(
        refiner(model, retries=0), semantic_mode="llm", diagnostics=reports.append
    )
    document = doc()
    baseline = list(engine().split(document))
    assert await collect(chunker, document) == baseline
    assert reports[-1].model_fallbacks
    assert len(await collect(chunker, document)) == 2
    assert model.calls == 2


@pytest.mark.asyncio
async def test_retry_timeout_cache_and_concurrency():
    async def slow(messages):
        await asyncio.sleep(0.05)
        return '{"boundaries": [16]}'

    model = FakeChat([slow, '{"boundaries": [16]}', '{"boundaries": [16]}'])
    strategy = refiner(
        model, timeout=0.01, retries=1, backoff=0, concurrency=1, cache_size=1
    )
    request = dict(node_id="node", text=doc().content, allowed=(16, 33))
    result = await strategy.refine(**request)
    assert result.boundaries == (16,)
    assert result.calls == 2
    assert (await strategy.refine(**request)).cache_hit
    await strategy.refine(**{**request, "node_id": "other"})
    assert strategy.cache_entries == 1
    assert model.peak == 1


@pytest.mark.asyncio
async def test_bounded_concurrency_and_cancellation():
    gate = asyncio.Event()

    async def blocked(messages):
        await gate.wait()
        return '{"boundaries": []}'

    model = FakeChat([blocked] * 4)
    strategy = refiner(model, concurrency=2)
    tasks = [
        asyncio.create_task(strategy.refine(node_id=str(i), text="A. B.", allowed=(3,)))
        for i in range(4)
    ]
    await asyncio.sleep(0.01)
    assert model.peak == 2
    for task in tasks:
        task.cancel()
    for task in tasks:
        with pytest.raises(asyncio.CancelledError):
            await task
    assert model.active == 0
    gate.set()
    assert (
        await strategy.refine(node_id="next", text="A. B.", allowed=(3,))
    ).boundaries == ()


@pytest.mark.asyncio
async def test_unsafe_structures_and_input_limit_never_transmitted():
    model = FakeChat([])
    chunker = engine(refiner(model, max_input_chars=30), semantic_mode="llm")
    for text in (
        "```python\nA. B.\n```",
        "- A. B.",
        "[A. B.](https://example.org)",
        "Long sentence. " * 10,
    ):
        await collect(chunker, doc(text))
    assert model.calls == 0


def test_nonsecret_config_identity():
    model = FakeChat([])
    model.api_key = "secret"
    model.base_url = "https://user:password@example.org"
    config = engine(refiner(model), semantic_mode="llm").config()
    assert "secret" not in json.dumps(config)
    assert "password" not in json.dumps(config)


@pytest.mark.asyncio
async def test_refined_oversized_fragments_still_obey_embedding_budget():
    # The proposed final fragment exceeds max_tokens and must be structurally
    # packed; the model cannot relax the embedding ceiling.
    document = doc("First sentence. " + "Longer sentence with details. " * 12)
    model = FakeChat(['{"boundaries": [16]}'])
    pieces = await collect(engine(refiner(model), semantic_mode="llm"), document)
    assert len(pieces) > 2
    assert pieces[0][0] == "First sentence."
    assert_coverage(document, pieces)


@pytest.mark.asyncio
async def test_tokenizer_disallows_semantic_offset():
    class Pairs(Characters):
        identity = "pairs:v1"

        def boundaries(self, text):
            return [0, *range(1, len(text), 2), len(text)]

    model = FakeChat(['{"boundaries": [16]}'])
    strategy = refiner(model)
    chunker = StructureAwareChunker(
        tokenizer=Pairs(),
        semantic_refiner=strategy,
        semantic_mode="llm",
        target_tokens=80,
        max_tokens=120,
        min_preferred_tokens=20,
    )
    reports = []
    chunker.diagnostics = reports.append
    assert len(await collect(chunker, doc())) == 1
    assert reports[-1].model_fallbacks == ("invalid_response",)


@pytest.mark.asyncio
async def test_retry_ceiling_success_cache_eviction_and_failure_not_cached():
    model = FakeChat(
        [
            RuntimeError("one"),
            RuntimeError("two"),
            '{"boundaries": []}',
            '{"boundaries": []}',
            '{"boundaries": []}',
        ]
    )
    strategy = refiner(model, retries=1, backoff=0, cache_size=1)
    request = dict(node_id="one", text="A. B.", allowed=(3,))
    result = await strategy.refine(**request)
    assert result.calls == 2 and result.fallback == "model_error"
    assert strategy.cache_entries == 0
    assert (await strategy.refine(**request)).fallback is None
    assert (await strategy.refine(**request)).cache_hit
    await strategy.refine(**{**request, "node_id": "two"})
    assert strategy.cache_entries == 1
    assert not (await strategy.refine(**request)).cache_hit
    assert model.calls == 5


@pytest.mark.asyncio
async def test_cancellation_in_retry_backoff_propagates():
    model = FakeChat([RuntimeError("provider")])
    strategy = refiner(model, retries=2, backoff=10)
    task = asyncio.create_task(
        strategy.refine(node_id="one", text="A. B.", allowed=(3,))
    )
    await asyncio.sleep(0.01)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert model.calls == 1


@pytest.mark.asyncio
async def test_html_and_rst_protected_prose():
    model = FakeChat([])
    chunker = engine(refiner(model), semantic_mode="llm")
    for fmt, text in [
        ("html", "<p>A. B.</p>"),
        ("rst", "* A. B.\n"),
        ("rst", ".. note::\n\n   A. B.\n"),
    ]:
        document = Document(
            id="doc", source_id="source", content=text, metadata={"content_format": fmt}
        )
        await collect(chunker, document)
    assert model.calls == 0


@pytest.mark.parametrize(
    "options",
    [
        {"timeout": float("nan")},
        {"timeout": 0},
        {"retries": True},
        {"retries": 11},
        {"concurrency": 0},
        {"cache_size": -1},
        {"backoff": float("inf")},
        {"max_input_chars": 0},
    ],
)
def test_invalid_refinement_config(options):
    with pytest.raises(ValueError):
        refiner(FakeChat([]), **options)


@pytest.mark.asyncio
async def test_concurrent_documents_isolate_boundaries_and_diagnostics():
    async def respond(messages):
        request = json.loads(messages[-1].content)
        await asyncio.sleep(0.005)
        return json.dumps(
            {
                "boundaries": request["allowed"][:1]
                if request["text"].startswith("First")
                else []
            }
        )

    model = FakeChat([respond, respond])
    chunker = engine(refiner(model), semantic_mode="llm")
    first, second = await asyncio.gather(
        collect(chunker, doc()),
        collect(chunker, doc("Other sentence. Another sentence.")),
    )
    assert len(first) == 2 and len(second) == 1
    assert second[0][0] == "Other sentence. Another sentence."


@pytest.mark.asyncio
async def test_failure_isolated_between_paragraphs_in_same_document():
    model = FakeChat(["malformed", '{"boundaries": [16]}'])
    document = doc(
        "Other sentence. Another sentence.\n\nFirst sentence. Second sentence."
    )
    reports = []
    pieces = await collect(
        engine(refiner(model), semantic_mode="llm", diagnostics=reports.append),
        document,
    )
    assert model.calls == 2
    assert reports[-1].model_fallbacks == ("invalid_response",)
    assert reports[-1].model_calls == 2
    assert pieces[-1][0] == "Second sentence."
    assert_coverage(document, pieces)
