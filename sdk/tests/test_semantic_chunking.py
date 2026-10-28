"""Semantic boundaries preserve source text and pipeline chunk identity."""

import json

import pytest
from helpers import model

from fastrag.exceptions import (
    ChatError,
    ChunkingError,
    ConfigurationError,
    TransientChatError,
)
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import SemanticChunker
from fastrag.models import Document


def document(text):
    return Document(
        id="parent",
        source_id="source",
        content=text,
        metadata={"source_uri": "https://example.com/docs", "title": "Guide"},
    )


async def test_semantic_boundaries_preserve_text_provenance_and_stable_ids():
    llm = model()
    chunker = SemanticChunker(llm=llm)
    source = document("# Apples\n\nApple explanation.\n\n# Pears\n\nPear explanation.")
    context = ExecutionContext(pipeline_id="test", policy=Policy())
    first = [chunk async for chunk in chunker.transform(source, context)]
    second = [chunk async for chunk in chunker.transform(source, context)]
    assert [c.content for c in first] == [
        "# Apples\n\nApple explanation.",
        "# Pears\n\nPear explanation.",
    ]
    assert [c.id for c in first] == [c.id for c in second]
    assert len({c.id for c in first}) == 2
    assert [c.metadata["chunk_index"] for c in first] == [0, 1]
    assert all(c.metadata["parent_id"] == source.id for c in first)
    assert all(c.metadata["title"] == "Guide" for c in first)
    assert all(c.source_id == source.source_id for c in first)
    assert all(c.step_config_hash == chunker.config_hash for c in first)


async def test_large_document_uses_bounded_windows_without_losing_text():
    llm = model('{"chunk_starts": [0]}')
    chunker = SemanticChunker(llm=llm, chunk_size=20, max_input_chars=60)
    text = "A" * 103 + "\n\n" + "B" * 107 + "\n"
    pieces = [piece async for piece in chunker.asplit(document(text))]
    assert "".join(pieces) == text
    assert llm.invoke.await_count > 1
    for call in llm.invoke.call_args_list:
        blocks = json.loads(call.args[0][1].content)["blocks"]
        assert sum(map(len, blocks)) <= 60


@pytest.mark.parametrize(
    "response",
    [
        "not JSON",
        "{}",
        "null",
        '{"chunk_starts": []}',
        '{"chunk_starts": [1]}',
        '{"chunk_starts": [0, 0]}',
        '{"chunk_starts": [0, 2, 1]}',
        '{"chunk_starts": [0, 100]}',
        '{"chunk_starts": [false]}',
        '{"chunk_starts": [0, "1"]}',
    ],
)
async def test_invalid_model_boundaries_fail_before_producing_chunks(response):
    chunker = SemanticChunker(llm=model(response))
    context = ExecutionContext(pipeline_id="test", policy=Policy())
    chunks = []
    with pytest.raises(ChunkingError):
        async for chunk in chunker.transform(document("First.\n\nSecond."), context):
            chunks.append(chunk)
    assert chunks == []


@pytest.mark.parametrize("text", ["", " \n\t", "Single paragraph."])
async def test_empty_or_single_block_needs_no_model_call(text):
    llm = model()
    chunks = [piece async for piece in SemanticChunker(llm=llm).asplit(document(text))]
    assert chunks == ([text] if text.strip() else [])
    llm.invoke.assert_not_awaited()


@pytest.mark.parametrize("chunk_size,max_input_chars", [(0, 100), (100, 50)])
def test_invalid_limits(chunk_size, max_input_chars):
    with pytest.raises(ConfigurationError):
        SemanticChunker(
            llm=model(), chunk_size=chunk_size, max_input_chars=max_input_chars
        )


async def test_semantic_chunker_emits_standard_pipeline_events():
    from fastrag.ingestion import Pipeline
    from fastrag.ingestion.events import DocumentChunked
    from fastrag.ingestion.steps.base import PipelineStep

    source = document("First topic.\n\nSecond topic.")

    class Documents(PipelineStep):
        def config(self):
            return {}

        async def evaluate(self, context):
            yield source

    chunker = SemanticChunker(llm=model('{"chunk_starts": [0, 1]}'))
    async with Pipeline(
        chunker(Documents()), policy=Policy(on_error="raise")
    ) as pipeline:
        events = [event async for event in pipeline]
    chunked = [event for event in events if isinstance(event, DocumentChunked)]
    assert len(chunked) == 1
    assert chunked[0].document == source
    assert [c.content for c in chunked[0].chunks] == ["First topic.", "Second topic."]
    assert chunked[0].event_name == "Chunking"


async def test_transient_fallback_preserves_all_text_and_stops_calling_unavailable_model():
    llm = model()
    llm.invoke.side_effect = TransientChatError("unavailable")
    chunker = SemanticChunker(
        llm=llm, chunk_size=20, max_input_chars=60, fallback_on_transient_error=True
    )
    text = "A" * 103 + "\n\n" + "B" * 107 + "\n"
    pieces = [piece async for piece in chunker.asplit(document(text))]
    assert "".join(pieces) == text
    assert all(len(piece) <= 20 for piece in pieces)
    llm.invoke.assert_awaited_once()
    other = [piece async for piece in chunker.asplit(document("Another.\n\nDocument."))]
    assert "".join(other) == "Another.\n\nDocument."
    llm.invoke.assert_awaited_once()
    assert (
        chunker.config_hash
        != SemanticChunker(llm=llm, chunk_size=20, max_input_chars=60).config_hash
    )


@pytest.mark.parametrize(
    "error", [ChatError("unauthorized"), TransientChatError("timeout")]
)
async def test_chunking_failures_without_applicable_fallback_still_raise(error):
    llm = model()
    llm.invoke.side_effect = error
    chunker = SemanticChunker(
        llm=llm, fallback_on_transient_error=not isinstance(error, TransientChatError)
    )
    with pytest.raises(type(error)):
        _ = [piece async for piece in chunker.asplit(document("First.\n\nSecond."))]


async def test_fallback_does_not_hide_invalid_model_boundaries():
    chunker = SemanticChunker(llm=model("not JSON"), fallback_on_transient_error=True)
    with pytest.raises(ChunkingError):
        _ = [piece async for piece in chunker.asplit(document("First.\n\nSecond."))]
