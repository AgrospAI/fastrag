"""Real isolated PostgreSQL capability checks before replay and child ownership."""

from dataclasses import replace

import pytest
from helpers import document, source, stream

from fastrag.exceptions import ChunkingError, ParseError
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import ExecutionContext
from fastrag.ingestion.steps.chunking import ChunkerRouter, HeadingChunker
from fastrag.ingestion.steps.parsing import ParserRouter, DefaultParser


@pytest.mark.parametrize("routed", [False, True])
async def test_cached_chunker_cannot_bypass_changed_representation(cache, routed):
    child = HeadingChunker(cache=cache)
    step = ChunkerRouter(cache=cache).default(child) if routed else child
    context = ExecutionContext(
        "chunk-capability-cache", Policy(on_error="raise"), cache=cache
    )
    item = document("markdown", "# Heading\n\nbody")
    first = [d async for d in step.evaluate_items(stream(item), context)]
    second = [d async for d in step.evaluate_items(stream(item), context)]
    assert second == [replace(d, cached=True) for d in first]
    incompatible = replace(item, metadata={"content_format": "docling"})
    with pytest.raises(ChunkingError, match="docling"):
        [d async for d in step.evaluate_items(stream(incompatible), context)]


@pytest.mark.parametrize("routed", [False, True])
async def test_cached_parser_cannot_bypass_changed_media_type(cache, routed):
    child = DefaultParser(cache=cache)
    step = ParserRouter(cache=cache).default(child) if routed else child
    context = ExecutionContext(
        "parser-capability-cache", Policy(on_error="raise"), cache=cache
    )
    item = source()
    first = [d async for d in step.evaluate_items(stream(item), context)]
    second = [d async for d in step.evaluate_items(stream(item), context)]
    assert second == [replace(d, cached=True) for d in first]
    with pytest.raises(ParseError, match="application/pdf"):
        [
            d
            async for d in step.evaluate_items(
                stream(replace(item, content_type="application/pdf")), context
            )
        ]
