"""Bounded local structural profile; JSON timings are evidence, never test gates."""

import argparse
import cProfile
import json
import signal
import time
import tracemalloc
from fastrag.models import Document
from fastrag.normalization import NormalizerRegistry
from fastrag.ingestion.steps.chunking import (
    StructureAwareChunker,
    validate_structure_chunks,
)


class Characters:
    """Count complete strings while recording tokenizer work."""

    identity = "profile:characters:v1"

    def __init__(self):
        """Initialize the operation counter."""
        self.calls = 0

    def count(self, text):
        """Count Unicode characters in the final serialized input."""
        self.calls += 1
        return len(text)

    def boundaries(self, text):
        """Permit every Unicode character boundary."""
        return range(len(text) + 1)


def fixture(fmt, blocks):
    """Build equivalent node-rich sources without external files."""
    prose = "Café has useful documentation and examples."
    if fmt == "html":
        return "<h1>Guide</h1>" + "".join(f"<p>{prose}</p>" for _ in range(blocks))
    if fmt == "rst":
        return "Guide\n=====\n\n" + (prose + "\n\n") * blocks
    return "# Guide\n\n" + (prose + "\n\n") * blocks


def main():
    """Measure bounded normalization, packing and validation phases."""
    p = argparse.ArgumentParser()
    p.add_argument("--format", choices=["markdown", "html", "rst"], default="markdown")
    p.add_argument("--blocks", type=int, default=200)
    p.add_argument("--profile")
    p.add_argument("--prose-chars", type=int, default=0)
    args = p.parse_args()
    if not 1 <= args.blocks <= 2000:
        p.error("blocks must be between 1 and 2000")
    if not 0 <= args.prose_chars <= 6000:
        p.error("prose-chars must be between 0 and 6000")
    signal.alarm(60)
    tokenizer = Characters()
    engine = StructureAwareChunker(tokenizer=tokenizer, cache=False)
    doc = Document(
        "profile",
        "local",
        ("x" * args.prose_chars)
        if args.prose_chars
        else fixture(args.format, args.blocks),
        metadata={"content_format": args.format},
    )
    result = {
        "format": args.format,
        "blocks": args.blocks,
        "chars": len(doc.content),
        "prose_chars": args.prose_chars,
    }
    profiler = cProfile.Profile() if args.profile else None
    if profiler:
        profiler.enable()

    def measure(name, action):
        calls = tokenizer.calls
        tracemalloc.start()
        start = time.perf_counter()
        value = action()
        seconds = time.perf_counter() - start
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        result[name] = {
            "seconds": round(seconds, 6),
            "peak_mib": round(peak / 2**20, 3),
            "count_calls": tokenizer.calls - calls,
        }
        return value

    normalized = measure("normalize", lambda: NormalizerRegistry().normalize(doc))
    pieces = measure(
        "split_normalized", lambda: list(engine.split_normalized(normalized))
    )
    measure(
        "validate",
        lambda: validate_structure_chunks(normalized, pieces, tokenizer, 600),
    )
    result.update(
        nodes=len(list(normalized.walk())),
        units=len(normalized.units),
        chunks=len(pieces),
    )
    if profiler:
        profiler.disable()
        profiler.dump_stats(args.profile)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
