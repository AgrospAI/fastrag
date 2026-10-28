"""Profile cached AgrospAI inspection without printing source content or credentials.

Run from the SDK directory with .venv/bin/python scripts/profile_agrospai_fetching.py.
Metrics are wall-clock measurements; async cProfile cumulative times overlap.
"""

import argparse
import asyncio
import cProfile
import json
import hashlib
from pathlib import Path
import runpy
import time
from typing import Any


def _measure():
    """Run the script with counters, wall timers and a CPU profile."""
    arguments = argparse.ArgumentParser(description=__doc__)
    arguments.add_argument("--label", default="cached")
    arguments.add_argument(
        "--output-dir", type=Path, default=Path("/tmp/fastrag-fetch-profile")
    )
    arguments.add_argument("--full-content", action="store_true")
    args = arguments.parse_args()
    if not args.label.replace("-", "").replace("_", "").isalnum():
        arguments.error(
            "--label must contain only letters, digits, hyphens or underscores"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    namespace = runpy.run_path(
        "agrospai-docs-chatbot/fetching.py", run_name="profile_import"
    )
    metrics: dict[str, Any] = {
        "sql_statements": 0,
        "import_seconds": time.perf_counter() - start,
        "cached_sources": 0,
        "fresh_sources": 0,
        "failed_items": 0,
        "failed_steps": 0,
        "content_bytes": 0,
        "output_bytes": 0,
        "render_calls": 0,
        "http_requests": 0,
        "http_seconds": 0,
        "cache_open_calls": 0,
        "cache_open_seconds": 0,
        "header_reads": 0,
        "header_read_seconds": 0,
        "bulk_reads": 0,
        "bulk_read_seconds": 0,
        "write_seconds": 0,
    }
    from sqlalchemy import event as sql_event
    from sqlalchemy.engine import Engine
    from fastrag.ingestion.events import DocumentFetched, ItemFailed, StepFailed
    from fastrag.ingestion.logger.source import SourceLogger
    from fastrag.ingestion.logger.source_tree import SourceTreeLogger
    from fastrag.ingestion import Pipeline
    from fastrag.ingestion.steps.fetching.http import _HttpTransport
    from fastrag.ingestion.cache import PostgresCache

    identities = set()
    output_hash = hashlib.sha256()

    def count_statement(*args):
        metrics["sql_statements"] += 1

    sql_event.listen(Engine, "before_cursor_execute", count_statement)

    class _Sink:
        def write(self, text):
            """Count encoded output without retaining source content."""
            output_hash.update(text.encode())
            metrics["output_bytes"] += len(text.encode())
            return len(text)

        def flush(self):
            """Accept flush requests from the inspection logger."""
            pass

        def isatty(self):
            """Use deterministic non-terminal rendering."""
            return False

    original_init = SourceLogger.__init__

    def init(self, *args, **kwargs):
        kwargs["stream"] = _Sink()
        original_init(self, *args, **kwargs)

    SourceLogger.__init__ = init
    tree_init = SourceTreeLogger.__init__

    def init_tree(self, *args, **kwargs):
        kwargs["stream"] = _Sink()
        tree_init(self, *args, **kwargs)

    SourceTreeLogger.__init__ = init_tree
    original_iter = Pipeline.__aiter__

    async def events(self):
        async for event in original_iter(self):
            if isinstance(event, DocumentFetched):
                metrics[
                    "cached_sources" if event.source.cached else "fresh_sources"
                ] += 1
                metrics["content_bytes"] += len(event.source.content)
                identities.add(
                    (event.source.id, event.source.uri, event.source.content_hash)
                )
            elif isinstance(event, ItemFailed):
                metrics["failed_items"] += 1
            elif isinstance(event, StepFailed):
                metrics["failed_steps"] += 1
            yield event

    Pipeline.__aiter__ = events
    original_records = SourceLogger._records

    def records(self, event):
        return original_records(self, event)

    SourceLogger._records = records
    original_write = SourceLogger._write

    def write(self, *args):
        start = time.perf_counter()
        try:
            return original_write(self, *args)
        finally:
            metrics["write_seconds"] += time.perf_counter() - start

    setattr(SourceLogger, "_write", write)

    def instrument(cls, method, count, duration=None):
        original = getattr(cls, method)

        async def wrapped(self, *args, **kwargs):
            metrics[count] += 1
            start = time.perf_counter()
            try:
                return await original(self, *args, **kwargs)
            finally:
                if duration:
                    metrics[duration] += time.perf_counter() - start

        setattr(cls, method, wrapped)

    instrument(
        PostgresCache, "get_source_entries_many", "bulk_reads", "bulk_read_seconds"
    )
    instrument(
        PostgresCache, "get_source_headers_many", "header_reads", "header_read_seconds"
    )
    instrument(PostgresCache, "__aenter__", "cache_open_calls", "cache_open_seconds")
    instrument(_HttpTransport, "request", "http_requests", "http_seconds")
    instrument(namespace["DecisionsRenderer"], "render", "render_calls")
    tree_render = SourceTreeLogger._render

    def render_tree(self, *args):
        start = time.perf_counter()
        try:
            return tree_render(self, *args)
        finally:
            metrics["write_seconds"] += time.perf_counter() - start

    setattr(SourceTreeLogger, "_render", render_tree)
    profile = cProfile.Profile()
    start = time.perf_counter()
    profile.enable()
    try:
        asyncio.run(
            namespace["main"](full_content=True)
            if args.full_content
            else namespace["main"]()
        )
    finally:
        profile.disable()
        metrics["main_seconds"] = time.perf_counter() - start
        metrics["output_sha256"] = output_hash.hexdigest()
        metrics["source_identity_sha256"] = hashlib.sha256(
            json.dumps(sorted(identities)).encode()
        ).hexdigest()
        (args.output_dir / f"{args.label}.json").write_text(
            json.dumps(metrics, indent=2)
        )
        profile.dump_stats(str(args.output_dir / f"{args.label}.prof"))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    _measure()
