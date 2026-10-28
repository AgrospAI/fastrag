"""Source selection applies before limits, including repeated evaluations."""

import pytest

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.filters import URLFilter
from fastrag.ingestion.steps.fetching import LocalFileFetcher, Sources


async def test_sources_filter_then_limit(tmp_path):
    for name in ("a.txt", "b.txt", "c.txt"):
        (tmp_path / name).write_text(name)
    sources = Sources(
        [LocalFileFetcher(patterns=[str(tmp_path / "*.txt")])],
        filter=URLFilter(include_patterns=[r"/[bc]\.txt$"]),
        max_sources=1,
    )
    for _ in range(2):
        selected = await sources.run()
        assert len(selected) == 1
        assert selected[0].uri.endswith(("/b.txt", "/c.txt"))
    sources.max_sources = None
    assert len(await sources.run()) == 2
    sources.filter = URLFilter(include_patterns=["never-matches"])
    assert await sources.run() == []


async def test_zero_limit_does_not_evaluate_children(tmp_path):
    sources = Sources(
        [LocalFileFetcher(patterns=[str(tmp_path / "missing.txt")])], max_sources=0
    )
    assert await sources.run() == []


def test_negative_limit_is_rejected():
    with pytest.raises(ConfigurationError, match="non-negative"):
        Sources([], max_sources=-1)
