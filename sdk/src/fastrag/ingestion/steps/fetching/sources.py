r"""``Sources``: aggregate several ``SourceStep``\ s into one stream.

``docs/specification/subclasses.md``, resolved against the base class task 003
actually built (which exposes ``config()``, not the pseudocode's
``compute_config_hash()`` — see ``.agent/results/003-step-base.md``'s own
Deviations for why the pseudocode is stale here too).
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any, Literal

from fastrag.ingestion.cache import CacheStore
from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.filters import Filter
from fastrag.ingestion.models import Source
from fastrag.ingestion.policy import Policy
from fastrag.ingestion.steps.base import (
    ExecutionContext,
    SourceStep,
    run_concurrently,
    run_source_step,
)


class Sources(SourceStep):
    r"""Aggregator step combining multiple ``Fetcher``\ s or ``SourceStep``\ s into one stream.

    .. code-block:: python

        sources = Sources(
            sources=[LocalFileFetcher(...), SitemapCrawler(...)],
            cache=cache,  # default cache for all sources
            policy=policy,  # default policy for all sources
        )
    """

    def __init__(
        self,
        sources: Sequence[SourceStep],
        cache: CacheStore | Literal[False] | None = None,
        policy: Policy | None = None,
        *,
        filter: Filter[Source] | None = None,
        max_sources: int | None = None,
    ) -> None:
        r"""Initialize the ``Sources`` aggregator.

        Args:
            sources: Child ``SourceStep``\ s to evaluate concurrently, bounded
                by ``policy.concurrency``.
            cache: Default cache for children that declare none of their own
                (``child.cache is None``). A child with its own explicit
                ``cache`` (including ``cache=False``) keeps it.
            policy: Default execution policy for children that declare none
                of their own.
            filter: Predicate applied to fetched sources before returning them.
            max_sources: Maximum number of matching sources to return, across
                all children, in completion order. Zero returns nothing.
                These options select results, not download targets; use each
                crawler's include/exclude/max_urls to reduce network requests.
        """
        # `self.cache` is deliberately hardcoded to `False`: `Sources` itself
        # is never cached as one aggregate bucket (see module docstring on
        # `_resolve_child_cache` below for why that would be actively wrong),
        # only its children are. The caller's `cache` is instead kept as
        # `_children_cache`, the default *children* inherit.
        if max_sources is not None and max_sources < 0:
            raise ConfigurationError("Sources.max_sources must be non-negative")
        super().__init__(cache=False, policy=policy)
        self.sources = sources
        self._children_cache = cache
        self.filter = filter
        self.max_sources = max_sources

    def config(self) -> dict[str, Any]:
        """No identity-affecting configuration of its own.

        `Sources` is never cached as a single bucket (see ``__init__``) so
        this only exists to satisfy `PipelineStep.config`'s abstract
        contract; nothing consults it.
        """
        return {}

    async def aclose(self) -> None:
        """Release resources held by children that expose an async close hook."""
        for child in self.sources:
            close = getattr(child, "aclose", None)
            if close is not None:
                await close()

    async def evaluate(self, context: ExecutionContext) -> AsyncIterator[Source]:
        """Merge every child's stream, each cached under its own resolution.

        Each child is driven through :func:`~fastrag.ingestion.steps.base.run_source_step` — the same
        caching + lifecycle wrap every ``SourceStep`` gets — under a context
        whose ``cache``/``policy`` are resolved once here: a child with its
        own explicit ``cache``/``policy`` keeps them (resolution happens
        again, inside the child's own wrap, against whatever this context
        carries); a child with neither inherits this aggregator's.
        """
        if self.max_sources == 0:
            return
        scoped = replace(
            context,
            cache=_resolve_child_cache(self._children_cache, context.cache),
            policy=self.effective_policy(context),
            cache_headers_only=context.cache_headers_only
            and (self.filter is None or self.filter.header_safe),
        )
        effective_policy = self.effective_policy(context)

        async def run_one(child: SourceStep) -> list[Source]:
            return [
                item
                async for item in run_source_step(child, scoped)
                if self.filter is None or self.filter.matches(item)
            ]

        async def _children() -> AsyncIterator[SourceStep]:
            for child in self.sources:
                yield child

        returned = 0
        async for produced in run_concurrently(
            _children(),
            run_one,
            context=context,
            policy=effective_policy,
            item_id=lambda child: type(child).__name__,
            step=self.name,
            event_name=self.event_name,
        ):
            for source in produced:
                if self.max_sources is None or returned < self.max_sources:
                    returned += 1
                    yield source


def _resolve_child_cache(
    own: CacheStore | Literal[False] | None, outer: CacheStore | None
) -> CacheStore | None:
    """The same three-state resolution ``PipelineStep``'s own cache uses.

    ``own`` is ``Sources``'s ``_children_cache`` (its constructor's ``cache``
    argument); ``outer`` is whatever cache the aggregator itself was run
    under. ``False`` disables the default entirely (children default to no
    cache unless they set their own); ``None`` passes the outer context's
    cache through unchanged; an explicit ``CacheStore`` is used directly.
    """
    if own is False:
        return None
    if own is None:
        return outer
    return own
