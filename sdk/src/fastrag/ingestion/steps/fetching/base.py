"""The ``Fetcher`` interface: ``docs/specification/subclasses.md``, verbatim."""

from __future__ import annotations

from abc import ABC
from typing import Any, Self

from fastrag.exceptions import ConfigurationError
from fastrag.ingestion.filters import Filter
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.base import SourceStep


class Fetcher(SourceStep, ABC):
    """Base class for leaf-level data fetchers that extract raw content from a single protocol/medium.

    Every event a fetcher emits is labelled ``"Fetching"`` — restated here
    rather than only inherited from :class:`~fastrag.ingestion.steps.base.SourceStep`
    so the family's own module states the stage it belongs to.

    Fetchers that discover URLs (crawlers) can delegate specific ones to other
    fetchers with :meth:`route`, e.g. YouTube links to a ``YouTubeFetcher``.
    """

    event_name = "Fetching"

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize the fetcher with no routes registered."""
        super().__init__(*args, **kwargs)
        self.routes: list[tuple[Filter[Source], Fetcher]] = []

    def route(self, filter_rule: Filter[Source], fetcher: Fetcher) -> Self:
        """Delegate the URLs matching ``filter_rule`` to ``fetcher``.

        The filter sees a content-less ``Source`` carrying only the URL, so
        URL-based filters such as :class:`~fastrag.ingestion.filters.URLFilter`
        apply. Routes are tried in registration order, and the first match
        wins.

        Args:
            filter_rule: Predicate over the URL about to be fetched.
            fetcher: Fetcher implementing :meth:`fetch`, used on match.

        Returns:
            Self, for fluent chaining.
        """
        self.routes.append((filter_rule, fetcher))
        return self

    def route_for(self, uri: str) -> Fetcher | None:
        """The fetcher routed to ``uri``, or ``None`` to fetch it normally."""
        probe = Source(id=uri, uri=uri, content=b"", content_type="", content_hash="")
        for filter_rule, fetcher in self.routes:
            if filter_rule.matches(probe):
                return fetcher
        return None

    def canonical_uri(self, uri: str) -> str:
        """Identity under which ``uri`` is fetched, to deduplicate equivalent spellings."""
        return uri

    async def fetch(self, uri: str) -> list[Source]:
        """Fetch one URL in-process, uncached. Needed to be a :meth:`route` target.

        Raises:
            fastrag.exceptions.ConfigurationError: This fetcher cannot fetch
                an arbitrary URL.
        """
        raise ConfigurationError(
            f"{type(self).__name__} cannot be used as a route target"
        )

    def routes_config(self) -> list[dict[str, Any]]:
        """Routed fetchers' fingerprints, for inclusion in a crawler's ``config()``."""
        return [
            {"fetcher": type(fetcher).__name__, "config": dict(fetcher.config())}
            for _, fetcher in self.routes
        ]
