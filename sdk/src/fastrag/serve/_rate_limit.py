"""In-process, per-worker rate limiting.

A small sliding-window-log limiter keyed by an arbitrary string (this module
never decides what the key is — callers compose it from a bucket name and a
client IP). No Redis, no cross-process coordination: a deployment running
several workers gets roughly ``limit * workers`` capacity, not a globally
enforced cap. That is a deliberate trade (see the serving guide's
rate-limiting section), not an oversight — a distributed limiter is out of
scope for what this SDK is for.

``parse_rate`` is also what :class:`~fastrag.serve.config.ServeConfig`
validates its three rate-limit strings with in ``__post_init__``, so a
malformed ``"10/fortnight"`` fails at construction rather than at the first
request that hits it.
"""

from __future__ import annotations

import re
import time
from collections import defaultdict, deque

from fastrag.exceptions import ConfigurationError

_RATE_RE = re.compile(r"^(?P<count>\d+)/(?P<unit>second|minute|hour|day)$")

_UNIT_SECONDS: dict[str, float] = {
    "second": 1.0,
    "minute": 60.0,
    "hour": 3600.0,
    "day": 86400.0,
}


def parse_rate(spec: str) -> tuple[int, float]:
    """Parse a ``"10/minute"``-style limit into ``(count, window_seconds)``.

    Raises:
        ConfigurationError: ``spec`` is not ``<positive int>/<second|minute|hour|day>``.
    """
    match = _RATE_RE.match(spec.strip())
    if match is None:
        raise ConfigurationError(
            f"Invalid rate limit {spec!r}; expected '<count>/<second|minute|hour|day>'"
        )
    count = int(match.group("count"))
    if count < 1:
        raise ConfigurationError(
            f"Invalid rate limit {spec!r}; count must be at least 1"
        )
    return count, _UNIT_SECONDS[match.group("unit")]


class RateLimiter:
    """A sliding-window-log limiter, one independent window per key.

    One instance lives on ``app.state.rate_limiter`` for the app's lifetime.
    :meth:`hit` is synchronous end to end (no ``await`` inside it), so the
    check-then-record it performs is atomic under asyncio's cooperative
    scheduling without needing a lock — nothing can interleave between the
    prune, the length check, and the append.
    """

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = defaultdict(deque)

    def hit(self, key: str, spec: str) -> bool:
        """Record one attempt at ``key``; return whether it is within ``spec``.

        Args:
            key: Identifies the bucket being limited — typically
                ``f"{bucket_name}:{client_ip}"``.
            spec: A ``"10/minute"``-style limit, parsed by :func:`parse_rate`.

        Returns:
            ``True`` and records the attempt if ``key`` is still under its
            limit; ``False`` (the attempt is not recorded) if it is not.
        """
        count, window = parse_rate(spec)
        now = time.monotonic()
        bucket = self._hits[key]
        while bucket and bucket[0] <= now - window:
            bucket.popleft()
        if len(bucket) >= count:
            return False
        bucket.append(now)
        return True
