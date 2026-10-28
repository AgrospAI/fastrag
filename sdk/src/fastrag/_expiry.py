"""Expiry computation and the write-throttle that keeps it cheap.

Computes when a session/chat expires and decides whether to bump the stored
expiry on activity, throttled so an active conversation does not write on
every request. Private to ``fastrag`` — the one underscore-prefixed
top-level module that package-private code may share across packages;
nothing here is public API and it is not documented in the reference.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta


def new_expiry(ttl_days: int, now: datetime | None = None) -> datetime:
    """Return the expiry ``ttl_days`` from now (or from ``now``, when given)."""
    return (now or datetime.now(UTC)) + timedelta(days=ttl_days)


def should_bump_expiry(
    current_expires_at: datetime,
    candidate_expires_at: datetime,
    min_bump_days: int,
) -> bool:
    """Write-throttle for expiry bumps.

    Only bump when the new expiry is at least ``min_bump_days`` later than
    the stored one, so active conversations stay alive without a write on
    every request.
    """
    return (candidate_expires_at - current_expires_at) >= timedelta(days=min_bump_days)
