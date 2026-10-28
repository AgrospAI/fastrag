"""Deterministic hashing: content addresses and stable config fingerprints.

Every hash here must be stable across processes, machines, and interpreter
runs, so nothing in this module may depend on :func:`hash`, on object
identity, or on dict insertion order.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

CONTENT_HASH_LENGTH = 32
CONFIG_HASH_LENGTH = 16


def content_hash(data: bytes | str) -> str:
    """Return the truncated sha256 hex digest of ``data``.

    Args:
        data: Bytes, or text encoded as UTF-8 before hashing.

    Returns:
        A stable, truncated hex digest usable as a content address.
    """
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()[:CONTENT_HASH_LENGTH]


def config_hash(name: str, config: Mapping[str, Any]) -> str:
    """Return a stable, short hex digest of ``name`` and ``config``.

    ``config`` is serialized with sorted keys, so dict insertion order never
    changes the digest — only the ``(name, key, value)`` triples do.

    Args:
        name: Identifies what is being fingerprinted, e.g. a step's class
            name.
        config: The configuration to fingerprint. Values must be JSON-safe or
            handled by the fallback stringifier.

    Returns:
        A stable, truncated hex digest.
    """
    payload = {"name": name, "config": dict(config)}
    encoded = json.dumps(payload, sort_keys=True, default=_stringify).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:CONFIG_HASH_LENGTH]


def _stringify(value: Any) -> str:
    """Render a value JSON cannot encode natively. Deliberately conservative."""
    if isinstance(value, set | frozenset):
        return repr(sorted(map(repr, value)))
    return repr(value)
