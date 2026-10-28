"""Cache identity helpers, shared by every step that caches.

A cache id has the ``"{step}:{hash}"`` shape shown in
``docs/specification/API.md``'s Example 1 (``"webcrawler:1a2b3c4"``) and
Example 2 (``"charactertextsplitter:e21qcasg"``): the step's (lowercased)
name, then a short digest of what makes this particular cache entry unique.

Both helpers build on ``content_hash`` rather
than adding a second hashing scheme — the digest is just ``content_hash`` of
the two identifying strings joined by a separator that cannot occur in either
half's raw form (a colon would: URIs and document ids may contain one).
"""

from __future__ import annotations

from fastrag.ingestion._hashing import content_hash

_SEPARATOR = "\n"


def source_cache_id(step_name: str, target: str, config_hash: str) -> str:
    """Return the cache id for one fetch target under one step configuration.

    Args:
        step_name: The fetching step's class name, e.g. ``"WebCrawler"``.
        target: What was fetched, e.g. a start URL — cached independently per
            target to increase the cache hit rate.
        config_hash: The step's ``config_hash``.
    """
    digest = content_hash(_SEPARATOR.join((target, config_hash)))
    return f"{step_name.lower()}:{digest}"


def document_cache_id(step_name: str, document_id: str, config_hash: str) -> str:
    """Return the cache id for one document under one step configuration.

    Args:
        step_name: The transforming step's class name, e.g.
            ``"CharacterTextSplitter"``.
        document_id: The upstream :class:`~fastrag.models.Document`'s
            id — cached independently per document to increase the cache hit
            rate.
        config_hash: The step's ``config_hash``.
    """
    digest = content_hash(_SEPARATOR.join((document_id, config_hash)))
    return f"{step_name.lower()}:{digest}"
