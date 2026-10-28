"""Entry encodings shared by every SQL cache backend.

Both :class:`~fastrag.ingestion.cache.postgres.PostgresCache` and
:class:`~fastrag.ingestion.cache.sqlite.SQLiteCache` store a document
snapshot's text and metadata as the same compressed payload, so a payload
written by one backend decodes identically in the other.
"""

from __future__ import annotations

import json
import zlib
from collections.abc import Sequence
from typing import Any

from fastrag.ingestion.models import Source
from fastrag.models import Document


def deduplicate[EntryT: (Source, Document)](items: Sequence[EntryT]) -> list[EntryT]:
    """Collapse ``items`` that share an ``id``, keeping the last of each.

    The snapshot tables key on ``(bucket, id)``, and an upsert cannot touch
    the same row twice — so a batch carrying one id twice fails the whole
    write. Producers are expected to deduplicate their own output (the
    fetchers do), but a cache write is the wrong place to discover that one
    of them didn't: two spellings of one URL that normalize to a single id
    (``…/audit`` and ``…/audit/``) describe the same row, and the last one
    wins exactly as it would if the batch were sent row by row.
    """
    return list({item.id: item for item in items}.values())


def document_payload(document: Document) -> dict[str, Any]:
    """Return the small immutable-stage payload, keeping vectors out of JSON."""
    return {
        "id": document.id,
        "source_id": document.source_id,
        "content": document.content,
        "step_config_hash": document.step_config_hash,
        "metadata": document.metadata,
    }


def pack_payload(payload: dict[str, Any]) -> bytes:
    """Compress stage text and metadata for storage and network transport."""
    return b"Z" + zlib.compress(json.dumps(payload, ensure_ascii=False).encode("utf-8"))


def unpack_payload(payload: bytes) -> dict[str, Any]:
    """Decode compressed payloads or plain JSON during an atomic upgrade."""
    raw = bytes(payload)
    return json.loads(zlib.decompress(raw[1:]) if raw.startswith(b"Z") else raw)
