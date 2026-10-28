"""Server-Sent Events framing and the keep-alive interleaver."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}

_KEEPALIVE = b": keep-alive\n\n"


def sse_frame(event: str, data: dict[str, Any]) -> bytes:
    """Frame one Server-Sent Event, with ``event`` naming its type.

    Anything else is delivered to the browser as an untyped ``message``.
    """
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


async def with_keepalive(
    source: AsyncIterator[bytes], interval: float = 15.0
) -> AsyncIterator[bytes]:
    """Interleave a ``: keep-alive`` comment every ``interval`` seconds.

    So idle proxies don't drop a slow generation.

    ``source`` is drained by one task for its whole life. An
    ``asyncio.timeout`` inside it cancels the task that entered it, so
    stepping it from a fresh task per frame would leave the timeout
    cancelling a task that had already finished. The queue never blocks
    that task, so a cancellation always lands inside ``source``.
    """
    frames: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def pump() -> None:
        try:
            async for frame in source:
                frames.put_nowait(frame)
        finally:
            frames.put_nowait(None)

    producer = asyncio.ensure_future(pump())

    try:
        while True:
            try:
                async with asyncio.timeout(interval):
                    frame = await frames.get()
            except TimeoutError:
                frame = _KEEPALIVE
            if frame is None:
                # Re-raise whatever ended ``source``, if anything did.
                await producer
                return
            yield frame
    finally:
        producer.cancel()
