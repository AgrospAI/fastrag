"""Run a coroutine to completion even under repeated cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Coroutine


async def shield_until_done[T](coro: Coroutine[None, None, T]) -> T:
    """Run ``coro`` to completion even if the awaiting task keeps being cancelled.

    Starlette/anyio cancel a streaming response's task group on client
    disconnect, and — unlike plain asyncio — anyio re-raises ``CancelledError``
    at every subsequent await point in that scope, not just once. A plain
    ``asyncio.shield()`` isn't enough on its own: the *await* of the shield
    still raises ``CancelledError`` each time, even though the shielded task
    keeps running in the background. This loops on that shield until the
    background task has actually finished, which is what makes cleanup
    that must not be abandoned mid-write — persisting the assistant
    message, releasing the advisory lock — actually reliable.
    """
    task = asyncio.ensure_future(coro)

    while True:
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            if task.done():
                # The task itself finished — successfully, or with its own
                # exception/cancellation — surface that outcome instead of
                # this shield-level CancelledError.
                return task.result()
