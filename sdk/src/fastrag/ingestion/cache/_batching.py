"""Coalesce concurrent cache reads without retaining results between calls."""

from __future__ import annotations

import asyncio
from collections import deque
from collections.abc import Awaitable, Callable, Sequence


class ReadBatcher[T]:
    """Collect independent waiters into bounded bulk reads.

    Each caller owns a future: cancelling one never cancels another's read.
    Only the active batch's results are retained, and failures reach every
    affected caller so normal per-item retry policies still apply.
    """

    def __init__(
        self,
        read: Callable[[Sequence[str]], Awaitable[dict[str, T]]],
        size: int,
        delay: float,
    ) -> None:
        self._read = read
        self._size = size
        self._delay = delay
        self._pending: deque[tuple[str, asyncio.Future[T]]] = deque()
        self._task: asyncio.Task[None] | None = None

    async def get(self, key: str) -> T:
        """Join the next batch and wait for this caller's result."""
        future = asyncio.get_running_loop().create_future()
        self._pending.append((key, future))
        if self._task is None:
            self._task = asyncio.create_task(self._run())
        return await future

    async def close(self) -> None:
        """Cancel outstanding reads before the backend disposes its engine."""
        if self._task is not None:
            self._task.cancel()
            await asyncio.gather(self._task, return_exceptions=True)

    async def _run(self) -> None:
        batch: list[tuple[str, asyncio.Future[T]]] = []
        try:
            while self._pending:
                # A timer flushes partial batches even at concurrency=1,
                # with slow upstreams, or at the end of a stream.
                await asyncio.sleep(self._delay)
                batch = []
                while self._pending and len(batch) < self._size:
                    key, future = self._pending.popleft()
                    if not future.done():
                        batch.append((key, future))
                if not batch:
                    continue
                read_task = asyncio.ensure_future(
                    self._read(list(dict.fromkeys(key for key, _ in batch)))
                )

                def cancel_unused_read(
                    _future,
                    waiters=tuple(future for _, future in batch),
                    task=read_task,
                ):
                    if not task.done() and all(future.done() for future in waiters):
                        task.cancel()

                for _, future in batch:
                    future.add_done_callback(cancel_unused_read)
                try:
                    results = await read_task
                    for key, future in batch:
                        if not future.done():
                            future.set_result(results[key])
                except asyncio.CancelledError:
                    # Timeouts cancel waiters, not the scheduler. Abandon an
                    # unused query so a blocked read cannot hold up all retries.
                    scheduler = asyncio.current_task()
                    assert scheduler is not None
                    if scheduler.cancelling():
                        raise
                    for _, future in batch:
                        if not future.done():
                            future.cancel()
                except Exception as error:
                    for _, future in batch:
                        if not future.done():
                            future.set_exception(error)
        finally:
            for _, future in [*batch, *self._pending]:
                if not future.done():
                    future.cancel()
            self._pending.clear()
            self._task = None
