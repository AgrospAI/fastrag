"""The ``QueryEngine`` → SSE loop: one generation, start to finish."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator

from fastrag._ids import new_id
from fastrag.models import Document
from fastrag.query.chat.base import (
    ChatMessage,
    ChatStore,
    ErrorCode,
    FinishReason,
    to_llm_history,
)
from fastrag.query.engine import (
    DoneChunk,
    QueryEngine,
    SourcesChunk,
    TokenChunk,
)
from fastrag.query.prompts.templates import group_sources, page_url
from fastrag.serve._errors import _error_code_for
from fastrag.serve._fragments import text_fragments
from fastrag.serve._locking import LockKeepAlive, release_chat_lock
from fastrag.serve._metrics import ServeMetrics
from fastrag.serve._schemas import (
    DeltaEvent,
    ErrorInfo,
    MessageEndEvent,
    MessageStartEvent,
    Source,
    SourcesEvent,
    TextDelta,
)
from fastrag.serve._shield import shield_until_done
from fastrag.serve._sse import sse_frame
from fastrag.serve.config import ServeConfig

#: Hard cap on stored assistant content. A generation that reaches it is
#: stored truncated as ``error`` / ``INTERNAL_ERROR``: the bound was
#: misconfigured, which is a different event from the model stopping at its
#: own limit.
MAX_CONTENT_CHARS = 20000

#: Hard cap on the number of sources persisted per message, keeping the
#: ``jsonb_array_length(sources) <= 20`` CHECK safe even at ``k=50``.
MAX_SOURCES = 20

#: Hard cap on the persisted sources' JSON size, keeping the
#: ``octet_length(sources::text) <= 10000`` CHECK safe: a payload over it
#: would fail the assistant message's INSERT.
MAX_SOURCES_BYTES = 10000

#: Page size used to replay a chat's history, capped at the store's own
#: ``MAX_LIMIT``.
_HISTORY_PAGE = 100


def _sources_payload(documents: list[Document]) -> list[Source] | None:
    """Map ``Document`` to :class:`Source`, one per page.

    ``url = page_url(document)``, the ``source_uri`` or, for a PDF, the
    ``#page=N`` link of its page — a retrieved ``Document``'s provenance lives
    in its metadata, not an attribute (see ``fastrag.query``'s notes).
    ``title = document.metadata.get("title") or url``;
    ``site_name``/``favicon_url`` are the ones the fetcher found on the page,
    when it found any. Documents with no URL are dropped.
    One source per page is deliberate: ``k=5`` chunks routinely come from two
    pages, and "the full retrieved set, not the cited subset" is preserved
    at page granularity, each chunk as one of the page's ``fragments``. Best
    page first, hard-capped at ``MAX_SOURCES`` and ``MAX_SOURCES_BYTES``.

    ``positions`` maps the answer's ``[n]`` citations back to pages:
    ``group_sources`` groups the documents exactly as
    ``PromptTemplate.render`` numbers them from ``[1]``.
    """
    payload: list[Source] = []
    for position, chunks in enumerate(group_sources(documents), start=1):
        metadata = chunks[0].metadata
        url = page_url(chunks[0])
        if not url:
            continue
        if len(payload) == MAX_SOURCES:
            break
        payload.append(
            Source(
                url=url,
                title=metadata.get("title") or url,
                site_name=metadata.get("site_name"),
                favicon_url=metadata.get("favicon_url"),
                positions=[position],
                fragments=text_fragments(chunks),
            )
        )
    while payload and _stored_size(payload) > MAX_SOURCES_BYTES:
        payload.pop()
    return payload


def _stored_size(sources: list[Source]) -> int:
    """Bytes ``sources`` occupy as Postgres renders the stored ``jsonb``.

    ``jsonb``'s text form separates with ``", "``/``": "`` and keeps
    non-ASCII characters unescaped, as ``json.dumps`` does with
    ``ensure_ascii=False``.
    """
    stored = [source.model_dump(exclude_none=True) for source in sources]
    return len(json.dumps(stored, ensure_ascii=False).encode())


async def stream_generation(
    *,
    query_engine: QueryEngine,
    chat_store: ChatStore,
    config: ServeConfig,
    metrics: ServeMetrics,
    chat_id: str,
    user_message: ChatMessage,
    lock_connection,
) -> AsyncIterator[bytes]:
    """Run one generation and persist its outcome, start to finish.

    The caller (``_routes.py``) is responsible for acquiring the advisory
    lock via ``_locking.acquire_chat_lock`` (synchronously, before the stream
    opens) and persisting the user message / applying the edit before calling
    this; ``lock_connection`` is pinged while the generation runs and
    released here, in a ``finally``, once it ends.
    """
    keepalive = LockKeepAlive(lock_connection, config.keepalive_interval_s)
    try:
        async for frame in _generate(
            query_engine=query_engine,
            chat_store=chat_store,
            config=config,
            metrics=metrics,
            chat_id=chat_id,
            user_message=user_message,
        ):
            yield frame
    finally:
        # Shielded: on client disconnect this finally runs inside an
        # already-cancelled scope, and an unshielded await here can be
        # cancelled before the unlock is sent — stranding the chat locked.
        await shield_until_done(
            release_chat_lock(lock_connection, chat_id, keepalive=keepalive)
        )


async def _generate(
    *,
    query_engine: QueryEngine,
    chat_store: ChatStore,
    config: ServeConfig,
    metrics: ServeMetrics,
    chat_id: str,
    user_message: ChatMessage,
) -> AsyncIterator[bytes]:
    assistant_id = new_id()

    finish_reason: FinishReason = "interrupted"
    error_code: ErrorCode | None = None
    error_message: str | None = None
    content_parts: list[str] = []
    sources_payload: list[Source] | None = None

    # The clock a TTFT/TTLT pair is measured against: started once, here, so
    # a generation that streams for several seconds records a TTFT strictly
    # smaller than its TTLT. TTFT is observed once, at the first TokenChunk
    # actually turned into an SSE frame below; TTLT is observed exactly once,
    # in the outer `finally`, which runs on every path this generator can
    # end on — done, timed out, errored, or a client disconnect.
    generation_started = time.monotonic()
    first_token_observed = False

    metrics.llm_question_length_chars.observe(len(user_message.content or ""))

    try:
        yield sse_frame(
            "message_start",
            MessageStartEvent(
                chat_id=chat_id,
                user_message_id=user_message.id,
                message_id=assistant_id,
            ).model_dump(mode="json"),
        )

        try:
            # The read runs from inside the SSE generator, where an early
            # client disconnect would otherwise race FastAPI's dependency
            # teardown mid-query. On a regenerate, the truncation was already
            # applied before this generator opened, so this sees the edited
            # history and not the old one.
            history = to_llm_history(
                await _load_history(chat_store, chat_id, user_message.id)
            )

            # GENERATION_TIMEOUT_S is an end-to-end budget (retrieval +
            # generation), not just the model call — an unreachable vector
            # store must not hang forever either.
            completed = False
            async with asyncio.timeout(config.generation_timeout_s):
                async for chunk in query_engine.stream(
                    user_message.content or "", history=history
                ):
                    match chunk:
                        case SourcesChunk():
                            sources_payload = _sources_payload(chunk.sources)
                            yield sse_frame(
                                "sources",
                                SourcesEvent(
                                    message_id=assistant_id,
                                    sources=sources_payload or [],
                                ).model_dump(mode="json"),
                            )
                        case TokenChunk():
                            content_parts.append(chunk.token)
                            if not first_token_observed:
                                first_token_observed = True
                                metrics.llm_time_to_first_token_seconds.observe(
                                    time.monotonic() - generation_started
                                )
                            yield sse_frame(
                                "delta",
                                DeltaEvent(
                                    message_id=assistant_id,
                                    delta=TextDelta(text=chunk.token),
                                ).model_dump(mode="json"),
                            )
                        case DoneChunk():
                            completed = True
                            break

            if completed:
                finish_reason = "stop"

        except TimeoutError:
            finish_reason = "error"
            error_code = "TIMEOUT"
            error_message = "Generation timed out"
        except Exception as exc:  # noqa: BLE001 - every failure lands on message_end
            finish_reason = "error"
            error_code = _error_code_for(exc)
            error_message = str(exc)

        yield sse_frame(
            "message_end",
            MessageEndEvent(
                message_id=assistant_id,
                finish_reason=finish_reason,
                error=ErrorInfo(code=error_code, message=error_message or "")
                if error_code
                else None,
            ).model_dump(mode="json", exclude_none=True),
        )

    finally:
        # Reached on the happy path, on a caught error above, and on a client
        # disconnect (finish_reason stays "interrupted" and no further frame
        # is yielded) — streaming is stateless by design. This is also the
        # one point every generation reaches exactly once, regardless of how
        # it ends, so it is where TTLT and the answer length are observed —
        # synchronous calls, safe to make even inside an already-cancelled
        # scope, unlike the awaited persist below.
        content = "".join(content_parts)[:MAX_CONTENT_CHARS]
        if len("".join(content_parts)) > MAX_CONTENT_CHARS:
            finish_reason = "error"
            error_code = "INTERNAL_ERROR"
            error_message = "Generation exceeded the content bound"

        metrics.llm_time_to_last_token_seconds.observe(
            time.monotonic() - generation_started
        )
        metrics.llm_answer_length_chars.observe(len(content))

        # shield_until_done: on client disconnect this finally runs inside an
        # already-cancelled scope, where an unshielded await can be cancelled
        # mid-write — after the INSERT is sent but before COMMIT — leaving
        # the generation's outcome unrecorded.
        await shield_until_done(
            _persist_assistant(
                chat_store,
                id=assistant_id,
                chat_id=chat_id,
                content=content,
                sources=sources_payload,
                finish_reason=finish_reason,
                error_code=error_code,
                error_message=error_message,
            )
        )


async def _load_history(
    chat_store: ChatStore, chat_id: str, message_id: str
) -> list[ChatMessage]:
    """Replay the chat's prior turns, newest-first across cursor pages.

    Fetches every message with ``id < message_id`` via cursor paging. The
    caller converts the returned list to model-facing ``Message`` history
    with :func:`~fastrag.query.chat.base.to_llm_history`, which reorders it
    chronologically and drops any turn with ``content is None`` (a failed
    generation that produced no text).
    """
    messages: list[ChatMessage] = []
    before: str | None = message_id
    while True:
        page = await chat_store.list_messages(
            chat_id, before=before, limit=_HISTORY_PAGE
        )
        messages.extend(page.items)
        if not page.has_more or not page.items:
            break
        before = page.items[-1].id
    return messages


async def _persist_assistant(
    chat_store: ChatStore,
    *,
    id: str,
    chat_id: str,
    content: str | None,
    sources: list[Source] | None,
    finish_reason: FinishReason,
    error_code: ErrorCode | None,
    error_message: str | None,
) -> None:
    """Write the assistant message's outcome via the store's own primitive.

    On client disconnect this write runs inside an already-cancelled scope,
    which is why the caller wraps it in
    :func:`~fastrag.serve._shield.shield_until_done`.
    """
    await chat_store.append_message(
        chat_id=chat_id,
        role="assistant",
        content=content,
        sources=[s.model_dump(exclude_none=True) for s in sources] if sources else None,
        finish_reason=finish_reason,
        error_code=error_code,
        error_message=error_message,
    )
