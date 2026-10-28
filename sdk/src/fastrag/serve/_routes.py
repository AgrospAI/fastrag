"""The ``/v1`` router and ``/healthz``."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends, Request, Response
from fastapi.responses import StreamingResponse

from fastrag.query.chat.base import Chat, ChatMessage, ChatStore
from fastrag.serve._auth import (
    Authenticated,
    get_or_create_session,
    require_session,
    set_session_cookie,
)
from fastrag.serve._errors import ApiError
from fastrag.serve._generation import stream_generation
from fastrag.serve._guards import enforce_content_length, rate_limit
from fastrag.serve._locking import acquire_chat_lock, release_chat_lock
from fastrag.serve._metrics import ServeMetrics
from fastrag.serve._schemas import (
    ChatListResponse,
    ChatSummary,
    ContentRequest,
    MessageListResponse,
    MessageResponse,
    OptionalContentRequest,
    SessionResponse,
)
from fastrag.serve._sse import SSE_HEADERS, with_keepalive
from fastrag.serve.config import ServeConfig

router = APIRouter(prefix="/v1", tags=["chats"])

# ``/healthz`` sits outside the ``/v1`` base path, so it gets its own router
# without the prefix.
healthz_router = APIRouter(tags=["health"])

#: Default and maximum page size for the list endpoints, matching the store's
#: own ``DEFAULT_LIMIT``/``MAX_LIMIT``.
DEFAULT_LIMIT = 50
MAX_LIMIT = 100


def clamp_limit(limit: int) -> int:
    """Clamp ``limit`` into ``[1, MAX_LIMIT]``."""
    return max(1, min(limit, MAX_LIMIT))


def _get_config(request: Request) -> ServeConfig:
    return request.app.state.config


def _get_store(request: Request) -> ChatStore:
    return request.app.state.chat_store


def _get_query_engine(request: Request):
    return request.app.state.query_engine


def _get_engine(request: Request):
    return request.app.state.engine


def _get_metrics(request: Request) -> ServeMetrics:
    return request.app.state.metrics


async def _lock_persist_and_stream(
    request: Request,
    chat: Chat,
    session: Authenticated,
    persist,
) -> StreamingResponse:
    """Acquire the per-chat advisory lock, persist, and open the SSE stream.

    The lock is acquired *before* persisting the prompt and *before* the
    stream opens, so lock contention is a plain HTTP ``409`` rather than a
    mid-stream failure. ``persist`` runs the caller's write (new user
    message) while the lock is held; the lock is released once the
    generation this call starts finishes (in
    ``_generation.stream_generation``'s ``finally``).
    """
    config = _get_config(request)
    engine = _get_engine(request)
    store = _get_store(request)

    lock_connection = await acquire_chat_lock(engine, chat.id)
    try:
        user_message = await persist()
    except Exception:
        await release_chat_lock(lock_connection, chat.id)
        raise

    response = StreamingResponse(
        with_keepalive(
            stream_generation(
                query_engine=_get_query_engine(request),
                chat_store=store,
                config=config,
                metrics=_get_metrics(request),
                chat_id=chat.id,
                user_message=user_message,
                lock_connection=lock_connection,
            ),
            interval=config.keepalive_interval_s,
        ),
        media_type="text/event-stream",
        headers=SSE_HEADERS,
    )
    # FastAPI only merges a dependency-injected Response's cookies into the
    # final response for implicit (model) returns; an endpoint returning its
    # own StreamingResponse bypasses that, so the session cookie must be
    # re-applied directly on the object actually returned here.
    if session.via == "cookie":
        set_session_cookie(response, session.session.id, config)
    return response


@router.post(
    "/chats", status_code=200, dependencies=[Depends(rate_limit("generation"))]
)
async def create_chat(
    request: Request,
    body: ContentRequest,
    session: Annotated[Authenticated, Depends(get_or_create_session)],
) -> StreamingResponse:
    """Create a chat and stream the first answer.

    The only endpoint that doesn't require an existing session.
    """
    config = _get_config(request)
    store = _get_store(request)
    enforce_content_length(body.content, config)

    chats = await store.list_chats(
        session.session.id, limit=config.max_chats_per_session + 1
    )
    if len(chats.items) >= config.max_chats_per_session:
        raise ApiError(422, "INVALID_REQUEST", "MAX_CHATS_PER_SESSION reached")

    chat = await store.create_chat(
        session_id=session.session.id, ttl_days=config.session_ttl_days
    )

    async def persist() -> ChatMessage:
        return await store.append_message(chat.id, role="user", content=body.content)

    return await _lock_persist_and_stream(request, chat, session, persist)


@router.post(
    "/chats/{chat_id}/messages",
    status_code=200,
    dependencies=[Depends(rate_limit("generation"))],
)
async def add_message(
    request: Request,
    chat_id: str,
    body: ContentRequest,
    session: Annotated[Authenticated, Depends(require_session)],
) -> StreamingResponse:
    config = _get_config(request)
    store = _get_store(request)
    enforce_content_length(body.content, config)

    chat = await store.get_chat(chat_id, session_id=session.session.id)
    if chat is None:
        raise ApiError(404, "NOT_FOUND", "Chat not found")

    messages = await store.list_messages(
        chat_id, limit=config.max_messages_per_chat + 1
    )
    if len(messages.items) >= config.max_messages_per_chat:
        raise ApiError(422, "INVALID_REQUEST", "MAX_MESSAGES_PER_CHAT reached")

    async def persist() -> ChatMessage:
        return await store.append_message(chat.id, role="user", content=body.content)

    return await _lock_persist_and_stream(request, chat, session, persist)


@router.post(
    "/chats/{chat_id}/messages/{message_id}",
    status_code=200,
    dependencies=[Depends(rate_limit("generation"))],
)
async def regenerate_message(
    request: Request,
    chat_id: str,
    message_id: str,
    body: OptionalContentRequest,
    session: Annotated[Authenticated, Depends(require_session)],
) -> StreamingResponse:
    """Regenerate or edit the answer for a turn and stream it.

    Covers both editing/resubmitting a user question (``body.content`` set,
    target a ``user`` message) and regenerating an assistant answer
    (``body.content`` absent, target an ``assistant`` message). The
    ``ChatStore`` has no range-delete truncation primitive (task 027), so the
    edit is expressed as appending the resubmitted question as a fresh user
    turn and regenerating from it — the chat stays linear and the new answer
    lands as the latest assistant message.
    """
    config = _get_config(request)
    store = _get_store(request)
    if body.content is not None:
        enforce_content_length(body.content, config)

    chat = await store.get_chat(chat_id, session_id=session.session.id)
    if chat is None:
        raise ApiError(404, "NOT_FOUND", "Chat not found")

    target = await _message_by_id(store, chat_id, message_id)
    if target is None:
        raise ApiError(404, "NOT_FOUND", "Message not found")

    if target.role == "assistant" and body.content is not None:
        raise ApiError(
            422,
            "INVALID_REQUEST",
            "content is not allowed when regenerating an assistant message",
        )

    if target.role == "assistant":
        reuse = await _preceding_user_message(store, chat_id, target.id)
        if reuse is None:
            raise ApiError(
                422, "INVALID_REQUEST", "No preceding user message to regenerate from"
            )
        question = reuse.content or ""
    else:
        question = body.content if body.content is not None else (target.content or "")

    async def persist() -> ChatMessage:
        # Append the (possibly edited) question as a fresh user turn; the
        # generation appends the new assistant answer. No rows are deleted —
        # the store has no range-delete primitive.
        return await store.append_message(chat_id, role="user", content=question)

    return await _lock_persist_and_stream(request, chat, session, persist)


async def _message_by_id(
    store: ChatStore, chat_id: str, message_id: str
) -> ChatMessage | None:
    """Fetch one message by id, scanning newest-first pages."""
    before: str | None = None
    while True:
        page = await store.list_messages(chat_id, before=before, limit=100)
        for message in page.items:
            if message.id == message_id:
                return message
        if not page.has_more or not page.items:
            return None
        before = page.items[-1].id


async def _preceding_user_message(
    store: ChatStore, chat_id: str, before_id: str
) -> ChatMessage | None:
    """Return the most recent ``user`` message strictly older than ``before_id``."""
    page = await store.list_messages(chat_id, before=before_id, limit=100)
    for message in page.items:
        if message.role == "user":
            return message
    return None


@router.get("/chats", dependencies=[Depends(rate_limit("read"))])
async def list_chats(
    request: Request,
    session: Annotated[Authenticated, Depends(require_session)],
    before: str | None = None,
    limit: int = 50,
) -> ChatListResponse:
    store = _get_store(request)

    page = await store.list_chats(
        session.session.id, before=before, limit=clamp_limit(limit)
    )
    summaries = [
        ChatSummary(id=chat.id, created_at=chat.created_at) for chat in page.items
    ]

    return ChatListResponse(
        first_id=summaries[0].id if summaries else None,
        last_id=summaries[-1].id if summaries else None,
        has_more=page.has_more,
        chats=summaries,
    )


@router.get("/chats/{chat_id}", dependencies=[Depends(rate_limit("read"))])
async def get_chat(
    request: Request,
    chat_id: str,
    session: Annotated[Authenticated, Depends(require_session)],
    before: str | None = None,
    limit: int = 50,
) -> MessageListResponse:
    store = _get_store(request)

    chat = await store.get_chat(chat_id, session_id=session.session.id)
    if chat is None:
        raise ApiError(404, "NOT_FOUND", "Chat not found")

    page = await store.list_messages(chat_id, before=before, limit=clamp_limit(limit))
    responses = [MessageResponse.from_chat_message(m) for m in page.items]

    return MessageListResponse(
        first_id=responses[0].id if responses else None,
        last_id=responses[-1].id if responses else None,
        has_more=page.has_more,
        messages=responses,
    )


@router.delete("/chats/{chat_id}", status_code=204)
async def delete_chat(
    request: Request,
    chat_id: str,
    session: Annotated[Authenticated, Depends(require_session)],
) -> Response:
    config = _get_config(request)
    store = _get_store(request)

    deleted = await store.delete_chat(chat_id, session_id=session.session.id)
    if not deleted:
        raise ApiError(404, "NOT_FOUND", "Chat not found")

    response = Response(status_code=204)
    if session.via == "cookie":
        set_session_cookie(response, session.session.id, config)
    return response


@router.post("/sessions", status_code=200)
async def get_session(
    request: Request,
    response: Response,
    session: Annotated[Authenticated, Depends(get_or_create_session)],
) -> SessionResponse:
    """Return the caller's current session id, minting one only when needed.

    ``200`` when it revealed an existing session, ``201`` when it
    minted one. ``404`` when bearer tokens are disabled. Sets no cookie:
    this is the header client's channel.
    """
    config = _get_config(request)
    if not config.bearer_tokens:
        raise ApiError(404, "NOT_FOUND", "Not found")
    response.status_code = 201 if session.minted else 200
    return SessionResponse(
        session_id=session.session.id,
        expires_at=session.session.expires_at,
    )


@healthz_router.get("/", status_code=200, include_in_schema=False)
async def root() -> dict[str, str]:
    """Root endpoint for health checks and basic info.

    Left out of the OpenAPI schema: clients call ``/v1`` and probes call
    ``/healthz``, so the generated client types have no use for it.
    """
    return {"status": "ok", "version": "1.0.0"}


@healthz_router.get("/healthz")
async def healthz(request: Request) -> Response:
    """Liveness and readiness.

    Returns ``200 {"status": "ok"}``. With a ``ChatStore`` entered, checks
    the store's engine connection (``503`` when unreachable); the store's
    engine is the only one the app owns — it creates none itself.
    """
    engine = request.app.state.engine
    if engine is None:
        return Response(
            content='{"status": "ok"}',
            media_type="application/json",
        )
    try:
        async with engine.connect():
            return Response(
                content='{"status": "ok"}',
                media_type="application/json",
            )
    except Exception:  # noqa: BLE001 - any connection failure is a 503
        return Response(
            content='{"status": "unavailable"}',
            media_type="application/json",
            status_code=503,
        )
