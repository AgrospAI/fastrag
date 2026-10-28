"""Pydantic request and response models for the serving API."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fastrag.query.chat import ChatMessage
from fastrag.query.chat.base import ERROR_CODES, FINISH_REASONS, MESSAGE_ROLES
from fastrag.serve._errors import REQUEST_ERROR_CODES

#: Named OpenAPI enum components, built from the single source of truth each
#: mirrors (``fastrag.query.chat.base``'s tuples, and this module's own
#: ``REQUEST_ERROR_CODES``) rather than a second hand-typed list — adding a
#: value to one of those tuples is picked up here automatically the next
#: time the app is imported and ``task openapi`` regenerates the checked-in
#: document. A plain ``Literal[...]`` field would work at runtime but is
#: inlined anonymously in the OpenAPI schema on every use; an ``Enum`` gets
#: a named, reusable ``components/schemas`` entry, which is what lets a
#: TypeScript generator produce one shared union type instead of repeating
#: the values at every call site.
MessageRole = Enum("MessageRole", {v.upper(): v for v in MESSAGE_ROLES}, type=str)
"""Who sent a turn — mirrors ``fastrag.query.chat.base.MESSAGE_ROLES``."""

FinishReason = Enum("FinishReason", {v.upper(): v for v in FINISH_REASONS}, type=str)
"""Why a generation stopped — mirrors ``fastrag.query.chat.base.FINISH_REASONS``."""

ErrorCode = Enum("ErrorCode", {v.upper(): v for v in ERROR_CODES}, type=str)
"""``error.code`` on a failed *generation* — mirrors
``fastrag.query.chat.base.ERROR_CODES``. Narrower than :data:`ApiErrorCode`:
a request that never reaches generation never produces one of these (see
``streaming.md`` and ``_errors.py``'s ``_error_code_for`` docstring)."""

ApiErrorCode = Enum(
    "ApiErrorCode", {v.upper(): v for v in REQUEST_ERROR_CODES}, type=str
)
"""``error.code`` on a request-level failure — a ``4xx``/``5xx`` raised
before any stream opens. Mirrors ``_errors.REQUEST_ERROR_CODES``."""


class ContentRequest(BaseModel):
    """``POST /v1/chats`` and ``POST /v1/chats/{chat_id}/messages`` body."""

    content: str

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        return v


class OptionalContentRequest(BaseModel):
    """``POST /v1/chats/{chat_id}/messages/{message_id}`` body.

    ``content`` is only meaningful when the target message is a ``user``
    message — rejected with ``422`` when the target is ``assistant`` (checked
    in the route).
    """

    content: str | None = None

    @field_validator("content")
    @classmethod
    def validate_content(cls, v: str | None) -> str | None:
        if v is None:
            return v
        v = v.strip()
        if not v:
            raise ValueError("content must not be empty")
        return v


class ChatSummary(BaseModel):
    id: str
    created_at: datetime

    model_config = ConfigDict(from_attributes=True)


class ChatListResponse(BaseModel):
    first_id: str | None
    last_id: str | None
    has_more: bool
    chats: Sequence[ChatSummary]


class ErrorInfo(BaseModel):
    """A failed *generation*'s outcome — ``code`` is one of :data:`ErrorCode`."""

    code: ErrorCode
    message: str


class ApiErrorInfo(BaseModel):
    """A request-level failure's outcome — ``code`` is one of :data:`ApiErrorCode`."""

    code: ApiErrorCode
    message: str


class ApiErrorEnvelope(BaseModel):
    """The ``{"error": {"code", "message"}}`` body of a request-level failure.

    Raised as a ``4xx``/``5xx`` before a generation's stream opens. Never
    returned through FastAPI's response-model validation — the
    exception handlers in ``_errors.py`` build this exact shape by hand as a
    plain dict, since a ``BaseHTTPMiddleware``/exception-handler response
    bypasses per-route ``response_model`` machinery. This model exists only
    so the shape still has a name in the OpenAPI document: it is attached to
    every ``/v1`` route via ``include_router(responses=...)`` in ``app.py``.
    """

    error: ApiErrorInfo


class TextFragment(BaseModel):
    """One passage of a page, as the terms of a URL text fragment.

    A link to ``url#:~:text=start,end`` scrolls to and highlights the text
    from ``start`` through ``end``; to ``url#:~:text=start`` when ``end`` is
    ``None``, the text ``start`` itself. The terms are plain text: a client
    percent-encodes them, including ``-``, ``,`` and ``&``, which the syntax
    reserves.
    """

    start: str
    end: str | None = None


class Source(BaseModel):
    """One retrieved page an answer may cite, deduplicated by ``url``."""

    url: str
    """The page's address; a PDF page's ends in ``#page=N``."""
    title: str
    site_name: str | None = None
    """The site's own name (``og:site_name``), when its page declared one."""
    favicon_url: str | None = None
    """The page's declared icon, an absolute ``http(s)`` URL, when it has one."""
    positions: list[int] = Field(default_factory=list)
    """Every ``n`` an answer cites this source as ``[n]``.

    The prompt numbers each retrieved page once, so this holds one number.
    Messages stored when it numbered each chunk can hold several, and ones
    stored before this field existed hold none.
    """
    fragments: list[TextFragment] = Field(default_factory=list)
    """The passages of the page the answer was given, in page order.

    Chunks of one section are joined into one passage. Empty for a page the
    browser cannot search, such as a PDF, and on older messages.
    """


class TextDelta(BaseModel):
    """One token of an assistant answer, as the ``delta`` SSE event's ``delta`` field."""

    type: Literal["text"] = "text"
    text: str


class MessageStartEvent(BaseModel):
    """Payload of the SSE ``message_start`` event — see ``streaming.md``."""

    chat_id: str
    user_message_id: str
    message_id: str


class SourcesEvent(BaseModel):
    """Payload of the SSE ``sources`` event."""

    message_id: str
    sources: Sequence[Source]


class DeltaEvent(BaseModel):
    """Payload of the SSE ``delta`` event."""

    message_id: str
    delta: TextDelta


class MessageEndEvent(BaseModel):
    """Payload of the SSE ``message_end`` event.

    ``error`` is present only when ``finish_reason == "error"`` — omitted
    from the wire frame entirely otherwise, not sent as ``null``
    (``_generation.py`` dumps this with ``exclude_none=True``).
    """

    message_id: str
    finish_reason: FinishReason
    error: ErrorInfo | None = None


class MessageResponse(BaseModel):
    id: str
    role: MessageRole
    content: str | None
    sources: Sequence[Source] | None = None
    finish_reason: FinishReason | None = None
    error: ErrorInfo | None = None

    @classmethod
    def from_chat_message(cls, message: ChatMessage) -> MessageResponse:
        error = None
        if message.error_code:
            error = ErrorInfo(
                code=message.error_code, message=message.error_message or ""
            )

        return cls(
            id=message.id,
            role=message.role,
            content=message.content,
            sources=message.sources,
            finish_reason=message.finish_reason,
            error=error,
        )


class MessageListResponse(BaseModel):
    first_id: str | None
    last_id: str | None
    has_more: bool
    messages: Sequence[MessageResponse]


class SessionResponse(BaseModel):
    session_id: str
    expires_at: datetime
