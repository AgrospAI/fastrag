"""The ``ChatStore`` contract: sessions, chats, and messages.

Serving a chatbot means remembering conversations: anonymous sessions, the
chats inside them, and the messages inside those. ``ChatStore`` is the
pluggable contract for that persistence — a session's identity and expiry, a
chat's ownership and lifecycle, and a message's role/content/generation
outcome — covering exactly what a serving layer needs and nothing about how
that layer is built (no FastAPI, no HTTP types, no SSE).

A persisted :class:`ChatMessage` is a different thing from
:class:`~fastrag.query.llm.base.Message`, the model-facing turn a
:class:`~fastrag.query.llm.base.ChatModel` consumes: a ``ChatMessage`` carries
an id, timestamps, sources, and a generation outcome that a ``Message`` has no
use for. :func:`to_llm_history` is the conversion between them.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Sequence
from contextlib import AbstractAsyncContextManager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Generic, Literal, Self, TypeVar

from fastrag.query.llm.base import Message

MessageRole = Literal["user", "assistant", "system"]
"""Who sent a turn."""

FinishReason = Literal["stop", "length", "error", "interrupted"]
"""Why an assistant turn stopped generating."""

ErrorCode = Literal[
    "MODEL_ERROR",
    "TIMEOUT",
    "RATE_LIMIT",
    "CONTEXT_TOO_LONG",
    "NETWORK_ERROR",
    "SERVICE_UNAVAILABLE",
    "INVALID_REQUEST",
    "INVALID_RESPONSE",
    "INTERNAL_ERROR",
]
"""Machine-readable reason an assistant turn ended in ``finish_reason="error"``."""

#: The literal values above, as plain tuples — the single source of truth
#: both this module's ``Literal`` aliases and ``fastrag.query.chat.schema``'s
#: Postgres ``ENUM`` types are built from, so the two can never drift apart.
MESSAGE_ROLES: tuple[str, ...] = ("user", "assistant", "system")
FINISH_REASONS: tuple[str, ...] = ("stop", "length", "error", "interrupted")
ERROR_CODES: tuple[str, ...] = (
    "MODEL_ERROR",
    "TIMEOUT",
    "RATE_LIMIT",
    "CONTEXT_TOO_LONG",
    "NETWORK_ERROR",
    "SERVICE_UNAVAILABLE",
    "INVALID_REQUEST",
    "INVALID_RESPONSE",
    "INTERNAL_ERROR",
)

#: Default and maximum page size for :meth:`ChatStore.list_chats` and
#: :meth:`ChatStore.list_messages`.
DEFAULT_LIMIT = 50
MAX_LIMIT = 100


@dataclass(frozen=True, slots=True)
class Session:
    """One anonymous browser identity, owning its chats.

    Attributes:
        id: ULID, minted on creation. The bearer credential itself — a
            caller who holds this id can act as this session — so it is
            never logged.
        created_at: Derived from ``id``, so ``ORDER BY id`` and
            ``ORDER BY created_at`` always agree.
        expires_at: When this session (and, by cascade, its chats and their
            messages) stops being usable.
        ip_address: The creating request's peer address, if known.
        country: A two-letter country code, if known.
    """

    id: str
    created_at: datetime
    expires_at: datetime
    ip_address: str | None = None
    country: str | None = None


@dataclass(frozen=True, slots=True)
class Chat:
    """One conversation: an ordered, linear list of messages.

    Attributes:
        id: ULID, minted on creation.
        session_id: The owning :class:`Session`.
        created_at: Derived from ``id``.
        expires_at: When this chat (and, by cascade, its messages) stops
            being usable. Independent of its session's ``expires_at`` — a
            chat can go idle while its session stays active.
    """

    id: str
    session_id: str
    created_at: datetime
    expires_at: datetime


@dataclass(frozen=True, slots=True)
class ChatMessage:
    """One persisted turn, from ``user``, ``assistant``, or ``system``.

    A different thing from :class:`~fastrag.query.llm.base.Message`, the
    model-facing turn a :class:`~fastrag.query.llm.base.ChatModel` consumes
    — see :func:`to_llm_history` for the conversion.

    Attributes:
        id: ULID, minted on creation.
        chat_id: The owning :class:`Chat`.
        created_at: Derived from ``id``.
        role: Who sent the turn.
        content: The turn's text. ``None`` only on an ``assistant`` turn
            whose ``finish_reason`` is ``"error"`` and nothing was generated
            before it failed.
        sources: The documents the answer was grounded in, serialized as
            plain JSON objects (a store never interprets their shape).
            ``None`` for a ``user``/``system`` turn.
        finish_reason: Why an ``assistant`` turn stopped. Required for
            ``assistant`` turns, absent otherwise.
        error_code: Set exactly when ``finish_reason == "error"``.
        error_message: Set exactly when ``finish_reason == "error"``.
    """

    id: str
    chat_id: str
    created_at: datetime
    role: MessageRole
    content: str | None = None
    sources: list[dict[str, Any]] | None = None
    finish_reason: FinishReason | None = None
    error_code: ErrorCode | None = None
    error_message: str | None = None


T = TypeVar("T")


@dataclass(frozen=True, slots=True)
class Page(Generic[T]):
    """One page of a ULID-cursor listing.

    Attributes:
        items: This page's rows, newest-first (highest id first).
        has_more: Whether an older row exists beyond this page — pass this
            page's last item's ``id`` as the next call's ``before`` to fetch
            it.
    """

    items: list[T]
    has_more: bool


def to_llm_history(messages: Sequence[ChatMessage]) -> list[Message]:
    """Turn a page of persisted messages into model-facing chat history.

    Reorders ``messages`` into chronological (oldest-first) order — the
    order every :class:`~fastrag.query.llm.base.ChatModel` expects — and
    drops any turn with ``content is None`` (an ``assistant`` turn that
    errored before generating anything), which a chat model has nothing to
    replay. A ``list_messages`` page is newest-first, so this is normally
    the conversion a caller applies right before building a prompt.
    """
    ordered = sorted(messages, key=lambda message: message.id)
    return [
        Message(role=message.role, content=message.content)
        for message in ordered
        if message.content is not None
    ]


class ChatStore(AbstractAsyncContextManager["ChatStore"], ABC):
    """Pluggable abstract persistence for sessions, chats, and messages.

    Storage only: no FastAPI, no HTTP types, no SSE, no ``QueryEngine`` — a
    serving layer built on top of a ``ChatStore`` owns all of that.
    """

    async def __aenter__(self) -> Self:
        """Open store resources."""
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: Any,
    ) -> None:
        """Release store resources."""
        return None

    # -- sessions ------------------------------------------------------

    @abstractmethod
    async def create_session(
        self,
        *,
        ttl_days: int,
        ip_address: str | None = None,
        country: str | None = None,
    ) -> Session:
        """Mint and persist a new session.

        Args:
            ttl_days: Days until the session expires without activity.
            ip_address: The creating request's peer address, if known.
            country: A two-letter country code, if known.
        """
        ...

    @abstractmethod
    async def get_session(self, session_id: str) -> Session | None:
        """Fetch an active session by id.

        Returns:
            The session, or ``None`` if ``session_id`` is unknown or its
            ``expires_at`` has already passed.
        """
        ...

    @abstractmethod
    async def touch_session(
        self, session_id: str, *, ttl_days: int, min_bump_days: int
    ) -> Session | None:
        """Bump an active session's expiry, throttled.

        The expiry is only written when the new value is at least
        ``min_bump_days`` later than the stored one, so an active
        conversation stays alive without a write on every request.

        Args:
            session_id: The session to touch.
            ttl_days: Days from now the session's new expiry would be, if
                bumped.
            min_bump_days: Minimum headroom, in days, the candidate expiry
                must gain over the stored one before it is written.

        Returns:
            The session as it now stands (bumped or not), or ``None`` if
            ``session_id`` is unknown or already expired.
        """
        ...

    # -- chats -----------------------------------------------------------

    @abstractmethod
    async def create_chat(self, *, session_id: str, ttl_days: int) -> Chat:
        """Start a new, empty chat owned by ``session_id``.

        Args:
            session_id: The owning session.
            ttl_days: Days until the chat expires without activity.
        """
        ...

    @abstractmethod
    async def get_chat(self, chat_id: str, *, session_id: str) -> Chat | None:
        """Fetch a chat, scoped to its owning session.

        Returns:
            The chat, or ``None`` if ``chat_id`` does not exist or does not
            belong to ``session_id``.
        """
        ...

    @abstractmethod
    async def list_chats(
        self,
        session_id: str,
        *,
        before: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> Page[Chat]:
        """List ``session_id``'s chats, newest-first.

        Args:
            session_id: The owning session.
            before: When given, only chats older than this chat id are
                returned — pass a previous page's last item's ``id`` to
                continue paging.
            limit: Maximum chats to return.
        """
        ...

    @abstractmethod
    async def delete_chat(self, chat_id: str, *, session_id: str) -> bool:
        """Delete a chat and every message in it.

        Args:
            chat_id: The chat to delete.
            session_id: The owning session — ``chat_id`` is only deleted if
                it belongs to this session.

        Returns:
            ``True`` if a chat was deleted, ``False`` if ``chat_id`` did not
            exist or did not belong to ``session_id``.
        """
        ...

    # -- messages --------------------------------------------------------

    @abstractmethod
    async def append_message(
        self,
        chat_id: str,
        *,
        role: MessageRole,
        content: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        finish_reason: FinishReason | None = None,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> ChatMessage:
        """Append a new turn to a chat.

        Args:
            chat_id: The chat this turn belongs to.
            role: Who sent the turn.
            content: The turn's text. May be ``None`` only for an
                ``assistant`` turn whose ``finish_reason`` is ``"error"``.
            sources: The documents the answer was grounded in, as plain JSON
                objects. Only meaningful for an ``assistant`` turn.
            finish_reason: Required for an ``assistant`` turn, must be
                absent otherwise.
            error_code: Set exactly when ``finish_reason == "error"``.
            error_message: Set exactly when ``finish_reason == "error"``.
        """
        ...

    @abstractmethod
    async def list_messages(
        self,
        chat_id: str,
        *,
        before: str | None = None,
        limit: int = DEFAULT_LIMIT,
    ) -> Page[ChatMessage]:
        """List a chat's messages, newest-first.

        Args:
            chat_id: The chat to list.
            before: When given, only messages older than this message id are
                returned — pass a previous page's last item's ``id`` to
                continue paging.
            limit: Maximum messages to return.
        """
        ...

    @abstractmethod
    async def update_last_assistant_message(
        self,
        chat_id: str,
        *,
        content: str | None,
        sources: list[dict[str, Any]] | None = None,
        finish_reason: FinishReason,
        error_code: ErrorCode | None = None,
        error_message: str | None = None,
    ) -> ChatMessage | None:
        """Overwrite a chat's most recent ``assistant`` turn in place.

        The storage primitive behind both the ``/retry`` flow (regenerate
        the last answer) and the ``/edit`` flow (regenerate after editing an
        earlier turn): both end by replacing the chat's latest ``assistant``
        message with a new outcome, in place, rather than deleting and
        re-appending.

        Args:
            chat_id: The chat whose latest ``assistant`` message is updated.
            content: The turn's new text. May be ``None`` only when
                ``finish_reason == "error"``.
            sources: The documents the new answer was grounded in.
            finish_reason: Why generation stopped this time.
            error_code: Set exactly when ``finish_reason == "error"``.
            error_message: Set exactly when ``finish_reason == "error"``.

        Returns:
            The updated message, or ``None`` if ``chat_id`` has no
            ``assistant`` message yet.
        """
        ...
