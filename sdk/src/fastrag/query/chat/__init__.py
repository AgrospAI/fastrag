"""Chat persistence: sessions, chats, and messages.

The :class:`~fastrag.query.chat.base.ChatStore` contract and its PostgreSQL
implementation. Storage only — no FastAPI, no HTTP types, no SSE; a serving
layer built on top of a ``ChatStore`` owns all of that. See
:mod:`fastrag.query.chat.base` for what a serving layer built on this
package gets: session create/read/touch, chat create/list/read/delete, and
message append/list/update-last-assistant.
"""

from __future__ import annotations

from fastrag.query.chat.base import (
    Chat,
    ChatMessage,
    ChatStore,
    ErrorCode,
    FinishReason,
    MessageRole,
    Page,
    Session,
    to_llm_history,
)
from fastrag.query.chat.postgres import PostgresChatStore

__all__ = [
    "Chat",
    "ChatMessage",
    "ChatStore",
    "ErrorCode",
    "FinishReason",
    "MessageRole",
    "Page",
    "PostgresChatStore",
    "Session",
    "to_llm_history",
]
