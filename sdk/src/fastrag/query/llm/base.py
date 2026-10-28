"""The ``ChatModel`` contract: messages to a generated answer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True, slots=True)
class Message:
    """One turn in a chat conversation.

    Attributes:
        role: Who sent the turn.
        content: The turn's text.
    """

    role: Literal["system", "user", "assistant"]
    content: str


class ChatModel(ABC):
    """Turns a conversation into a generated answer.

    ``invoke`` and ``stream`` are separate methods, not one method with a
    flag, because a streaming caller and a one-shot caller want different
    return shapes — tokens as they land versus one finished string — and
    forcing both through one signature would make one of the two callers
    reconstruct what it actually wanted from the other's shape.
    """

    @abstractmethod
    async def invoke(self, messages: Sequence[Message]) -> str:
        """Generate the whole answer at once, as plain text."""

    @abstractmethod
    def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        """Generate the answer as text deltas, nothing else.

        Yields plain text tokens only, so a provider that cannot report usage
        mid-stream never has to fabricate it.
        """

    @property
    @abstractmethod
    def model_id(self) -> str:
        """Stable identifier of the model, for logging and events."""
