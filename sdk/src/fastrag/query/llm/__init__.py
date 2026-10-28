"""Chat models: turning a conversation into a generated answer.

``OpenAIChatModel`` is the one implementation: any endpoint speaking the
OpenAI protocol, which also covers Ollama, vLLM and llama.cpp — only the
model name and credential change.
"""

from __future__ import annotations

from fastrag.query.llm.base import ChatModel, Message
from fastrag.query.llm.openai import OpenAIChatModel

__all__ = ["ChatModel", "Message", "OpenAIChatModel"]
