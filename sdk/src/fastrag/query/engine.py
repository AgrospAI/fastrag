"""The ``QueryEngine``: retrieval plus generation, one question at a time."""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass

from fastrag.models import Document
from fastrag.query.llm.base import ChatModel, Message
from fastrag.query.prompts.templates import DEFAULT_RAG_PROMPT, PromptTemplate
from fastrag.retrieval.base import Retriever


class QueryEngine:
    """Answers one question by retrieving, then generating.

    Takes a :class:`~fastrag.retrieval.base.Retriever`, not a
    :class:`~fastrag.vector_store.base.VectorStore` — so every retrieval
    strategy (:class:`~fastrag.retrieval.vector.VectorRetriever`,
    :class:`~fastrag.retrieval.reranking_retriever.RerankingRetriever`) is
    available to the engine without it knowing any of them exist, and there
    is exactly one place retrieval is configured.

    .. code-block:: python

        import os

        from fastrag.query import QueryEngine
        from fastrag.query.llm import OpenAIChatModel

        engine = QueryEngine(
            retriever=retriever,
            llm=OpenAIChatModel(
                model=os.environ["FASTRAG_CHAT_MODEL"],
                api_key=os.environ["FASTRAG_CHAT_API_KEY"],
                base_url=os.environ["FASTRAG_CHAT_BASE_URL"],
            ),
        )

        answer = await engine.ask("What is a data space?")
        print(answer.text, [source.id for source in answer.sources])

    Args:
        retriever: Where the context for each question comes from.
        llm: Generates the answer from the rendered prompt.
        prompt: Turns a question and its retrieved context into messages.
    """

    def __init__(
        self,
        *,
        retriever: Retriever,
        llm: ChatModel,
        prompt: PromptTemplate = DEFAULT_RAG_PROMPT,
    ) -> None:
        """Store the collaborators; no I/O happens until :meth:`ask`/:meth:`stream`."""
        self.retriever = retriever
        self.llm = llm
        self.prompt = prompt

    async def ask(self, question: str, history: Sequence[Message] = ()) -> Answer:
        """Retrieve, then generate the whole answer at once.

        ``history`` is inserted into the messages sent to ``llm`` as prior
        turns, so a follow-up question can resolve against earlier context.
        It is sent to the model only — retrieval embeds ``question`` alone.
        """
        sources = await self.retriever.retrieve(question)
        messages = _with_history(
            self.prompt.render(context=sources, question=question), history
        )
        text = await self.llm.invoke(messages)
        return Answer(text=text, sources=sources)

    async def stream(
        self, question: str, history: Sequence[Message] = ()
    ) -> AsyncIterator[StreamChunk]:
        """Retrieve, then yield the answer as it generates.

        Sources first, then one :class:`TokenChunk` per delta, then a
        :class:`DoneChunk`.

        ``history`` is inserted into the messages sent to ``llm`` as prior
        turns, so a follow-up question can resolve against earlier context.
        It is sent to the model only — retrieval embeds ``question`` alone.
        """
        sources = await self.retriever.retrieve(question)
        yield SourcesChunk(sources=sources)

        messages = _with_history(
            self.prompt.render(context=sources, question=question), history
        )
        parts: list[str] = []
        async for token in self.llm.stream(messages):
            parts.append(token)
            yield TokenChunk(token=token)

        yield DoneChunk(answer=Answer(text="".join(parts), sources=sources))


def _with_history(rendered: list[Message], history: Sequence[Message]) -> list[Message]:
    """Insert ``history`` between the rendered system message, if any, and the turn.

    Only the newest turn carries retrieved context: re-rendering it into
    every past turn would multiply the prompt by the chat length and put
    stale documents in front of the model. ``rendered`` is
    ``PromptTemplate.render``'s own output — a system message (if the
    template has one) followed by exactly one user message.
    """
    if rendered and rendered[0].role == "system":
        return [rendered[0], *history, *rendered[1:]]
    return [*history, *rendered]


@dataclass(frozen=True, slots=True)
class Answer:
    """One generated answer, with its sources."""

    text: str
    sources: list[Document]


@dataclass(frozen=True, slots=True)
class SourcesChunk:
    """The retrieved context, sent once before the first :class:`TokenChunk`."""

    sources: list[Document]


@dataclass(frozen=True, slots=True)
class TokenChunk:
    """One text delta from the chat model."""

    token: str


@dataclass(frozen=True, slots=True)
class DoneChunk:
    """The stream is finished; carries the assembled :class:`Answer`."""

    answer: Answer


StreamChunk = SourcesChunk | TokenChunk | DoneChunk
