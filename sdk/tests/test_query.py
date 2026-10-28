"""Coverage for ``fastrag.query``: prompt rendering, the engine, and chat persistence.

``PromptTemplate``/``QueryEngine`` are exercised as pure units, with a local
stub :class:`~fastrag.query.llm.base.ChatModel` and a local stub
:class:`~fastrag.retrieval.base.Retriever` standing in for a real model and a
real vector store -- unlike ``tests/test_retrieval.py``'s own end-to-end
coverage, this half never makes a network call and needs no Docker
infrastructure. ``fastrag.query.chat`` (sessions, chats, messages) runs
against the real, isolated ``chat_store`` fixture instead -- the same "few
tests against real infrastructure" design the rest of ``tests/`` follows.
"""

from __future__ import annotations

from collections.abc import AsyncIterator, Sequence

from fastrag.exceptions import ChatStoreError
from fastrag.models import Document
from fastrag.query import (
    DEFAULT_RAG_PROMPT,
    DoneChunk,
    PromptTemplate,
    QueryEngine,
    SourcesChunk,
    TokenChunk,
    group_sources,
)
from fastrag.query.chat import PostgresChatStore, to_llm_history
from fastrag.query.llm import ChatModel, Message
from fastrag.retrieval import Retriever

_DOCUMENTS = [
    Document(
        id="1",
        source_id="s",
        content="AgrospAI is a data space.",
        metadata={"source_uri": "https://example.com/a"},
    ),
    Document(
        id="2",
        source_id="s",
        content="It serves agricultural data.",
        metadata={"source_uri": "https://example.com/b"},
    ),
]


class _StubRetriever(Retriever):
    """Returns a fixed set of documents, ignoring ``query``/``k``."""

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        del query, k
        return list(_DOCUMENTS)


class _StubChatModel(ChatModel):
    """Returns fixed text without ever calling out to a provider."""

    def __init__(self, text: str = "the answer") -> None:
        self.text = text
        self.received: list[Message] | None = None

    @property
    def model_id(self) -> str:
        return "stub"

    async def invoke(self, messages: Sequence[Message]) -> str:
        self.received = list(messages)
        return self.text

    async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
        self.received = list(messages)
        for token in self.text.split(" "):
            yield token + " "


def test_prompt_template_render_numbers_context_and_substitutes_question() -> None:
    """``render`` produces a system message and a user message with numbered context."""
    template = PromptTemplate(
        system="Cite sources as [n].",
        template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
    )

    messages = template.render(context=_DOCUMENTS, question="What is AgrospAI?")

    assert len(messages) == 2
    assert messages[0].role == "system"
    assert messages[0].content == "Cite sources as [n]."
    assert messages[1].role == "user"
    body = messages[1].content
    assert "[1] https://example.com/a\nAgrospAI is a data space." in body
    assert "[2] https://example.com/b\nIt serves agricultural data." in body
    assert "Question: What is AgrospAI?" in body


def test_prompt_template_render_numbers_each_page_once() -> None:
    """A page's chunks share one ``[n]``, in page order, under its title and language.

    Each page of a PDF is cited apart, under its ``#page=N`` link.
    """

    def chunk(url: str | None, index: int, content: str, **extra: str) -> Document:
        page = {"source_uri": url, "title": f"Title {url[-1]}"} if url else {}
        return Document(
            id=f"{url}{index}",
            source_id="s",
            content=content,
            metadata={"chunk_index": index, **page, **extra},
        )

    context = [
        chunk("https://example.com/a", 5, "Fifth."),
        chunk("https://example.com/b", 0, "Other page.", language="en"),
        chunk(None, 0, "Unknown."),
        chunk("https://example.com/a", 1, "First."),
        chunk("https://example.com/a", 2, "Second."),
        chunk(
            "https://example.com/c",
            0,
            "Page two.",
            source_url="https://example.com/c#page=2",
        ),
        chunk(
            "https://example.com/c",
            0,
            "Page four.",
            source_url="https://example.com/c#page=4",
        ),
    ]

    groups = group_sources(context)
    assert [[document.content for document in group] for group in groups] == [
        ["First.", "Second.", "Fifth."],
        ["Other page."],
        ["Unknown."],
        ["Page two."],
        ["Page four."],
    ]

    body = DEFAULT_RAG_PROMPT.render(context=context, question="Q?")[1].content
    assert (
        "[1] https://example.com/a\nTitle: Title a\n"
        "First.\n\nSecond.\n\n…\n\nFifth.\n\n"
        "[2] https://example.com/b\nTitle: Title b\nLanguage: en\nOther page.\n\n"
        "[3] unknown source\nUnknown.\n\n"
        "[4] https://example.com/c#page=2\nTitle: Title c\nPage two.\n\n"
        "[5] https://example.com/c#page=4\nTitle: Title c\nPage four."
    ) in body


async def test_query_engine_ask_returns_stub_text_and_retrieved_sources() -> None:
    """``ask`` returns the stub model's text plus the retriever's documents."""
    llm = _StubChatModel(text="AgrospAI is a data space [1].")
    engine = QueryEngine(retriever=_StubRetriever(), llm=llm)

    answer = await engine.ask("What is AgrospAI?")

    assert answer.text == "AgrospAI is a data space [1]."
    assert answer.sources == _DOCUMENTS
    assert llm.received is not None
    assert llm.received[-1].content.count("Question: What is AgrospAI?") == 1


async def test_query_engine_stream_emits_sources_then_tokens_then_one_done() -> None:
    """``stream`` yields sources first, then tokens, then exactly one ``DoneChunk`` last."""
    llm = _StubChatModel(text="alpha beta")
    engine = QueryEngine(retriever=_StubRetriever(), llm=llm)

    chunks = [chunk async for chunk in engine.stream("What is AgrospAI?")]

    assert isinstance(chunks[0], SourcesChunk)
    assert chunks[0].sources == _DOCUMENTS

    token_chunks = chunks[1:-1]
    assert token_chunks and all(isinstance(chunk, TokenChunk) for chunk in token_chunks)
    assert "".join(chunk.token for chunk in token_chunks) == "alpha beta "

    assert isinstance(chunks[-1], DoneChunk)
    assert sum(1 for chunk in chunks if isinstance(chunk, DoneChunk)) == 1
    assert chunks[-1].answer.text == "alpha beta "
    assert chunks[-1].answer.sources == _DOCUMENTS


# -- fastrag.query.chat: sessions, chats, and messages -----------------------------


async def test_session_round_trip(chat_store: PostgresChatStore) -> None:
    """A created session reads back identically, and an unknown id misses."""
    session = await chat_store.create_session(
        ttl_days=30, ip_address="203.0.113.5", country="ES"
    )

    fetched = await chat_store.get_session(session.id)
    assert fetched == session

    assert await chat_store.get_session("unknown-session-id") is None

    # Throttled: a bump smaller than min_bump_days is a no-op.
    unchanged = await chat_store.touch_session(session.id, ttl_days=30, min_bump_days=1)
    assert unchanged is not None
    assert unchanged.expires_at == session.expires_at

    # A bump that clears the threshold is written.
    bumped = await chat_store.touch_session(session.id, ttl_days=90, min_bump_days=1)
    assert bumped is not None
    assert bumped.expires_at > session.expires_at

    assert (
        await chat_store.touch_session(
            "unknown-session-id", ttl_days=30, min_bump_days=1
        )
        is None
    )


async def test_chats_listed_newest_first_with_cursor_paging_honoured(
    chat_store: PostgresChatStore,
) -> None:
    """``list_chats`` returns newest-first, and ``before`` walks older pages."""
    session = await chat_store.create_session(ttl_days=30)

    chat_a = await chat_store.create_chat(session_id=session.id, ttl_days=30)
    chat_b = await chat_store.create_chat(session_id=session.id, ttl_days=30)

    full_page = await chat_store.list_chats(session.id)
    assert [chat.id for chat in full_page.items] == [chat_b.id, chat_a.id]
    assert full_page.has_more is False

    first_page = await chat_store.list_chats(session.id, limit=1)
    assert [chat.id for chat in first_page.items] == [chat_b.id]
    assert first_page.has_more is True

    second_page = await chat_store.list_chats(
        session.id, before=first_page.items[-1].id, limit=1
    )
    assert [chat.id for chat in second_page.items] == [chat_a.id]
    assert second_page.has_more is False

    # Scoped to its owning session: a different session sees no chats here.
    other_session = await chat_store.create_session(ttl_days=30)
    assert (await chat_store.list_chats(other_session.id)).items == []


async def test_messages_read_back_in_order(chat_store: PostgresChatStore) -> None:
    """Appended messages read back in a well-defined order, chat-model-ready via ``to_llm_history``."""
    session = await chat_store.create_session(ttl_days=30)
    chat = await chat_store.create_chat(session_id=session.id, ttl_days=30)

    user_1 = await chat_store.append_message(chat.id, role="user", content="hello")
    assistant_1 = await chat_store.append_message(
        chat.id, role="assistant", content="hi there", finish_reason="stop"
    )
    user_2 = await chat_store.append_message(
        chat.id, role="user", content="how are you"
    )
    assistant_2 = await chat_store.append_message(
        chat.id, role="assistant", content="doing fine", finish_reason="stop"
    )

    page = await chat_store.list_messages(chat.id)
    assert [message.id for message in page.items] == [
        assistant_2.id,
        user_2.id,
        assistant_1.id,
        user_1.id,
    ]
    assert page.has_more is False

    history = to_llm_history(page.items)
    assert [(message.role, message.content) for message in history] == [
        ("user", "hello"),
        ("assistant", "hi there"),
        ("user", "how are you"),
        ("assistant", "doing fine"),
    ]


async def test_update_last_assistant_message(chat_store: PostgresChatStore) -> None:
    """Overwrites the most recent ``assistant`` row in place, for ``/retry``/``/edit``."""
    session = await chat_store.create_session(ttl_days=30)
    chat = await chat_store.create_chat(session_id=session.id, ttl_days=30)

    # No assistant message yet: nothing to update.
    assert (
        await chat_store.update_last_assistant_message(
            chat.id, content="anything", finish_reason="stop"
        )
        is None
    )

    await chat_store.append_message(chat.id, role="user", content="hello")
    first_answer = await chat_store.append_message(
        chat.id, role="assistant", content="hi there", finish_reason="stop"
    )

    updated = await chat_store.update_last_assistant_message(
        chat.id,
        content="hi there, retried",
        sources=[{"url": "https://example.com", "title": "Example"}],
        finish_reason="stop",
    )
    assert updated is not None
    # Same row, in place -- not a new id.
    assert updated.id == first_answer.id
    assert updated.content == "hi there, retried"
    assert updated.sources == [{"url": "https://example.com", "title": "Example"}]

    page = await chat_store.list_messages(chat.id)
    assert len(page.items) == 2
    assert page.items[0].id == first_answer.id
    assert page.items[0].content == "hi there, retried"

    # A later, second assistant turn is the one that gets overwritten next.
    await chat_store.append_message(chat.id, role="user", content="and you?")
    second_answer = await chat_store.append_message(
        chat.id, role="assistant", content="doing fine", finish_reason="stop"
    )

    updated_again = await chat_store.update_last_assistant_message(
        chat.id, content="doing great actually", finish_reason="stop"
    )
    assert updated_again is not None
    assert updated_again.id == second_answer.id
    assert updated_again.content == "doing great actually"

    # The first answer is untouched by the second update.
    refreshed_page = await chat_store.list_messages(chat.id)
    by_id = {message.id: message for message in refreshed_page.items}
    assert by_id[first_answer.id].content == "hi there, retried"


async def test_delete_chat_and_check_constraint_violations(
    chat_store: PostgresChatStore,
) -> None:
    """Deleting a chat removes it and every message inside it, ownership is
    enforced, and a ``CHECK`` constraint violation surfaces as a
    ``ChatStoreError``, never a raw driver error.
    """
    session = await chat_store.create_session(ttl_days=30)
    chat = await chat_store.create_chat(session_id=session.id, ttl_days=30)
    await chat_store.append_message(chat.id, role="user", content="hello")
    await chat_store.append_message(
        chat.id, role="assistant", content="hi", finish_reason="stop"
    )

    deleted = await chat_store.delete_chat(chat.id, session_id=session.id)
    assert deleted is True

    assert await chat_store.get_chat(chat.id, session_id=session.id) is None
    assert (await chat_store.list_messages(chat.id)).items == []

    # Idempotent-ish: deleting again reports nothing to delete, not an error.
    assert await chat_store.delete_chat(chat.id, session_id=session.id) is False

    # Ownership is enforced: a session that doesn't own the chat can't delete it.
    other_chat = await chat_store.create_chat(session_id=session.id, ttl_days=30)
    other_session = await chat_store.create_session(ttl_days=30)
    assert (
        await chat_store.delete_chat(other_chat.id, session_id=other_session.id)
        is False
    )
    assert await chat_store.get_chat(other_chat.id, session_id=session.id) is not None

    # `finish_reason_required_for_assistant`: an assistant row needs one.
    check_chat = await chat_store.create_chat(session_id=session.id, ttl_days=30)
    try:
        await chat_store.append_message(
            check_chat.id, role="assistant", content="no reason given"
        )
    except ChatStoreError:
        pass
    else:
        raise AssertionError("expected a ChatStoreError on the CHECK violation")
