"""Serving-layer tests: ``fastrag.serve`` against a stubbed ``QueryEngine``.

These exercise the FastAPI app end to end over a TestClient/ASGITransport,
with a stub ``QueryEngine`` (no model provider ever called) and the real
``PostgresChatStore`` on an isolated schema (the ``chat_store`` fixture). The
whole point of the design is that the app is a thin adapter: swapping the
real engine for a stub must change nothing about HTTP behaviour, SSE framing,
auth, or the request guards. ``cleanup_expired`` is exercised separately,
against its own isolated ``PostgresChatStore`` and a raw engine scoped to
that same schema (see its own docstring: unqualified table names resolve on
the connection's search path).
"""

from __future__ import annotations

import asyncio
import importlib.util
import json
from collections.abc import AsyncIterator, Sequence
from pathlib import Path
from uuid import uuid4

import asyncpg
import pytest

#: Imported, not re-declared, so this file follows ``conftest.py``'s
#: ``FASTRAG_DATABASE_URL`` resolution: ``postgres:5432`` under ``task e2e``,
#: ``localhost:5433`` under ``task test``.
from conftest import DATABASE_URL
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import create_async_engine

from fastrag.models import Document
from fastrag.query.chat import PostgresChatStore
from fastrag.query.engine import (
    Answer,
    DoneChunk,
    QueryEngine,
    SourcesChunk,
    StreamChunk,
    TokenChunk,
)
from fastrag.query.llm.base import Message
from fastrag.serve import ServeConfig, cleanup_expired, create_app
from fastrag.serve._locking import LockKeepAlive
from fastrag.serve._sse import with_keepalive
import fastrag.serve.geolocalization
from fastrag.serve.geolocalization import CountryResolver, get_country_resolver

#: The same database, spelled for SQLAlchemy's asyncpg dialect --
#: ``cleanup_expired`` takes an ``AsyncEngine``, not a raw DSN.
ASYNCPG_DATABASE_URL = DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://", 1)


class _StubEngine(QueryEngine):
    """A ``QueryEngine`` that never calls a model.

    Retrieval is stubbed by overriding :meth:`retrieve`; ``stream`` is
    stubbed here to yield a fixed answer on every call, so no ``ChatModel``
    is ever constructed. ``ask`` is left on the base (unused by the API,
    which only streams). Both the question and the ``history`` argument of
    every ``stream`` call are recorded, so a test can assert what actually
    reaches the model.
    """

    def __init__(self) -> None:
        self.seen_questions: list[str] = []
        self.seen_history: list[list[Message]] = []

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        """Return one fixed grounded document, recording the question."""
        self.seen_questions.append(query)
        return [
            Document(
                id="src-1",
                source_id="src-1",
                content="AgrospAI is a data space for agriculture.",
                metadata={"source_uri": "https://agrospai.udl.cat/en/about/"},
            )
        ]

    async def stream(
        self, question: str, history: Sequence = ()
    ) -> AsyncIterator[StreamChunk]:
        """Yield a grounded answer: sources, tokens, then one done."""
        self.seen_history.append(list(history))
        sources = await self.retrieve(question)
        yield SourcesChunk(sources=sources)
        for token in ["Ground", "ed", " answer"]:
            yield TokenChunk(token=token)
        yield DoneChunk(answer=Answer(text="Grounded answer", sources=sources))


def _app(
    chat_store: PostgresChatStore,
    *,
    config: ServeConfig | None = None,
    engine: _StubEngine | None = None,
):
    if engine is None:
        engine = _StubEngine()
    if config is None:
        # cookie_secure=False so the test client's cookie jar will echo the
        # session cookie back over plain http://test.
        config = ServeConfig(allowed_origins=[], cookie_secure=False)
    return create_app(engine=engine, chat_store=chat_store, config=config)


def _parse_sse(text: str) -> list[tuple[str, dict]]:
    """Parse a raw SSE body into ``(event, data)`` pairs."""
    events: list[tuple[str, dict]] = []
    event = ""
    data_lines: list[str] = []
    for line in text.splitlines():
        if line == "":
            if event:
                events.append(
                    (event, json.loads("".join(data_lines)) if data_lines else {})
                )
            event = ""
            data_lines = []
        elif line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])
    return events


async def test_healthz(chat_store: PostgresChatStore) -> None:
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


async def test_create_chat_streams_answer(
    chat_store: PostgresChatStore,
) -> None:
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chats", json={"content": "What is AgrospAI?"})
    assert response.status_code == 200

    events = _parse_sse(response.text)
    names = [name for name, _ in events]
    assert "message_start" in names
    assert "sources" in names
    assert "delta" in names
    assert "message_end" in names
    # message_start first, message_end last.
    assert names.index("message_start") < names.index("message_end")

    sources = dict(events)["sources"]["sources"]
    assert sources[0]["url"] == "https://agrospai.udl.cat/en/about/"

    end = dict(events)["message_end"]
    assert end["finish_reason"] == "stop"
    assert "error" not in end


def test_sources_payload_numbers_each_page_once() -> None:
    """``[n]`` numbers pages, each PDF page apart, as the prompt does.

    ``positions`` holds that ``n``, and a PDF page's ``url`` is its
    ``#page=N`` link.
    """
    from fastrag.serve._generation import MAX_SOURCES_BYTES, _sources_payload

    def chunk(url: str | None, **metadata: str) -> Document:
        return Document(
            id=str(uuid4()),
            source_id="",
            content="...",
            metadata={"source_uri": url, **metadata} if url else metadata,
        )

    about = "https://agrospai.udl.cat/en/about/"
    pdf = "https://agrospai.udl.cat/en/files/AgrospAI.en.pdf"
    payload = _sources_payload(
        [
            chunk(
                about,
                title="About",
                site_name="AgrospAI",
                favicon_url="https://agrospai.udl.cat/en/img/favicon.ico",
            ),
            chunk(pdf, source_url=f"{pdf}#page=3"),
            chunk(None),
            chunk(about, title="About"),
            chunk(pdf, source_url=f"{pdf}#page=7"),
            chunk(pdf, source_url=f"{pdf}#page=3"),
        ]
    )
    assert payload is not None
    assert [source.model_dump(exclude_none=True) for source in payload] == [
        {
            "url": about,
            "title": "About",
            "site_name": "AgrospAI",
            "favicon_url": "https://agrospai.udl.cat/en/img/favicon.ico",
            "positions": [1],
            "fragments": [],
        },
        {
            "url": f"{pdf}#page=3",
            "title": f"{pdf}#page=3",
            "positions": [2],
            "fragments": [],
        },
        {
            "url": f"{pdf}#page=7",
            "title": f"{pdf}#page=7",
            "positions": [4],
            "fragments": [],
        },
    ]

    # The persisted JSON stays within the table's octet_length CHECK.
    long_title = "x" * 2000
    oversized = _sources_payload(
        [chunk(f"https://example.test/{i}", title=long_title) for i in range(10)]
    )
    assert oversized
    stored = [source.model_dump(exclude_none=True) for source in oversized]
    assert len(json.dumps(stored, ensure_ascii=False).encode()) <= MAX_SOURCES_BYTES


def test_text_fragments_locate_passages_as_the_page_renders_them() -> None:
    """Terms come from rendered text; one section's chunks form one passage."""
    from fastrag.serve._fragments import text_fragments
    from fastrag.serve._schemas import TextFragment

    def chunk(index: int, content: str, section: str, **metadata: str) -> Document:
        return Document(
            id=str(uuid4()),
            source_id="",
            content=content,
            metadata={
                "content_type": "text/html",
                "chunk_index": index,
                "heading_path": [section],
                **metadata,
            },
        )

    page = [
        chunk(
            3,
            "## Paso 1​\n\n"
            "Para completar la **obtención** de las credenciales, todos deben "
            "aportar datos.\n\n\nnota\n\n\n"
            "Puedes escribir a <agrospai@udl.cat> para procesar tu solicitud. :::",
            "Paso 1",
        ),
        chunk(
            4,
            "Image description: A screenshot of the form.\n"
            "It continues here.\n\n"
            "- Según el [Marco de Confianza](https://gaia-x.eu) de Gaia-X\n\n"
            "| Campo | Valor |\n| --- | --- |",
            "Paso 1",
        ),
        chunk(9, "Otra sección empieza aquí.", "Paso 2"),
    ]

    assert text_fragments(page) == [
        TextFragment(
            start="Para completar la obtención de",
            end="Marco de Confianza de Gaia-X",
        ),
        TextFragment(start="Otra sección empieza aquí."),
    ]

    # No spaces between words: terms end after a clause instead.
    chinese = chunk(
        0,
        "为了完成凭证的获取，所有参与者都必须提供其组织的以下详细信息。\n\n"
        "任何法律实体所需的信息，与Gaia-X信任框架对[参与者](https://gaia-x.eu)"
        "描述的要求一致，包括：",
        "S",
    )
    assert text_fragments([chinese]) == [
        TextFragment(
            start="为了完成凭证的获取",
            end="与Gaia-X信任框架对参与者描述的要求一致，包括",
        )
    ]

    pdf = chunk(0, "Not searchable.", "S", content_type="application/pdf")
    assert text_fragments([pdf]) == []


async def test_401_without_session(chat_store: PostgresChatStore) -> None:
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/v1/chats")
    assert response.status_code == 401
    assert response.json()["error"]["code"] == "UNAUTHORIZED"


async def test_list_and_read_chats(chat_store: PostgresChatStore) -> None:
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # POST /v1/chats mints a session and returns its cookie; httpx stores
        # it in the client jar and echoes it back on the next request.
        created = await client.post("/v1/chats", json={"content": "hi"})
        assert created.status_code == 200
        assert created.cookies.get("session_id") is not None

        listed = await client.get("/v1/chats")
        assert listed.status_code == 200
        body = listed.json()
        assert body["chats"] and len(body["chats"]) >= 1

        chat_id = body["chats"][0]["id"]
        read = await client.get(f"/v1/chats/{chat_id}")
        assert read.status_code == 200
        messages = read.json()["messages"]
        # The stub streamed an assistant message, so there are real roles.
        assert any(m["role"] == "user" for m in messages)
        assert any(m["role"] == "assistant" and m["content"] for m in messages)


async def test_request_guard_413_and_415(chat_store: PostgresChatStore) -> None:
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # 415: non-JSON Content-Type on a POST.
        bad_type = await client.post(
            "/v1/chats", content="content", headers={"Content-Type": "text/plain"}
        )
        assert bad_type.status_code == 415
        assert bad_type.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"

        # 413: body over max_request_bytes (default 32768).
        big = '{"content": "' + "x" * 40000 + '"}'
        too_big = await client.post(
            "/v1/chats",
            content=big,
            headers={"Content-Type": "application/json"},
        )
        assert too_big.status_code == 413
        assert too_big.json()["error"]["code"] == "PAYLOAD_TOO_LARGE"


async def test_max_content_chars_enforced(chat_store: PostgresChatStore) -> None:
    """A ``content`` over ``max_content_chars`` is ``422``; at the limit, ``200``.

    Exercises all three routes that carry a ``content`` field:
    ``POST /v1/chats``, ``POST /v1/chats/{id}/messages``, and
    ``POST /v1/chats/{id}/messages/{mid}``.
    """
    config = ServeConfig(allowed_origins=[], cookie_secure=False, max_content_chars=10)
    app = _app(chat_store, config=config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        # Exactly at the limit: accepted.
        at_limit = await client.post("/v1/chats", json={"content": "x" * 10})
        assert at_limit.status_code == 200
        chat_id = dict(_parse_sse(at_limit.text))["message_start"]["chat_id"]

        # One over the limit: rejected with the standard envelope.
        too_long = await client.post("/v1/chats", json={"content": "x" * 11})
        assert too_long.status_code == 422
        assert too_long.json()["error"]["code"] == "INVALID_REQUEST"

        # Same check on POST /v1/chats/{id}/messages.
        add_too_long = await client.post(
            f"/v1/chats/{chat_id}/messages", json={"content": "x" * 11}
        )
        assert add_too_long.status_code == 422
        assert add_too_long.json()["error"]["code"] == "INVALID_REQUEST"

        # And on POST /v1/chats/{id}/messages/{mid} (editing a user message).
        messages = (await client.get(f"/v1/chats/{chat_id}")).json()["messages"]
        user_message_id = next(m["id"] for m in messages if m["role"] == "user")
        regen_too_long = await client.post(
            f"/v1/chats/{chat_id}/messages/{user_message_id}",
            json={"content": "x" * 11},
        )
        assert regen_too_long.status_code == 422
        assert regen_too_long.json()["error"]["code"] == "INVALID_REQUEST"


async def test_followup_receives_prior_turns_and_drops_a_failed_assistant_turn(
    chat_store: PostgresChatStore,
) -> None:
    """A follow-up question reaches the model with the prior turns present --
    regression for the reported defect where the serving app "forgot"
    earlier turns -- and a turn whose ``content is None`` (an assistant turn
    that errored before generating anything) is dropped from that history
    rather than replayed to the model.
    """
    engine = _StubEngine()
    app = _app(chat_store, engine=engine)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post("/v1/chats", json={"content": "My name is Jordi"})
        assert first.status_code == 200
        chat_id = dict(_parse_sse(first.text))["message_start"]["chat_id"]

        second = await client.post(
            f"/v1/chats/{chat_id}/messages", json={"content": "What is my name?"}
        )
        assert second.status_code == 200

        # First turn: no prior history. Second turn: the earlier
        # user/assistant pair, chronological, current question last.
        assert engine.seen_history[0] == []
        assert engine.seen_history[1] == [
            Message(role="user", content="My name is Jordi"),
            Message(role="assistant", content="Grounded answer"),
        ]
        # Retrieval still embeds only the current question -- history stays
        # out of the retriever.
        assert engine.seen_questions == ["My name is Jordi", "What is my name?"]

        # Append a failed assistant turn directly (content=None), then ask a
        # third question.
        await chat_store.append_message(
            chat_id,
            role="assistant",
            content=None,
            finish_reason="error",
            error_code="MODEL_ERROR",
            error_message="boom",
        )
        third = await client.post(
            f"/v1/chats/{chat_id}/messages", json={"content": "Are you sure?"}
        )
        assert third.status_code == 200

    # The failed turn (content=None) is dropped from history; the first
    # turn's user/assistant pair survives, unaffected by the failed one in
    # between.
    assert engine.seen_history[2] == [
        Message(role="user", content="My name is Jordi"),
        Message(role="assistant", content="Grounded answer"),
        Message(role="user", content="What is my name?"),
        Message(role="assistant", content="Grounded answer"),
    ]


async def test_followup_history_ordering_and_system(
    chat_store: PostgresChatStore,
) -> None:
    """The system message stays first; prior turns follow; the current turn last.

    Exercises the full ``QueryEngine.stream`` path (not the stub's override)
    so ``_with_history`` and ``PromptTemplate.render`` are both in play: the
    message list the model receives must be ``[system, ...prior turns,
    current rendered turn]``.
    """
    from fastrag.query.engine import QueryEngine
    from fastrag.query.llm.base import ChatModel
    from fastrag.query.prompts.templates import DEFAULT_RAG_PROMPT
    from fastrag.retrieval.base import Retriever

    class _RecordingLLM(ChatModel):
        def __init__(self) -> None:
            self.calls: list[list[Message]] = []

        @property
        def model_id(self) -> str:
            return "recording"

        async def invoke(self, messages: Sequence[Message]) -> str:
            self.calls.append(list(messages))
            return "ok"

        async def stream(self, messages: Sequence[Message]) -> AsyncIterator[str]:
            self.calls.append(list(messages))
            yield "ok"

    class _RecordingRetriever(Retriever):
        def __init__(self) -> None:
            self.queries: list[str] = []

        async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
            self.queries.append(query)
            return [
                Document(
                    id="src-1",
                    source_id="src-1",
                    content="AgrospAI is a data space for agriculture.",
                    metadata={"source_uri": "https://agrospai.udl.cat/en/about/"},
                )
            ]

    llm = _RecordingLLM()
    retriever = _RecordingRetriever()
    engine = QueryEngine(retriever=retriever, llm=llm, prompt=DEFAULT_RAG_PROMPT)
    app = _app(chat_store, engine=engine)  # type: ignore[arg-type]
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = await client.post("/v1/chats", json={"content": "My name is Jordi"})
        chat_id = dict(_parse_sse(first.text))["message_start"]["chat_id"]
        await client.post(
            f"/v1/chats/{chat_id}/messages", json={"content": "What is my name?"}
        )

    # Turn 1: system + rendered user turn only.
    assert [m.role for m in llm.calls[0]] == ["system", "user"]
    # Turn 2: system first, prior turns, current rendered turn last.
    roles = [m.role for m in llm.calls[1]]
    assert roles == ["system", "user", "assistant", "user"]
    assert llm.calls[1][0].role == "system"
    assert llm.calls[1][1] == Message(role="user", content="My name is Jordi")
    assert llm.calls[1][2] == Message(role="assistant", content="ok")
    assert llm.calls[1][3].role == "user"
    assert "What is my name?" in llm.calls[1][3].content
    # Retrieval embeds only the current question.
    assert retriever.queries == ["My name is Jordi", "What is my name?"]


async def test_metrics_endpoint_responds_and_parses(
    chat_store: PostgresChatStore,
) -> None:
    """``GET /metrics`` is ``200 text/plain; version=0.0.4`` and parses."""
    from prometheus_client.parser import text_string_to_metric_families

    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/plain")
    assert "version=0.0.4" in response.headers["content-type"]
    # Parses as valid Prometheus text exposition.
    list(text_string_to_metric_families(response.text))


async def test_metrics_series_appear_after_exercising_the_relevant_paths(
    chat_store: PostgresChatStore,
) -> None:
    """Each of the seven series is present once its path has been exercised."""
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chats", json={"content": "What is AgrospAI?"})
        assert response.status_code == 200

        metrics_text = (await client.get("/metrics")).text

    for series in (
        "http_requests_total",
        "http_requests_in_flight",
        "http_request_errors_total",
        "llm_time_to_first_token_seconds",
        "llm_time_to_last_token_seconds",
        "llm_question_length_chars",
        "llm_answer_length_chars",
    ):
        assert series in metrics_text, f"{series} missing from /metrics output"


async def test_metrics_excludes_itself_from_http_requests_total(
    chat_store: PostgresChatStore,
) -> None:
    """Scraping ``/metrics`` must not inflate ``http_requests_total``."""
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        first = (await client.get("/metrics")).text
        before = _counter_total(first, "http_requests_total")

        for _ in range(3):
            await client.get("/metrics")

        after_text = (await client.get("/metrics")).text
        after = _counter_total(after_text, "http_requests_total")

    assert after == before


def _counter_total(metrics_text: str, name: str) -> float:
    """Sum every ``<name>{...} <value>`` sample line in ``metrics_text``."""
    total = 0.0
    for line in metrics_text.splitlines():
        if line.startswith(f"{name}{{") or line.startswith(f"{name} "):
            total += float(line.rsplit(" ", 1)[-1])
    return total


async def test_metrics_guarded_by_token(chat_store: PostgresChatStore) -> None:
    """A configured ``metrics_token`` rejects a missing/wrong bearer token."""
    config = ServeConfig(
        allowed_origins=[], cookie_secure=False, metrics_token="secret-token"
    )
    app = _app(chat_store, config=config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        missing = await client.get("/metrics")
        assert missing.status_code == 401
        assert missing.text == ""

        wrong = await client.get("/metrics", headers={"Authorization": "Bearer nope"})
        assert wrong.status_code == 401
        assert wrong.text == ""

        right = await client.get(
            "/metrics", headers={"Authorization": "Bearer secret-token"}
        )
        assert right.status_code == 200
        assert "http_requests_total" in right.text


async def test_metrics_disabled_returns_404(chat_store: PostgresChatStore) -> None:
    """``metrics_enabled=False`` withholds the route entirely."""
    config = ServeConfig(allowed_origins=[], cookie_secure=False, metrics_enabled=False)
    app = _app(chat_store, config=config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.get("/metrics")
    assert response.status_code == 404


async def test_ttft_strictly_smaller_than_ttlt_for_a_multi_token_stream(
    chat_store: PostgresChatStore,
) -> None:
    """A generation that streams over time records TTFT < TTLT.

    Uses an engine whose ``stream`` sleeps between tokens, so time-to-first-
    token is a real observation of the first token reaching the client, not
    an approximation from the whole request's duration.
    """

    class _SlowStreamEngine(_StubEngine):
        async def stream(self, question: str, history: Sequence = ()):
            self.seen_history.append(list(history))
            sources = await self.retrieve(question)
            yield SourcesChunk(sources=sources)
            for token in ["Ground", "ed", " answer"]:
                await asyncio.sleep(0.05)
                yield TokenChunk(token=token)
            yield DoneChunk(answer=Answer(text="Grounded answer", sources=sources))

    engine = _SlowStreamEngine()
    app = _app(chat_store, engine=engine)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chats", json={"content": "Hi"})
        assert response.status_code == 200

        metrics_text = (await client.get("/metrics")).text

    ttft_sum = _histogram_sum(metrics_text, "llm_time_to_first_token_seconds")
    ttlt_sum = _histogram_sum(metrics_text, "llm_time_to_last_token_seconds")
    assert ttft_sum is not None and ttlt_sum is not None
    assert ttft_sum < ttlt_sum


async def test_generation_timeout_fires_after_streaming_has_started(
    chat_store: PostgresChatStore,
) -> None:
    """A generation that stalls after its first token still lands as ``TIMEOUT``."""

    class _StallingEngine(_StubEngine):
        async def stream(self, question: str, history: Sequence = ()):
            sources = await self.retrieve(question)
            yield SourcesChunk(sources=sources)
            yield TokenChunk(token="Ground")
            await asyncio.Event().wait()

    config = ServeConfig(
        allowed_origins=[], cookie_secure=False, generation_timeout_s=0.2
    )
    app = _app(chat_store, config=config, engine=_StallingEngine())
    async with (
        asyncio.timeout(10),
        AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client,
    ):
        response = await client.post("/v1/chats", json={"content": "Hi"})

    end = dict(_parse_sse(response.text))["message_end"]
    assert end["finish_reason"] == "error"
    assert end["error"]["code"] == "TIMEOUT"


async def test_with_keepalive_lets_a_timeout_inside_the_source_fire() -> None:
    """A timeout the source entered fires even after it has yielded a frame."""

    async def source() -> AsyncIterator[bytes]:
        try:
            async with asyncio.timeout(0.1):
                yield b"first"
                await asyncio.Event().wait()
        except TimeoutError:
            yield b"timed out"

    async with asyncio.timeout(5):
        frames = [frame async for frame in with_keepalive(source(), interval=0.02)]

    assert [frame for frame in frames if not frame.startswith(b":")] == [
        b"first",
        b"timed out",
    ]


async def test_lock_keepalive_pings_until_stopped() -> None:
    """A held lock's connection is pinged, and left alone once stopped."""

    class _Connection:
        def __init__(self) -> None:
            self.statements: list[str] = []

        async def execute(self, statement, parameters=None) -> None:
            self.statements.append(str(statement))

    connection = _Connection()
    keepalive = LockKeepAlive(connection, interval=0.02)  # type: ignore[arg-type]
    await asyncio.sleep(0.1)
    await keepalive.stop()
    pings = len(connection.statements)
    await asyncio.sleep(0.05)

    assert pings >= 2
    assert connection.statements == ["SELECT 1"] * pings


def _histogram_sum(metrics_text: str, name: str) -> float | None:
    """Return a histogram's ``_sum`` value, or ``None`` if absent."""
    prefix = f"{name}_sum"
    for line in metrics_text.splitlines():
        if line.startswith(prefix):
            return float(line.rsplit(" ", 1)[-1])
    return None


async def test_generation_rate_limit_rejects_the_nth_plus_one_request(
    chat_store: PostgresChatStore,
) -> None:
    """The generation limit's ``429`` carries the ``RATE_LIMIT`` code."""
    config = ServeConfig(
        allowed_origins=[],
        cookie_secure=False,
        rate_limiting_enabled=True,
        rate_limit_generation="1/minute",
    )
    app = _app(chat_store, config=config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        allowed = await client.post("/v1/chats", json={"content": "Hi"})
        assert allowed.status_code == 200

        rejected = await client.post("/v1/chats", json={"content": "Hi again"})

    assert rejected.status_code == 429
    assert rejected.json() == {
        "error": {
            "code": "RATE_LIMIT",
            "message": "Rate limit exceeded (1/minute)",
        }
    }


async def test_rate_limiting_disabled_by_default(chat_store: PostgresChatStore) -> None:
    """A config built with only ``allowed_origins=`` never rejects with ``429``.

    The CLI's own generation default (``10/minute``) would already have
    rejected an 11th request in a minute if it were enforced — it isn't,
    because ``rate_limiting_enabled`` defaults to ``False``.
    """
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        for _ in range(15):
            response = await client.post("/v1/chats", json={"content": "Hi"})
            assert response.status_code == 200


async def test_x_forwarded_for_ignored_unless_trusted(
    chat_store: PostgresChatStore,
) -> None:
    """An untrusted ``X-Forwarded-For`` never overrides the peer address."""
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chats",
            json={"content": "Hi"},
            headers={"X-Forwarded-For": "9.9.9.9"},
        )
        assert response.status_code == 200
        session_id = client.cookies.get("session_id")

    session = await chat_store.get_session(session_id)
    assert session is not None
    assert session.ip_address == "127.0.0.1"


async def test_x_forwarded_for_honoured_when_trusted(
    chat_store: PostgresChatStore,
) -> None:
    """A trusted ``X-Forwarded-For`` overrides the peer address, leftmost entry."""
    config = ServeConfig(
        allowed_origins=[], cookie_secure=False, trust_proxy_headers=True
    )
    app = _app(chat_store, config=config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post(
            "/v1/chats",
            json={"content": "Hi"},
            headers={"X-Forwarded-For": "9.9.9.9, 10.0.0.1"},
        )
        assert response.status_code == 200
        session_id = client.cookies.get("session_id")

    session = await chat_store.get_session(session_id)
    assert session is not None
    assert session.ip_address == "9.9.9.9"


async def test_rejected_requests_total_increments_on_429(
    chat_store: PostgresChatStore,
) -> None:
    """``rejected_requests_total`` counts a ``429``; ``requests_per_ip_total`` appears."""
    config = ServeConfig(
        allowed_origins=[],
        cookie_secure=False,
        rate_limiting_enabled=True,
        rate_limit_generation="1/minute",
    )
    app = _app(chat_store, config=config)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        await client.post("/v1/chats", json={"content": "Hi"})
        rejected = await client.post("/v1/chats", json={"content": "Hi again"})
        assert rejected.status_code == 429

        metrics_text = (await client.get("/metrics")).text

    assert _counter_total(metrics_text, "rejected_requests_total") == 1
    assert "requests_per_ip_total" in metrics_text


async def test_session_row_carries_resolved_ip_and_null_country_by_default(
    chat_store: PostgresChatStore,
) -> None:
    """A minted session persists its resolved IP; ``country`` stays ``NULL``
    with no ``geoip_database_path`` configured (the default)."""
    app = _app(chat_store)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chats", json={"content": "Hi"})
        assert response.status_code == 200
        session_id = client.cookies.get("session_id")

    session = await chat_store.get_session(session_id)
    assert session is not None
    assert session.ip_address == "127.0.0.1"
    assert session.country is None


def test_country_resolver_returns_none_with_no_database_configured() -> None:
    resolver = CountryResolver(None)
    assert resolver.lookup("8.8.8.8") is None


def test_country_resolver_returns_none_for_a_missing_database_file(tmp_path) -> None:
    resolver = CountryResolver(tmp_path / "does-not-exist.mmdb")
    assert resolver.lookup("8.8.8.8") is None


def test_country_resolver_returns_none_for_loopback_without_querying_the_reader() -> (
    None
):
    """A private/loopback address never reaches the reader, database or not."""

    class _FailingReader:
        def country(self, ip_address: str):
            raise AssertionError(f"reader queried for a loopback address: {ip_address}")

    resolver = CountryResolver(None)
    resolver._reader = _FailingReader()  # simulate a successfully opened database
    assert resolver.lookup("127.0.0.1") is None


_BUNDLED_GEOIP_DB = (
    Path(fastrag.serve.geolocalization.__file__).parent / "GeoLite2-Country.mmdb"
)


@pytest.mark.skipif(
    not _BUNDLED_GEOIP_DB.exists() or importlib.util.find_spec("geoip2") is None,
    reason="needs the GeoLite2 database and the `geoip` extra",
)
def test_country_resolver_resolves_a_public_address() -> None:
    resolver = CountryResolver(_BUNDLED_GEOIP_DB)
    try:
        assert resolver.lookup("8.8.8.8") == "US"
    finally:
        resolver.close()
    assert resolver.lookup("8.8.8.8") is None


async def test_session_row_carries_country_from_the_resolver_dependency(
    chat_store: PostgresChatStore,
) -> None:
    """``get_or_create_session`` receives the resolver through
    ``get_country_resolver``, so overriding it changes the persisted country."""

    class _FixedResolver(CountryResolver):
        def lookup(self, ip_address: str | None) -> str | None:
            return "ES"

    app = _app(chat_store)
    app.dependency_overrides[get_country_resolver] = lambda: _FixedResolver(None)
    async with AsyncClient(
        transport=ASGITransport(app=app), base_url="http://test"
    ) as client:
        response = await client.post("/v1/chats", json={"content": "Hi"})
        assert response.status_code == 200
        session_id = client.cookies.get("session_id")

    session = await chat_store.get_session(session_id)
    assert session is not None
    assert session.country == "ES"


async def test_cleanup_expired_deletes_only_past_rows() -> None:
    """``cleanup_expired`` deletes expired sessions/chats and leaves live
    ones untouched, reporting exactly what it removed.

    Runs against its own isolated ``PostgresChatStore`` schema, and its own
    raw engine pointed at that schema via ``server_settings.search_path`` --
    ``cleanup_expired``'s unqualified SQL resolves on the connection's
    search path, per its own docstring, so a caller integrating it with a
    non-default schema builds an engine that sets one, exactly as done here.
    """
    schema = f"fastrag_test_{uuid4().hex}"
    store = PostgresChatStore(database_url=DATABASE_URL, schema=schema)
    await store.__aenter__()
    try:
        live_session = await store.create_session(ttl_days=30)
        expired_session = await store.create_session(ttl_days=-1)
        expired_chat = await store.create_chat(session_id=live_session.id, ttl_days=-1)
        live_chat = await store.create_chat(session_id=live_session.id, ttl_days=30)

        scoped_engine = create_async_engine(
            ASYNCPG_DATABASE_URL,
            connect_args={"server_settings": {"search_path": schema}},
        )
        try:
            report = await cleanup_expired(scoped_engine, target="all")
        finally:
            await scoped_engine.dispose()

        assert report.sessions == 1
        assert report.chats == 1

        assert await store.get_session(live_session.id) is not None
        assert await store.get_session(expired_session.id) is None
        assert (
            await store.get_chat(live_chat.id, session_id=live_session.id) is not None
        )
        assert await store.get_chat(expired_chat.id, session_id=live_session.id) is None
    finally:
        await store.__aexit__(None, None, None)
        admin = await asyncpg.connect(DATABASE_URL)
        try:
            await admin.execute(f'DROP SCHEMA IF EXISTS "{schema}" CASCADE')
        finally:
            await admin.close()
