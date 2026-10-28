"""Tests for the ask CLI: its pure helpers, and the interactive loop itself.

The first two tests cover the parts of ``fastrag.serve.cli`` importable with
no ``serving`` extra and no server -- token-file handling and the SSE parser
-- as deterministic units. The third drives the whole Typer app offline,
over an ``httpx.MockTransport`` standing in for a serving app, with
``builtins.input`` feeding it a script. That is where the chat-id threading
lives, and it needs the loop: no unit below the loop knows that the id
announced on ``message_start`` is what makes the *next* question a
follow-up.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import httpx
import pytest

from fastrag.serve.cli import CLI, load_token, parse_sse, save_token

# -- token-file handling ---------------------------------------------------


def test_token_file_handling(monkeypatch: object, tmp_path: Path) -> None:
    """``FASTRAG_SESSION_TOKEN`` wins over the file; a blank env value is
    treated as absent; a blank or missing file reads back as ``None`` (so
    the caller mints); and ``save_token``/``load_token`` are inverses for a
    real path, written mode 0600 -- it is a live credential.
    """
    token_file = tmp_path / "session"

    token_file.write_text("file-token\n")
    monkeypatch.setenv("FASTRAG_SESSION_TOKEN", "env-token")
    assert load_token(token_file) == "env-token"

    monkeypatch.delenv("FASTRAG_SESSION_TOKEN")
    assert load_token(token_file) == "file-token"

    token_file.write_text("  \n")
    assert load_token(token_file) is None

    assert load_token(tmp_path / "does-not-exist") is None

    token_file.write_text("file-token\n")
    monkeypatch.setenv("FASTRAG_SESSION_TOKEN", "")
    assert load_token(token_file) == "file-token"
    monkeypatch.delenv("FASTRAG_SESSION_TOKEN")

    written_file = tmp_path / "nested" / "session"
    written = save_token("sekret-token", written_file)
    assert written == written_file
    assert written_file.read_text().strip() == "sekret-token"
    # 0600 on POSIX; meaningless (but harmless) on platforms without os.chmod.
    assert (written_file.stat().st_mode & 0o777) == 0o600

    round_trip_file = tmp_path / "round-trip"
    save_token("round-trip-token", round_trip_file)
    assert load_token(round_trip_file) == "round-trip-token"


# -- SSE parser --------------------------------------------------------------


def test_parse_sse() -> None:
    """``event:``/``data:`` pairs accumulate and fire on the blank line;
    multiline ``data:`` concatenates before JSON-parsing (the server may
    split it); ``: keep-alive`` comments are ignored; a ``data:`` line with
    no preceding ``event:`` is dropped; ``event:`` with no ``data:`` fires
    once with ``{}`` (e.g. a ping); and an unterminated final event still
    dispatches at end of stream.
    """
    dispatch_text = (
        "event: message_start\n"
        'data: {"chat_id":"c1"}\n'
        "\n"
        "event: delta\n"
        'data: {"delta":{"text":"Hello"}}\n'
        "\n"
        "event: message_end\n"
        'data: {"finish_reason":"stop"}\n'
        "\n"
    )
    dispatch_events = parse_sse(dispatch_text)
    assert [e for e, _ in dispatch_events] == ["message_start", "delta", "message_end"]
    assert dispatch_events[0][1] == {"chat_id": "c1"}
    assert dispatch_events[2][1] == {"finish_reason": "stop"}

    multiline_text = 'event: delta\ndata: {"delta":{"te\ndata: xt":"a"}}\n\n'
    ((multiline_event, multiline_data),) = parse_sse(multiline_text)
    assert multiline_event == "delta"
    assert multiline_data == {"delta": {"text": "a"}}

    keepalive_text = ": keep-alive\n\nevent: message_end\ndata: {}\n\n: keep-alive\n"
    assert [e for e, _ in parse_sse(keepalive_text)] == ["message_end"]

    assert parse_sse('data: {"orphaned":true}\n\n') == []

    ((empty_event, empty_data),) = parse_sse("event: message_start\n\n")
    assert empty_event == "message_start"
    assert empty_data == {}

    trailing_text = 'event: message_end\ndata: {"finish_reason":"stop"}\n'
    ((trailing_event, trailing_data),) = parse_sse(trailing_text)
    assert trailing_event == "message_end"
    assert trailing_data == {"finish_reason": "stop"}


# -- the interactive loop ----------------------------------------------------
#
# Driven through the real public entry point (``CLI(...)()`` -- the Typer
# app, exactly as example 15 invokes it) with two seams: ``builtins.input``
# feeds the script, and ``transport=`` swaps a ``MockTransport`` in for the
# network. What these pin down is the chat id: the client must read it off
# the first answer's ``message_start`` and thread it through every later
# question, so the server replays the history -- the bug being that every
# turn opened a new chat and the assistant remembered nothing.


def _sse(*frames: tuple[str, dict[str, object]]) -> bytes:
    """Frame ``(event, data)`` pairs the way the server's ``sse_frame`` does."""
    return b"".join(
        f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()
        for event, data in frames
    )


class FakeServer:
    """A ``MockTransport`` handler with the serving API's chat semantics.

    Records every request as ``(method, path)`` and mints a fresh chat id per
    ``POST /v1/chats``, so a test can tell "continued the chat" from "opened
    a new one" by the paths alone.
    """

    def __init__(self, *, bearer_tokens: bool = True) -> None:
        self.bearer_tokens = bearer_tokens
        self.requests: list[tuple[str, str]] = []
        self.auth: list[str | None] = []
        self.chats: list[str] = []

    @property
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def _answer(self, chat_id: str) -> httpx.Response:
        return httpx.Response(
            200,
            headers={"content-type": "text/event-stream"},
            content=_sse(
                ("message_start", {"chat_id": chat_id, "message_id": "m1"}),
                ("sources", {"sources": [{"url": "u", "title": "t"}]}),
                ("delta", {"delta": {"type": "text", "text": "hi"}}),
                ("message_end", {"message_id": "m1", "finish_reason": "stop"}),
            ),
        )

    def handle(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        self.requests.append((request.method, path))
        self.auth.append(request.headers.get("authorization"))

        if request.method == "POST" and path == "/v1/sessions":
            if not self.bearer_tokens:
                return httpx.Response(404, json={"error": {"code": "NOT_FOUND"}})
            return httpx.Response(
                200,
                json={"session_id": "sess-1", "expires_at": "2099-01-01T00:00:00Z"},
            )
        if request.method == "POST" and path == "/v1/chats":
            chat_id = f"chat-{len(self.chats) + 1}"
            self.chats.append(chat_id)
            return self._answer(chat_id)
        if request.method == "POST" and path.endswith("/messages"):
            return self._answer(path.split("/")[3])
        if request.method == "GET" and path == "/v1/chats":
            return httpx.Response(
                200,
                json={
                    "first_id": None,
                    "last_id": None,
                    "has_more": False,
                    "chats": [
                        {"id": c, "created_at": "2026-01-01T00:00:00Z"}
                        for c in self.chats
                    ],
                },
            )
        if request.method == "GET" and path.startswith("/v1/chats/"):
            return httpx.Response(
                200,
                json={
                    "first_id": None,
                    "last_id": None,
                    "has_more": False,
                    "messages": [
                        {"id": "u1", "role": "user", "content": "earlier"},
                        {"id": "m1", "role": "assistant", "content": "hi"},
                    ],
                },
            )
        if request.method == "DELETE" and path.startswith("/v1/chats/"):
            return httpx.Response(204)
        raise AssertionError(f"unexpected request: {request.method} {path}")


def _run_cli(monkeypatch, server: FakeServer, script: list[str], **kwargs) -> None:
    """Run the Typer app to completion against ``server``, feeding ``script``."""
    lines = iter(script)

    def fake_input(prompt: str = "") -> str:
        try:
            return next(lines)
        except StopIteration:  # the script ran out: same as Ctrl-D
            raise EOFError from None

    monkeypatch.setattr("builtins.input", fake_input)
    monkeypatch.setattr(sys, "argv", ["fastrag-cli"])
    cli = CLI(backend_url="http://test", transport=server.transport, **kwargs)
    with pytest.raises(SystemExit) as exit_info:
        cli()
    assert exit_info.value.code == 0


def test_interactive_loop(monkeypatch, tmp_path) -> None:
    """The chat id from ``message_start`` targets every later question --
    without it each turn POSTs ``/v1/chats`` again and the model is handed
    an empty history every time -- and ``/new``/``/delete``/``/open``/a bare
    Enter each affect that threading exactly as documented. Separately: one
    ``POST /v1/sessions``, saved 0600 and bearer-sent from then on; a known
    token means no ``/v1/sessions`` round trip at all; and a ``404`` from
    ``/v1/sessions`` (bearer tokens disabled) is not fatal -- the client
    falls back to the cookie jar.
    """
    continuing = FakeServer()
    _run_cli(
        monkeypatch,
        continuing,
        ["What is AgrospAI?", "And who runs it?", "/quit"],
        token_file=tmp_path / "session-continuing",
    )
    assert continuing.requests == [
        ("POST", "/v1/sessions"),
        ("POST", "/v1/chats"),
        ("POST", "/v1/chats/chat-1/messages"),
    ]
    assert continuing.chats == ["chat-1"]

    starting_fresh = FakeServer()
    _run_cli(
        monkeypatch,
        starting_fresh,
        ["one", "two", "/new", "three", "four", "/quit"],
        token_file=tmp_path / "session-new",
    )
    assert [p for _, p in starting_fresh.requests if p.startswith("/v1/chats")] == [
        "/v1/chats",
        "/v1/chats/chat-1/messages",
        "/v1/chats",
        "/v1/chats/chat-2/messages",
    ]

    deleting = FakeServer()
    _run_cli(
        monkeypatch,
        deleting,
        ["one", "/delete", "two", "/quit"],
        token_file=tmp_path / "session-delete",
    )
    assert deleting.requests[1:] == [
        ("POST", "/v1/chats"),
        ("DELETE", "/v1/chats/chat-1"),
        ("POST", "/v1/chats"),
    ]

    opening = FakeServer()
    _run_cli(
        monkeypatch,
        opening,
        ["/open chat-9", "follow up", "/quit"],
        token_file=tmp_path / "session-open",
    )
    assert opening.requests[1:] == [
        ("GET", "/v1/chats/chat-9"),
        ("POST", "/v1/chats/chat-9/messages"),
    ]

    empty_line = FakeServer()
    _run_cli(
        monkeypatch,
        empty_line,
        ["", "   ", "/quit"],
        token_file=tmp_path / "session-empty",
    )
    assert empty_line.requests == [("POST", "/v1/sessions")]

    monkeypatch.delenv("FASTRAG_SESSION_TOKEN", raising=False)
    minting_file = tmp_path / "session-mint"
    minting_server = FakeServer()
    _run_cli(
        monkeypatch, minting_server, ["one", "two", "/quit"], token_file=minting_file
    )

    assert [p for _, p in minting_server.requests].count("/v1/sessions") == 1
    assert minting_file.read_text().strip() == "sess-1"
    assert (minting_file.stat().st_mode & 0o777) == 0o600
    # Everything after the mint is header-authenticated.
    assert minting_server.auth == [None, "Bearer sess-1", "Bearer sess-1"]

    supplied_server = FakeServer()
    _run_cli(
        monkeypatch,
        supplied_server,
        ["one", "/quit"],
        token_file=tmp_path / "session-supplied",
        token="preset-token",
    )
    assert supplied_server.requests == [("POST", "/v1/chats")]
    assert supplied_server.auth == ["Bearer preset-token"]
    assert not (tmp_path / "session-supplied").exists()

    monkeypatch.delenv("FASTRAG_SESSION_TOKEN", raising=False)
    fallback_file = tmp_path / "session-fallback"
    fallback_server = FakeServer(bearer_tokens=False)
    _run_cli(
        monkeypatch, fallback_server, ["one", "two", "/quit"], token_file=fallback_file
    )

    assert fallback_server.requests == [
        ("POST", "/v1/sessions"),
        ("POST", "/v1/chats"),
        ("POST", "/v1/chats/chat-1/messages"),
    ]
    assert fallback_server.auth == [None, None, None]
    assert not fallback_file.exists()
