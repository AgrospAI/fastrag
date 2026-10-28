"""An interactive terminal client for the serving API, built on Typer.

It is pure ``httpx`` against the running server — no browser, no
``EventSource``. That is the point: all three streaming endpoints are
``POST``, so a browser's ``EventSource`` cannot be used, and this client
implements the ``fetch``-equivalent reader that replaces it (the SSE parsing
is written out rather than imported, so the public surface can't drag in
anything but ``httpx``).

**The token is a credential.** It is read from ``FASTRAG_SESSION_TOKEN`` if
set, else from a token file (default ``~/.fastrag/session``, created ``0600``
on first run), else minted on the fly by a single ``POST /v1/sessions``,
which mints a session when the request carries none, and is saved for later
runs. Every request after that is header-authenticated. On a server built
with ``bearer_tokens=False`` that route is a ``404``; the client then stays
tokenless and rides the session cookie its own jar holds. The token is never
logged; ``/token`` is the explicit opt-in to printing it.

**History.** The chat id is the client's whole memory of a conversation: it
is read off the ``message_start`` frame of the first answer and reused as the
``POST /v1/chats/{chat_id}/messages`` target for every later turn, so the
server replays the prior turns into the model. ``/new`` (or ``/delete``)
clears it, and only then does the next question open a fresh chat. Each
turn's prior messages are sent to the model (so a follow-up pronoun resolves
against earlier turns) but not to the retriever — the retriever only ever
sees the current question. A pronoun may therefore resolve in the answer
while the citations do not follow it.

**Edit semantics.** ``/retry`` and ``/edit`` POST to an existing message id
and stream the regenerate endpoint. The serving app validates the request
and streams a fresh answer, **appending** it as the latest assistant message
rather than truncating the run to that point — the store that backs the app
has no range-delete primitive, so the chat history grows instead of being
rewound. A long chat may therefore accumulate the retried turns; ``/open``
shows them all.

Commands: ``/new``, ``/chats``, ``/open <id>``, ``/retry``, ``/edit <text>``,
``/delete``, ``/token``, ``/help``, ``/quit``. Ctrl-C during a stream closes
the response and returns to the prompt; the server persists
``finish_reason: "interrupted"`` and ``/open`` afterwards shows the partial
answer.

Wire contract: this client only speaks HTTP to the side the app was started
with in mind — it never issues a request for a route that doesn't exist and
never changes the server's behaviour. If a route or event name looks wrong,
this client is wrong; the server's contract is owned by ``fastrag.serve``.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path
from typing import Any

import httpx

from fastrag.exceptions import ConfigurationError

_MISSING_EXTRA = (
    'CLI needs the "serving" extra: pip install "fastrag-ai[serving]" '
    "(or, for development, `task setup`/`uv sync --all-extras`)"
)

HELP = """\
Type a question to ask it. Commands:
  /new            start a new chat
  /chats          list chats
  /open <id>      replay a chat's history
  /retry          regenerate the last assistant message
  /edit <text>    resubmit the last user message with new text
  /delete         delete the current chat
  /token          print the current session token
  /help           this help
  /quit           exit
Ctrl-C during a stream stops it and returns to the prompt.
"""

#: The public names. The SSE parser and token-file helpers are deliberately
#: importable with no ``serving`` extra — they only pull in ``httpx``/``os``.
__all__ = ["CLI", "parse_sse", "load_token", "save_token", "HELP"]


def _default_token_file() -> Path:
    """Return the platform-appropriate location for the session token.

    ``~/.fastrag/session`` — a dotfile in the user's home, like ``~/.netrc``
    and ``~/.aws/credentials`` — rather than a platform config dir, because
    a session token is a private credential that SSH-style tooling expects
    under ``$HOME``, and the ``~/.fastrag/`` directory keeps sibling
    config (a future host list, a ``config.toml``) together. 0600 is applied
    on write; on a POSIX home this is readable only by the owning user.
    """
    return Path.home() / ".fastrag" / "session"


def load_token(token_file: Path | None = None) -> str | None:
    """Return the token, if one is already known: env, then file.

    ``FASTRAG_SESSION_TOKEN`` wins over the token file. Returns ``None`` when
    neither holds one — the caller then mints on first request.
    """
    env = os.environ.get("FASTRAG_SESSION_TOKEN")
    if env:
        return env
    path = token_file or _default_token_file()
    if path.exists():
        return path.read_text().strip() or None
    return None


def save_token(token: str, token_file: Path | None = None) -> Path:
    """Persist ``token`` 0600 — it is a live credential.

    Writes to ``token_file`` (default ``~/.fastrag/session``), creating the
    parent directory, with mode 0600. Returns the path that was written.
    """
    path = token_file or _default_token_file()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(token + "\n")
    os.chmod(path, 0o600)
    return path


def _auth_headers(token: str | None) -> dict[str, str]:
    return {"Authorization": f"Bearer {token}"} if token else {}


def parse_sse(text: str) -> list[tuple[str, dict[str, Any]]]:
    """Parse one raw SSE body into ``(event, data)`` pairs.

    A pure, deterministic reimplementation of the reader — line by line on
    the blank-line boundary, skipping ``: keep-alive`` comments, and
    dispatching separately. It is factored out of the streaming path so it
    can be unit-tested with no server and no network. Each ``data`` block is
    parsed as JSON; a ``data`` block with no lines yields ``{}``.
    """
    events: list[tuple[str, dict[str, Any]]] = []
    event = ""
    data_lines: list[str] = []

    def dispatch() -> None:
        nonlocal event, data_lines
        if not event:
            return
        data: dict[str, Any] = {}
        if data_lines:
            data = json.loads("".join(data_lines))
        events.append((event, data))
        event = ""
        data_lines = []

    for line in text.splitlines():
        if line == "":
            dispatch()
        elif line.startswith(": "):
            continue  # keep-alive comment
        elif line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])
    dispatch()
    return events


async def _read_sse(
    response: httpx.Response, result: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Consume one SSE stream, dispatching on the blank-line boundary.

    Accumulates ``event:`` / ``data:`` lines, fires on the blank line, and
    skips ``: keep-alive`` comments. Live: prints sources before the answer
    and streams ``delta`` text as it arrives.

    Returns ``message_start`` merged under the last ``message_end`` — the
    two frames carry different keys and the caller needs both: ``chat_id``
    is announced once, on ``message_start``, and is what makes the *next*
    question a follow-up rather than a new chat, while ``finish_reason`` and
    ``error`` only ever arrive on ``message_end``.

    ``result``, when passed, is that same dict, filled in place as the frames
    arrive; the caller keeps it when the stream is interrupted (Ctrl-C mid
    answer still leaves it holding the ``chat_id``, so the new chat is not
    orphaned).
    """
    event = ""
    data_lines: list[str] = []
    merged: dict[str, Any] = result if result is not None else {}

    async def dispatch() -> None:
        nonlocal event, data_lines
        if not event:
            return
        data = json.loads("".join(data_lines)) if data_lines else {}
        if event in ("message_start", "message_end"):
            merged.update(data)
        elif event == "sources":
            _print_sources(data.get("sources", []))
        elif event == "delta":
            print(data["delta"]["text"], end="", flush=True)
        event = ""
        data_lines = []

    async for line in response.aiter_lines():
        if line == "":
            await dispatch()
        elif line.startswith(": "):
            continue  # keep-alive comment
        elif line.startswith("event: "):
            event = line[len("event: ") :]
        elif line.startswith("data: "):
            data_lines.append(line[len("data: ") :])

    await dispatch()
    return merged


def _print_sources(sources: list[dict[str, Any]]) -> None:
    """Print the sources header, then the ``answer:`` label."""
    print()
    for i, source in enumerate(sources, 1):
        print(f"  {i}. {source.get('title') or source.get('url')}")
    print("\nanswer:")


class CLI:
    """An interactive terminal client for the serving API, built on Typer.

    ``CLI(backend_url=...)`` returns an object; calling it (``cli()``) runs
    the interactive loop. ``backend_url`` is the base URL of a serving app
    started from :func:`~fastrag.serve.create_app`. ``token_file`` points at
    where the session token is read from and written to (default
    ``~/.fastrag/session``); ``token`` short-circuits all token discovery by
    supplying the credential directly.

    Typer is imported lazily inside the constructor, so merely importing this
    class needs no ``serving`` extra. Only constructing a ``CLI`` does.
    """

    def __init__(
        self,
        backend_url: str,
        *,
        token_file: Path | str | None = None,
        token: str | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        """Build the Typer app for a serving backend at ``base_url``.

        Imports Typer (the only ``serving``-extra dependency this class
        needs); a missing Typer surfaces as a
        :class:`~fastrag.exceptions.ConfigurationError` naming the install
        command. `backend_url` is the serving app's base; `token_file`
        overrides the default ``~/.fastrag/session`` for the session-token;
        ``token`` supplies the credential directly, skipping all discovery.
        ``transport`` replaces the HTTP transport of the one client this loop
        opens — an ``httpx.ASGITransport`` to drive an in-process app, or an
        ``httpx.MockTransport`` to drive the loop from a test.
        """
        try:
            import typer
        except ImportError as error:  # pragma: no cover - env-dependent
            raise ConfigurationError(_MISSING_EXTRA) from error
        self._typer = typer
        self._backend_url = backend_url.rstrip("/")
        self._token_file = Path(token_file) if token_file else None
        self._transport = transport
        # Explicit token beats the file and the env; a None falls through to
        # load_token's env-then-file order at call time.
        if token is not None:
            self._token = token
        else:
            self._token = load_token(self._token_file)
        self._app = typer.Typer(
            help=HELP,
            # Calling the object runs the interactive loop — a bare
            # invocation is the point, so no_args_is_help would be wrong
            # here. --help still works from Typer.
            add_completion=False,
        )
        self._app.callback(invoke_without_command=True)(self._run)

    @property
    def token(self) -> str | None:
        """The current session token (bare credential — handle with care)."""
        return self._token

    def __call__(self) -> Any:
        """Run the interactive client (``CLI(...)()``)."""
        return self._app()

    def _run(self) -> None:
        asyncio.run(self._amain())

    # -- the loop -------------------------------------------------------

    async def _amain(self) -> None:
        token = self._token
        chat_id: str | None = None

        async with httpx.AsyncClient(timeout=None, transport=self._transport) as client:
            if token is None:
                token = await self._mint_token(client)

            print(HELP)
            while True:
                try:
                    line = await asyncio.to_thread(input, "> ")
                except (EOFError, KeyboardInterrupt):
                    print()
                    break

                line = line.strip()
                if not line:
                    continue

                if line.startswith("/"):
                    cmd, _, arg = line.partition(" ")
                    arg = arg.strip()
                    stop, chat_id = await self._command(
                        client, token, chat_id, cmd, arg
                    )
                    if stop:
                        break
                    continue

                # A plain question.
                chat_id, _ = await self._stream_question(client, token, chat_id, line)

    async def _mint_token(self, client: httpx.AsyncClient) -> str | None:
        """Mint and persist a session token, or fall back to the cookie jar.

        ``POST /v1/sessions`` mints a session when the request presents none,
        so this costs one request and no generation. A ``404`` means the
        server was built with ``bearer_tokens=False``: nothing is saved and
        ``None`` is returned, leaving the client to authenticate with the
        session cookie the first ``POST /v1/chats`` hands its jar.
        """
        response = await client.post(f"{self._backend_url}/v1/sessions", json={})
        if response.status_code == 404:
            print("server has bearer tokens disabled - using the session cookie")
            print()
            return None
        response.raise_for_status()
        token = response.json()["session_id"]
        path = save_token(token, self._token_file)
        self._token = token
        print(f"minted session {token} (saved to {path.name})")
        print()
        return token

    async def _stream_question(
        self,
        client: httpx.AsyncClient,
        token: str | None,
        chat_id: str | None,
        content: str,
    ) -> tuple[str | None, dict[str, Any]]:
        """POST the question and stream the answer.

        Returns ``(chat_id, end)``, where ``end`` merges the stream's
        ``message_start`` and ``message_end`` frames. The returned chat id is
        what the caller threads into the next question — on the first turn it
        is the id the server just minted, read off ``message_start``.

        Ctrl-C mid-stream closes the response and returns to the prompt — the
        server persists ``finish_reason: "interrupted"`` — without killing
        the client; the chat id seen so far survives, so the interrupted chat
        stays the current one instead of being orphaned.
        """
        if chat_id is None:
            url = f"{self._backend_url}/v1/chats"
        else:
            url = f"{self._backend_url}/v1/chats/{chat_id}/messages"

        started = time.monotonic()
        end: dict[str, Any] = {}
        try:
            async with client.stream(
                "POST",
                url,
                json={"content": content},
                headers=_auth_headers(token),
            ) as response:
                response.raise_for_status()
                await _read_sse(response, end)
        except (KeyboardInterrupt, asyncio.CancelledError):
            # Close the response so the server sees the disconnect and
            # persists finish_reason "interrupted"; return to the prompt.
            print()
            return end.get("chat_id") or chat_id, {}

        seconds = time.monotonic() - started
        reason = end.get("finish_reason", "interrupted")
        if "error" in end:
            err = end["error"]
            print(
                f"\n\n[{reason}] error {err.get('code')}: "
                f"{err.get('message')} ({seconds:.1f}s)"
            )
        else:
            print(f"\n\n[{reason}] ({seconds:.1f}s)")
        return end.get("chat_id") or chat_id, end

    async def _command(
        self,
        client: httpx.AsyncClient,
        token: str | None,
        chat_id: str | None,
        cmd: str,
        arg: str,
    ) -> tuple[bool, str | None]:
        """Run one slash command.

        Returns ``(stop, chat_id)`` — whether the loop should exit, and the
        current chat id (``/open`` switches to the opened chat, ``/new`` and
        ``/delete`` clear it).
        """
        if cmd == "/quit":
            return True, chat_id
        elif cmd == "/help":
            print(HELP)
        elif cmd == "/new":
            chat_id = None
            print("new chat")
        elif cmd == "/chats":
            r = await client.get(
                f"{self._backend_url}/v1/chats", headers=_auth_headers(token)
            )
            r.raise_for_status()
            for c in r.json()["chats"]:
                print(f"  {c['id']}  {c['created_at']}")
        elif cmd == "/open":
            if not arg:
                print("usage: /open <chat_id>")
            else:
                chat_id = arg
                await self._open(client, token, chat_id)
        elif cmd == "/retry":
            await self._retry(client, token, chat_id)
        elif cmd == "/edit":
            if chat_id is None:
                print("no chat open")
            elif not arg:
                print("usage: /edit <text>")
            else:
                await self._edit(client, token, chat_id, arg)
        elif cmd == "/delete":
            if chat_id is None:
                print("no chat open")
            else:
                r = await client.delete(
                    f"{self._backend_url}/v1/chats/{chat_id}",
                    headers=_auth_headers(token),
                )
                r.raise_for_status()
                print("deleted")
                chat_id = None
        elif cmd == "/token":
            print(token)
        else:
            print(f"unknown command: {cmd} (try /help)")
        return False, chat_id

    async def _open(
        self, client: httpx.AsyncClient, token: str | None, chat_id: str
    ) -> None:
        """Replay a chat's history and make it the current chat."""
        r = await client.get(
            f"{self._backend_url}/v1/chats/{chat_id}", headers=_auth_headers(token)
        )
        r.raise_for_status()
        for m in r.json()["messages"]:
            role = m["role"]
            content = m.get("content") or ""
            if role == "user":
                print(f"you: {content}")
            elif role == "assistant":
                print(f"bot: {content[:200]}")

    async def _last_message_id(
        self,
        client: httpx.AsyncClient,
        token: str | None,
        chat_id: str,
        role: str,
    ) -> str | None:
        """Return the latest ``role`` message's id in the current chat, if any."""
        r = await client.get(
            f"{self._backend_url}/v1/chats/{chat_id}", headers=_auth_headers(token)
        )
        r.raise_for_status()
        messages = r.json()["messages"]
        matches = [m["id"] for m in messages if m["role"] == role]
        return matches[-1] if matches else None

    async def _retry(
        self, client: httpx.AsyncClient, token: str | None, chat_id: str | None
    ) -> None:
        """Regenerate the last assistant message (appends; see module doc)."""
        if chat_id is None:
            print("no chat open")
            return
        target = await self._last_message_id(client, token, chat_id, "assistant")
        if target is None:
            print("no assistant message to regenerate")
            return
        async with client.stream(
            "POST",
            f"{self._backend_url}/v1/chats/{chat_id}/messages/{target}",
            json={},
            headers=_auth_headers(token),
        ) as response:
            response.raise_for_status()
            await _read_sse(response)

    async def _edit(
        self,
        client: httpx.AsyncClient,
        token: str | None,
        chat_id: str,
        text: str,
    ) -> None:
        """Resubmit the last user message with new text (appends; see module doc)."""
        target = await self._last_message_id(client, token, chat_id, "user")
        if target is None:
            print("no user message to edit")
            return
        async with client.stream(
            "POST",
            f"{self._backend_url}/v1/chats/{chat_id}/messages/{target}",
            json={"content": text},
            headers=_auth_headers(token),
        ) as response:
            response.raise_for_status()
            await _read_sse(response)
