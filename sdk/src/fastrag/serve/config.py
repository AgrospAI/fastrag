"""Frozen configuration for :func:`fastrag.serve.create_app`."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fastrag.exceptions import ConfigurationError
from fastrag.serve._rate_limit import parse_rate


@dataclass(frozen=True, kw_only=True)
class ServeConfig:
    """Configuration for the serving app.

    Frozen and keyword-only. Nothing under ``src/fastrag/`` reads
    ``os.environ`` — the caller reads the environment and passes the result
    in, exactly as ``08_sources.py`` already does for the chat model. Fail
    fast rather than start in a degraded state: every field is validated in
    ``__post_init__``, raising
    :class:`~fastrag.exceptions.ConfigurationError` earlier than startup, at
    the line that got it wrong.

    Args:
        allowed_origins: Origins allowed by CORS. The one required field,
            with no default. A ``"*"`` wildcard is rejected: credentialed
            requests cannot use ``Access-Control-Allow-Origin: *``. The app
            configures CORS from this list itself, so :func:`~fastrag.serve.create_app`
            callers never add their own ``CORSMiddleware``.
        cookie_secure: Whether the session cookie carries the ``Secure`` flag.
        cookie_domain: Optional ``Domain`` for the session cookie.
        session_ttl_days: Days a session (and its chats) live without activity.
        expiry_bump_min_days: Minimum headroom before an expiry bump is written.
        max_request_bytes: Reject a request body larger than this with ``413``.
        max_content_chars: Reject a ``content`` longer than this with ``422``.
        max_messages_per_chat: Cap on messages per chat (``422`` beyond).
        max_chats_per_session: Cap on chats per session (``422`` beyond).
        generation_timeout_s: End-to-end budget (retrieval + generation) per
            generation; a timeout lands as ``TIMEOUT``.
        keepalive_interval_s: Seconds between SSE keep-alive comments, and
            between pings on the chat's lock connection while it answers.
            Keep it below the idle timeout of any proxy in front of the app
            or of Postgres.
        log_level: Root logger level for the JSON log output.
        bearer_tokens: Enable the bearer-token path and ``POST /v1/sessions``.
            Off by default — see "Bearer tokens, in full" in the serving
            design notes.
        metrics_enabled: Whether ``GET /metrics`` is mounted at all. On by
            default; every series is still collected when this is ``False``,
            only the route that exposes them is withheld.
        metrics_token: Optional bearer token guarding ``GET /metrics``.
            ``None`` (the default) leaves the endpoint unguarded — reasonable
            for a scrape target reachable only from inside a private network.
            When set, a request without a matching ``Authorization: Bearer``
            header gets ``401`` and no metric text.
        trust_proxy_headers: Whether the leftmost entry of an incoming
            ``X-Forwarded-For`` header is trusted as the real client IP,
            instead of ``request.client.host``. Off by default: this stack
            usually runs behind a reverse proxy, but trusting a
            client-settable header unconditionally is a trivial rate-limit
            and audit-log bypass. Turn this on only when a proxy in front of
            this app is known to set (and not merely pass through) the
            header.
        rate_limiting_enabled: Whether the three limits below are enforced at
            all. Off by default, so an existing caller or the test suite is
            unaffected until it opts in.
        rate_limit_generation: Limit on ``POST /v1/chats``,
            ``POST /v1/chats/{id}/messages``, and
            ``POST /v1/chats/{id}/messages/{mid}``, as a ``"10/minute"``-style
            string (``<count>/<second|minute|hour|day>``), per client IP.
        rate_limit_read: Limit on ``GET /v1/chats`` and ``GET /v1/chats/{id}``,
            same format, per client IP.
        rate_limit_new_session: Limit on the branch of a request that mints a
            new session (``POST /v1/chats`` and ``POST /v1/sessions`` when
            neither presents a live session), same format, per client IP —
            not a limit on those endpoints as a whole, which
            ``rate_limit_generation`` already covers for ``POST /v1/chats``.
        geoip_database_path: Path to a MaxMind GeoLite2-Country ``.mmdb``
            file, for resolving a session's ``country``. ``None`` (the
            default) disables country resolution entirely; nothing is
            downloaded or vendored by this package — see the serving guide
            for where to get one. Requires the ``geoip`` extra
            (``geoip2``); without it, country resolution is silently
            unavailable even when a path is set.
    """

    allowed_origins: list[str]
    cookie_secure: bool = True
    cookie_domain: str | None = None
    session_ttl_days: int = 30
    expiry_bump_min_days: int = 1
    max_request_bytes: int = 32768
    max_content_chars: int = 4000
    max_messages_per_chat: int = 200
    max_chats_per_session: int = 100
    generation_timeout_s: float = 120.0
    keepalive_interval_s: float = 15.0
    log_level: str = "INFO"
    bearer_tokens: bool = False
    metrics_enabled: bool = True
    metrics_token: str | None = None
    trust_proxy_headers: bool = False
    rate_limiting_enabled: bool = False
    rate_limit_generation: str = "10/minute"
    rate_limit_read: str = "60/minute"
    rate_limit_new_session: str = "20/hour"
    geoip_database_path: Path | None = None

    def __post_init__(self) -> None:
        """Validate every field, failing fast rather than at first request."""
        if "*" in self.allowed_origins:
            raise ConfigurationError(
                "ServeConfig.allowed_origins must not contain a wildcard; "
                "credentialed requests cannot use Access-Control-Allow-Origin: *"
            )
        for name in (
            "session_ttl_days",
            "expiry_bump_min_days",
            "max_request_bytes",
            "max_content_chars",
            "max_messages_per_chat",
            "max_chats_per_session",
        ):
            if getattr(self, name) < 1:
                raise ConfigurationError(f"ServeConfig.{name} must be at least 1")
        if self.generation_timeout_s <= 0:
            raise ConfigurationError(
                "ServeConfig.generation_timeout_s must be positive"
            )
        if self.keepalive_interval_s <= 0:
            raise ConfigurationError(
                "ServeConfig.keepalive_interval_s must be positive"
            )
        if self.metrics_token is not None and not self.metrics_token:
            raise ConfigurationError(
                "ServeConfig.metrics_token must not be an empty string; use "
                "None to leave /metrics unguarded"
            )
        for name in (
            "rate_limit_generation",
            "rate_limit_read",
            "rate_limit_new_session",
        ):
            try:
                parse_rate(getattr(self, name))
            except ConfigurationError as exc:
                raise ConfigurationError(f"ServeConfig.{name}: {exc}") from exc
