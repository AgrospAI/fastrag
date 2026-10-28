"""Bearer/cookie session authentication and session persistence.

``session_id`` is the bearer secret: a bearer token *is* a session ULID,
valid exactly as long as its session row is, and revoking is deleting the
session. Session rows are persisted through the :class:`~fastrag.query.chat.base.ChatStore`;
this module only orchestrates resolving/store-touching around it. See
"Bearer tokens, in full" in the serving design notes.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Annotated, Literal

from fastapi import Cookie, Header, Request, Response

from fastrag.query.chat.base import ChatStore, Session
from fastrag.serve._errors import ApiError
from fastrag.serve._metrics import ServeMetrics
from fastrag.serve._net import resolve_client_ip
from fastrag.serve._rate_limit import RateLimiter
from fastrag.serve.config import ServeConfig
from fastrag.serve.geolocalization import CountryResolverDep

SESSION_COOKIE_NAME = "session_id"


@dataclass(frozen=True)
class Authenticated:
    """The resolved session and how it authenticated.

    ``via`` is ``"bearer"`` only when an ``Authorization`` header
    authenticated the request and ``"cookie"`` otherwise (including the mint
    case). Routes call :func:`set_session_cookie` only when ``via == "cookie"``.
    ``minted`` is ``True`` only when this request created the session.
    """

    session: Session
    via: Literal["cookie", "bearer"]
    minted: bool = False


def set_session_cookie(
    response: Response, session_id: str, config: ServeConfig
) -> None:
    response.set_cookie(
        key=SESSION_COOKIE_NAME,
        value=session_id,
        max_age=config.session_ttl_days * 86400,
        httponly=True,
        secure=config.cookie_secure,
        samesite="lax",
        domain=config.cookie_domain,
        path="/",
    )


def _get_store(request: Request) -> ChatStore:
    return request.app.state.chat_store


def _get_config(request: Request) -> ServeConfig:
    return request.app.state.config


def _resolve_bearer(authorization: str | None, config: ServeConfig) -> str | None:
    """Resolve ``Authorization: Bearer <token>`` to a session id.

    A malformed header — present but not ``Bearer <token>`` — is ``401``
    immediately and never falls through to the cookie: falling through would
    let a browser's ambient cookie silently rescue a broken token and hide
    the client bug. Returns ``None`` when the header is absent.
    """
    if authorization is None:
        return None
    if not config.bearer_tokens:
        raise ApiError(401, "UNAUTHORIZED", "Bearer tokens are disabled")
    scheme, _, token = authorization.partition(" ")
    if scheme.lower() != "bearer" or not token:
        raise ApiError(401, "UNAUTHORIZED", "Malformed Authorization header")
    return token.strip()


async def _touch_session(
    store: ChatStore, session: Session, config: ServeConfig
) -> None:
    """Throttled expiry bump for a session on activity.

    The store's ``touch_session`` writes only when the candidate expiry is at
    least ``expiry_bump_min_days`` later than the stored one, so an active
    conversation stays alive without a write on every request.
    """
    await store.touch_session(
        session.id,
        ttl_days=config.session_ttl_days,
        min_bump_days=config.expiry_bump_min_days,
    )


async def _resolve_bearer_session(
    store: ChatStore, authorization: str | None, config: ServeConfig
) -> Authenticated | None:
    """Resolve a bearer header to an authenticated session, or ``None``.

    Shared by :func:`require_session` and :func:`get_or_create_session`:
    both treat a valid bearer identically, differing only in what happens
    when no bearer is presented.
    """
    bearer = _resolve_bearer(authorization, config)
    if bearer is None:
        return None
    session = await store.get_session(bearer)
    if session is None:
        raise ApiError(401, "UNAUTHORIZED", "Unknown or expired session")
    await _touch_session(store, session, config)
    return Authenticated(session=session, via="bearer")


async def require_session(
    request: Request,
    response: Response,
    authorization: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Cookie()] = None,
) -> Authenticated:
    """Resolve the caller's session, or ``401``.

    Resolution order: bearer header, then cookie, then unauthenticated.
    Used by every endpoint except ``POST /v1/chats`` and
    ``POST /v1/sessions`` (see :func:`get_or_create_session`).
    """
    config = _get_config(request)
    store = _get_store(request)

    authenticated = await _resolve_bearer_session(store, authorization, config)
    if authenticated is not None:
        return authenticated

    if not session_id:
        raise ApiError(401, "UNAUTHORIZED", "Missing session_id cookie")

    session = await store.get_session(session_id)
    if session is None:
        raise ApiError(401, "UNAUTHORIZED", "Unknown or expired session")

    await _touch_session(store, session, config)
    set_session_cookie(response, session.id, config)
    return Authenticated(session=session, via="cookie")


async def get_or_create_session(
    request: Request,
    response: Response,
    country_resolver: CountryResolverDep,
    authorization: Annotated[str | None, Header()] = None,
    session_id: Annotated[str | None, Cookie()] = None,
) -> Authenticated:
    """Resolve the caller's session, minting one when none is presented.

    Used only by ``POST /v1/chats`` and ``POST /v1/sessions``: mints when
    absent, unknown, **or expired**, because failing there strands a
    returning visitor with no way to recover.

    Minting is the one place ``ServeConfig.rate_limit_new_session`` applies —
    deliberately narrower than the endpoint as a whole, which already sits
    behind ``rate_limit_generation`` (``POST /v1/chats``) or is otherwise
    ungated (``POST /v1/sessions``) — and the one place a session's
    ``ip_address``/``country`` are ever written, since they are set once, at
    creation, never updated after.
    """
    config = _get_config(request)
    store = _get_store(request)

    authenticated = await _resolve_bearer_session(store, authorization, config)
    if authenticated is not None:
        return authenticated

    session = await store.get_session(session_id) if session_id else None

    if session is None:
        _enforce_new_session_limit(request, config)
        ip_address = resolve_client_ip(request, config)
        session = await store.create_session(
            ttl_days=config.session_ttl_days,
            ip_address=ip_address,
            country=country_resolver.lookup(ip_address),
        )
        minted = True
    else:
        await _touch_session(store, session, config)
        minted = False

    set_session_cookie(response, session.id, config)
    return Authenticated(session=session, via="cookie", minted=minted)


def _enforce_new_session_limit(request: Request, config: ServeConfig) -> None:
    """Apply ``rate_limit_new_session`` to the branch that mints a session.

    A no-op when rate limiting is disabled. Raises ``ApiError(429, ...)`` —
    a normal ``ApiError``, resolved by the registered exception handler like
    any other — when the limit is exceeded.
    """
    if not config.rate_limiting_enabled:
        return
    metrics: ServeMetrics = request.app.state.metrics
    limiter: RateLimiter = request.app.state.rate_limiter
    route = request.scope.get("route")
    route_path = route.path if route is not None else request.url.path
    metrics.requests_per_ip_total.labels(route=route_path).inc()

    ip = resolve_client_ip(request, config)
    if not limiter.hit(f"new_session:{ip}", config.rate_limit_new_session):
        metrics.rejected_requests_total.inc()
        raise ApiError(
            429,
            "RATE_LIMIT",
            f"Rate limit exceeded ({config.rate_limit_new_session})",
        )
