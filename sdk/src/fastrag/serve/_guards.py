"""Reject a malformed or inadmissible request before it reaches a handler."""

from __future__ import annotations

from typing import Literal

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

from fastrag.serve._errors import ApiError
from fastrag.serve._metrics import ServeMetrics
from fastrag.serve._net import resolve_client_ip
from fastrag.serve._rate_limit import RateLimiter
from fastrag.serve.config import ServeConfig


class RequestGuardMiddleware(BaseHTTPMiddleware):
    """Rejects a malformed request before its body is parsed.

    Non-JSON ``Content-Type`` is ``415``, an oversized body is ``413``. Only
    the streaming POST endpoints take a body; every other route is a
    GET/DELETE, so gating on method is sufficient.
    """

    def __init__(self, app, config: ServeConfig) -> None:
        super().__init__(app)
        self.config = config

    async def dispatch(self, request: Request, call_next):
        if request.method == "POST":
            content_type = request.headers.get("content-type", "")
            if not content_type.startswith("application/json"):
                return JSONResponse(
                    status_code=415,
                    content={
                        "error": {
                            "code": "UNSUPPORTED_MEDIA_TYPE",
                            "message": "Content-Type must be application/json",
                        }
                    },
                )

            content_length = request.headers.get("content-length")
            max_bytes = self.config.max_request_bytes
            if content_length is not None and int(content_length) > max_bytes:
                return JSONResponse(
                    status_code=413,
                    content={
                        "error": {
                            "code": "PAYLOAD_TOO_LARGE",
                            "message": "Request body exceeds max_request_bytes",
                        }
                    },
                )

        return await call_next(request)


def enforce_content_length(content: str, config: ServeConfig) -> None:
    """Reject a ``content`` longer than ``config.max_content_chars`` with ``422``.

    The sibling check to this module's ``max_request_bytes`` guard above, but
    it cannot live in the same middleware: ``RequestGuardMiddleware`` runs
    before the body is parsed (it only ever looks at headers), while
    ``content`` only exists once FastAPI has parsed the body into a
    ``ContentRequest``/``OptionalContentRequest``. So this is a plain
    function instead, called once per route that carries a ``content``
    field — ``create_chat``, ``add_message``, and ``regenerate_message`` in
    ``_routes.py`` — right after the body is available, rather than
    middleware every route would otherwise have to duplicate the same
    length check to reach.
    """
    if len(content) > config.max_content_chars:
        raise ApiError(
            422,
            "INVALID_REQUEST",
            f"content exceeds max_content_chars ({config.max_content_chars})",
        )


def rate_limit(bucket: Literal["generation", "read"]):
    """Build a FastAPI dependency enforcing one of the two per-route limits.

    A plain dependency, not a middleware: it runs inside the router (after
    ``request.scope["route"]`` is already set by FastAPI's route matching,
    and before the endpoint body), so raising :class:`~fastrag.serve._errors.ApiError`
    here reaches the exception handlers registered in ``app.py`` exactly like
    any other request-time error — unlike a ``BaseHTTPMiddleware`` (see
    ``RequestGuardMiddleware`` above), which sits outside the exception
    handling machinery and would have to construct the error envelope by
    hand instead.

    The third limit, ``rate_limit_new_session``, is not wired up here: it
    only applies to the branch of a request that actually mints a session,
    not to the endpoint as a whole, so it is checked inline in
    ``_auth.get_or_create_session`` instead.

    A no-op when ``ServeConfig.rate_limiting_enabled`` is ``False`` — no
    metric is touched and no limiter state is recorded, so a config built
    without opting in behaves exactly as it did before this existed.
    """

    async def _check(request: Request) -> None:
        config: ServeConfig = request.app.state.config
        if not config.rate_limiting_enabled:
            return

        metrics: ServeMetrics = request.app.state.metrics
        limiter: RateLimiter = request.app.state.rate_limiter
        spec = (
            config.rate_limit_generation
            if bucket == "generation"
            else config.rate_limit_read
        )
        route = request.scope.get("route")
        route_path = route.path if route is not None else request.url.path
        metrics.requests_per_ip_total.labels(route=route_path).inc()

        ip = resolve_client_ip(request, config)
        if not limiter.hit(f"{bucket}:{ip}", spec):
            metrics.rejected_requests_total.inc()
            raise ApiError(429, "RATE_LIMIT", f"Rate limit exceeded ({spec})")

    return _check
