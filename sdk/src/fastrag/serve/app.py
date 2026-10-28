"""The FastAPI app: :func:`create_app`, lifespan, middleware, handlers."""

from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import cast

from fastapi import FastAPI
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ExceptionHandler

from fastrag import __version__
from fastrag.query.chat.base import ChatStore
from fastrag.query.engine import QueryEngine
from fastrag.serve._errors import (
    ApiError,
    api_error_handler,
    http_exception_handler,
    unhandled_exception_handler,
    validation_exception_handler,
)
from fastrag.serve._guards import RequestGuardMiddleware
from fastrag.serve._logging import configure_serve_logging
from fastrag.serve._metrics import (
    METRICS_ROUTE,
    MetricsMiddleware,
    ServeMetrics,
    metrics_response,
)
from fastrag.serve._rate_limit import RateLimiter
from fastrag.serve._routes import healthz_router, router
from fastrag.serve._schemas import ApiErrorEnvelope
from fastrag.serve.config import ServeConfig
from fastrag.serve.geolocalization import CountryResolver

#: Attached to every ``/v1`` route below via ``include_router(responses=...)``
#: — the request-level failures every route can produce (auth, guards, rate
#: limiting, validation, an unhandled error), documented once instead of
#: per-route. See ``ApiErrorEnvelope``'s own docstring for why this is the
#: only way FastAPI learns about a shape its exception handlers build by
#: hand. Generation-time failures (a stream that opens and then fails) are
#: a different, narrower vocabulary carried in the SSE ``message_end``
#: event's ``error`` field, not here — see ``_schemas.ErrorInfo``.
_COMMON_ERROR_RESPONSES: dict[int | str, dict] = {
    401: {
        "model": ApiErrorEnvelope,
        "description": "Missing or invalid session cookie/bearer token.",
    },
    404: {"model": ApiErrorEnvelope, "description": "Chat or message not found."},
    409: {
        "model": ApiErrorEnvelope,
        "description": "A generation is already running for this chat.",
    },
    413: {"model": ApiErrorEnvelope, "description": "Request body too large."},
    415: {"model": ApiErrorEnvelope, "description": "Unsupported Content-Type."},
    422: {
        "model": ApiErrorEnvelope,
        "description": "Validation failed, or a configured cap was reached.",
    },
    429: {"model": ApiErrorEnvelope, "description": "Rate limit exceeded."},
    500: {"model": ApiErrorEnvelope, "description": "Unhandled server error."},
}


@asynccontextmanager
async def _lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Release app-owned resources on shutdown.

    The collaborators passed to :func:`create_app` stay the caller's to
    close; only what the app builds itself (the GeoIP reader) is closed here.
    """
    try:
        yield
    finally:
        app.state.country_resolver.close()


def create_app(
    *,
    engine: QueryEngine,
    chat_store: ChatStore,
    config: ServeConfig | None = None,
) -> FastAPI:
    """Build the serving app.

    A thin adapter over its two collaborators, laid out in the example the
    SDK ships: a :class:`~fastrag.query.engine.QueryEngine` and a
    :class:`~fastrag.query.chat.base.ChatStore`. The app itself owns no
    database engine and no ORM session — chatting goes through the store's
    methods, generation through ``engine.stream``, and the advisory lock and
    ``/healthz`` connection check borrow the store's engine via its
    ``engine`` property. Both collaborators are required: there is no
    half-configured mode.

    Args:
        engine: The :class:`~fastrag.query.engine.QueryEngine` the API
            exposes — the single place retrieval and generation happen.
        chat_store: The :class:`~fastrag.query.chat.base.ChatStore` all
            session/chat/message persistence goes through. Already entered
            by the caller (``async with ... as chat_store``).
        config: :class:`~fastrag.serve.config.ServeConfig`. Defaults to a
            config with an empty ``allowed_origins`` list.

    Returns:
        A FastAPI app serving ``/healthz`` and the full ``/v1`` chats API.
    """
    config = config or ServeConfig(allowed_origins=[])

    configure_serve_logging("fastrag-serving", config.log_level)

    app = FastAPI(title="FastRAG API", version=__version__, lifespan=_lifespan)

    metrics = ServeMetrics()

    app.state.config = config
    app.state.query_engine = engine
    app.state.chat_store = chat_store
    app.state.engine = getattr(chat_store, "engine", None)
    app.state.metrics = metrics
    # In-process, per-worker only (see the serving guide) — both are cheap
    # to construct even when their config knob is off, so they are always
    # built rather than made conditional on `rate_limiting_enabled` /
    # `geoip_database_path`.
    app.state.rate_limiter = RateLimiter()
    app.state.country_resolver = CountryResolver(config.geoip_database_path)

    app.add_exception_handler(ApiError, cast(ExceptionHandler, api_error_handler))
    app.add_exception_handler(
        StarletteHTTPException, cast(ExceptionHandler, http_exception_handler)
    )
    app.add_exception_handler(
        RequestValidationError, cast(ExceptionHandler, validation_exception_handler)
    )
    app.add_exception_handler(
        Exception, cast(ExceptionHandler, unhandled_exception_handler)
    )

    # Registered first == outermost, so CORS headers apply to every
    # response, including errors raised by inner middleware.
    # MetricsMiddleware is registered next, outward of the guard, so a
    # request the guard rejects with 413/415 before it ever reaches the
    # router is still counted in http_requests_total.
    app.add_middleware(
        CORSMiddleware,  # type: ignore[arg-type]
        allow_origins=config.allowed_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(MetricsMiddleware, metrics=metrics)  # type: ignore[arg-type]
    app.add_middleware(RequestGuardMiddleware, config=config)  # type: ignore[arg-type]

    # Rate limiting is enforced per-route inside `router` itself (a
    # `rate_limit(...)` dependency on each gated endpoint — see
    # `_rate_limit.py` and `_routes.py`), not by a middleware here: a
    # dependency raises `ApiError` and is caught by the exception handlers
    # registered above, where a `BaseHTTPMiddleware` raising the same
    # exception would bypass them (see `RequestGuardMiddleware`'s own
    # hand-rolled `JSONResponse`, for the same reason).
    app.include_router(router, responses=_COMMON_ERROR_RESPONSES)
    app.include_router(healthz_router)

    if config.metrics_enabled:
        # A plain route, not a router: it sits outside /v1 and /healthz, the
        # same way healthz_router does, and is excluded from
        # http_requests_total by MetricsMiddleware itself (matching on the
        # path), not by omission here.
        app.add_api_route(METRICS_ROUTE, metrics_response, methods=["GET"])

    return app
