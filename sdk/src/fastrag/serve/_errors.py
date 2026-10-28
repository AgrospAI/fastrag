"""The typed API error and its handlers, plus the ``error_code`` mapping."""

from __future__ import annotations

from fastapi import Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from fastrag.exceptions import (
    ChatError,
    ContextLengthError,
    EmbeddingError,
    TransientChatError,
    VectorStoreError,
)
from fastrag.query.chat.base import ErrorCode


#: The request-level (pre-generation) status-code → ``code`` mapping used by
#: ``http_exception_handler`` below. Named so ``serve/_schemas.py`` can build
#: the OpenAPI-facing ``ApiErrorCode`` enum from the same values instead of
#: retyping them — this dict and ``REQUEST_ERROR_CODES`` are the single
#: source of truth for "every code a request-level failure can carry."
HTTP_STATUS_ERROR_CODES: dict[int, str] = {
    401: "UNAUTHORIZED",
    404: "NOT_FOUND",
    409: "CONFLICT",
    413: "PAYLOAD_TOO_LARGE",
    415: "UNSUPPORTED_MEDIA_TYPE",
    429: "RATE_LIMIT",
}

#: Every ``code`` a request-level (pre-generation) failure can carry: the
#: explicit mapping above, plus the two defaults ``http_exception_handler``
#: falls back to and the two handlers below always use. A generation that
#: starts and then fails uses the separate, narrower vocabulary in
#: ``fastrag.query.chat.base.ERROR_CODES`` instead (see that module's
#: ``ErrorCode` and this file's own ``_error_code_for`` docstring).
REQUEST_ERROR_CODES: tuple[str, ...] = (
    *HTTP_STATUS_ERROR_CODES.values(),
    "INVALID_REQUEST",
    "INTERNAL_ERROR",
)


class ApiError(Exception):
    """Raised for any non-streaming API failure.

    Produces the documented ``{"error": {"code", "message"}}`` envelope.
    Streaming endpoints stop using this once the stream has opened — see
    ``_generation.py``.
    """

    def __init__(self, status_code: int, code: str, message: str) -> None:
        self.status_code = status_code
        self.code = code
        self.message = message
        super().__init__(message)


def _envelope(code: str, message: str) -> dict:
    return {"error": {"code": code, "message": message}}


async def api_error_handler(_: Request, exc: ApiError) -> JSONResponse:
    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(exc.code, exc.message),
    )


async def http_exception_handler(
    _: Request, exc: StarletteHTTPException
) -> JSONResponse:
    code = HTTP_STATUS_ERROR_CODES.get(
        exc.status_code,
        "INTERNAL_ERROR" if exc.status_code >= 500 else "INVALID_REQUEST",
    )

    return JSONResponse(
        status_code=exc.status_code,
        content=_envelope(code, str(exc.detail)),
        headers=getattr(exc, "headers", None),
    )


async def validation_exception_handler(
    _: Request, exc: RequestValidationError
) -> JSONResponse:
    first = exc.errors()[0] if exc.errors() else None
    message = first["msg"] if first else "Invalid request body"
    return JSONResponse(
        status_code=422,
        content=_envelope("INVALID_REQUEST", message),
    )


async def unhandled_exception_handler(_: Request, __: Exception) -> JSONResponse:
    return JSONResponse(
        status_code=500,
        content=_envelope("INTERNAL_ERROR", "Something went wrong"),
    )


def _error_code_for(error: Exception) -> ErrorCode:
    """Map a generation-time exception onto the ``error_code`` ENUM.

    Called only from ``_generation.py``, once a stream has already opened,
    to fill the ``error_code`` persisted on the failed assistant
    ``ChatMessage`` and sent in the SSE ``message_end`` frame. A
    request-level failure (one caught before generation starts) never goes
    through this function — it gets a free-form ``code`` string from the
    handlers below instead (``api_error_handler``, ``http_exception_handler``,
    ``validation_exception_handler``), which is a separate vocabulary that
    is not restricted to the nine ``ErrorCode`` values (``UNAUTHORIZED``,
    ``NOT_FOUND``, ``CONFLICT``, ``PAYLOAD_TOO_LARGE`` and
    ``UNSUPPORTED_MEDIA_TYPE`` all appear there and are not in the ENUM at
    all).

    The ENUM keeps all nine spec values — the database type must match the
    spec, and narrowing it later is a migration — but this function returns
    only five of them: ``TIMEOUT``, ``CONTEXT_TOO_LONG``,
    ``SERVICE_UNAVAILABLE``, ``MODEL_ERROR``, and the ``INTERNAL_ERROR``
    default. Of the remaining four:

    - ``RATE_LIMIT`` does have a raise site now (``_guards.rate_limit`` and
      ``_auth._enforce_new_session_limit``, both raising ``ApiError(429,
      "RATE_LIMIT", ...)`` — see the serving guide's admission page), but it
      is still never returned *by this function*: every rate limit check
      runs before a generation's stream opens, so it is always a
      request-level failure like the others in this list, never a
      generation-time one.
    - ``NETWORK_ERROR`` and ``INVALID_RESPONSE`` have no raise site anywhere
      in ``fastrag.serve`` today.
    - ``INVALID_REQUEST`` is never *returned by this function*, but it is
      the single most common code the module as a whole produces: every
      request ``validation_exception_handler`` or an ``ApiError(422, ...)``
      rejects before a stream opens carries it. It just never becomes a
      persisted generation error, because a request that never starts a
      generation never reaches ``_error_code_for``.
    """
    if isinstance(error, TimeoutError):
        return "TIMEOUT"
    if isinstance(error, ContextLengthError):
        return "CONTEXT_TOO_LONG"
    if isinstance(error, (VectorStoreError, EmbeddingError)):
        return "SERVICE_UNAVAILABLE"
    if isinstance(error, (ChatError, TransientChatError)):
        return "MODEL_ERROR"
    return "INTERNAL_ERROR"
