"""Client IP resolution.

One resolution of the *real* client IP is shared by the rate limiter and the
session writer (see ``_rate_limit.py`` and ``_auth.py``) — resolving it
twice would risk the two disagreeing about who a request is from. Behind a
reverse proxy, ``request.client.host`` is the proxy's own address, not the
caller's; trusting ``X-Forwarded-For`` instead is opt-in via
:attr:`~fastrag.serve.config.ServeConfig.trust_proxy_headers` and off by
default, because an untrusted ``X-Forwarded-For`` is a header any client can
set on itself — trusting it unconditionally would make both the rate limiter
and the per-session IP trivially spoofable.

Country resolution from that IP lives in :mod:`fastrag.serve.geolocalization`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from starlette.requests import Request

if TYPE_CHECKING:
    from fastrag.serve.config import ServeConfig


def resolve_client_ip(request: Request, config: ServeConfig) -> str | None:
    """Resolve the request's real client IP.

    ``request.client.host`` by default. Only when
    ``config.trust_proxy_headers`` is ``True`` does the leftmost entry of an
    ``X-Forwarded-For`` header (the original client, by convention — every
    proxy in the chain appends its own address to the right) override it.
    Falls back to ``request.client.host`` when the header is trusted but
    absent or empty.
    """
    if config.trust_proxy_headers:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            candidate = forwarded.split(",", 1)[0].strip()
            if candidate:
                return candidate
    client = request.client
    return client.host if client is not None else None
