"""Shared HTTP plumbing for remote PDF parsing services."""

from __future__ import annotations

from typing import Any

import httpx

from fastrag.exceptions import ParseError, TransientParseError


async def post_pdf(
    client: httpx.AsyncClient, service: str, url: str, **kwargs: Any
) -> dict[str, Any]:
    """POST and return the JSON body; secrets never reach the error text."""
    try:
        response = await client.post(url, **kwargs)
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as error:
        status = error.response.status_code
        message = f"{service} returned HTTP {status}"
        raise (
            TransientParseError if status in {408, 429} or status >= 500 else ParseError
        )(message) from error
    except (httpx.TransportError, httpx.TimeoutException) as error:
        raise TransientParseError(
            f"{service} request failed: {type(error).__name__}"
        ) from error
    except ValueError as error:
        raise ParseError(f"{service} returned a non-JSON response") from error
    if not isinstance(body, dict):
        raise ParseError(f"{service} returned an unexpected response")
    return body
