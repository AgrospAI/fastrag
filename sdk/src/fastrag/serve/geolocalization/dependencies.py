"""FastAPI dependency exposing the app's :class:`CountryResolver`."""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, Request

from fastrag.serve.geolocalization.resolver import CountryResolver


def get_country_resolver(request: Request) -> CountryResolver:
    """Return the :class:`CountryResolver` built once by ``create_app``.

    A dependency rather than a direct ``app.state`` read, so tests can swap
    it via ``app.dependency_overrides[get_country_resolver]``.
    """
    return request.app.state.country_resolver


CountryResolverDep = Annotated[CountryResolver, Depends(get_country_resolver)]
