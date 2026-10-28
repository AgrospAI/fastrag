"""Optional GeoIP country resolution for the serving app.

Grouped by feature: :mod:`.resolver` holds :class:`CountryResolver` (the
lookup itself) and :mod:`.dependencies` the FastAPI dependency that hands
the app-owned instance to a route or another dependency.
"""

from fastrag.serve.geolocalization.dependencies import (
    CountryResolverDep,
    get_country_resolver,
)
from fastrag.serve.geolocalization.resolver import CountryResolver

__all__ = ["CountryResolver", "CountryResolverDep", "get_country_resolver"]
