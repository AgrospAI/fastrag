"""IP → country resolution against an operator-supplied GeoLite2 database.

Country resolution is an optional concern layered on top of client IP
resolution (``fastrag.serve._net.resolve_client_ip``): it needs nothing from
the request except that IP, reads an operator-supplied MaxMind GeoLite2
database, and degrades to ``None`` whenever anything about that is
unavailable — no database configured, one that can't be opened, the
``geoip2`` extra not installed, or an address that was never going to be
found in it (private/loopback).
"""

from __future__ import annotations

import ipaddress
import logging
from pathlib import Path

logger = logging.getLogger("fastrag.serve")


class CountryResolver:
    """Resolves an IP to an ISO-3166 alpha-2 country code, or ``None``.

    Built once per app (see ``app.py``) from
    ``ServeConfig.geoip_database_path`` and closed by the app's lifespan on
    shutdown. Every failure mode — no path configured, the ``geoip2`` extra
    not installed, the database file missing or unreadable, the address not
    present in it, or a private/loopback address that was never going to
    be — degrades silently to ``None`` rather than failing the request. Only
    the "a path was given but couldn't be opened" case is worth an
    operator's attention, and it is logged at most once per resolver, not
    once per request.
    """

    def __init__(self, database_path: Path | str | None) -> None:
        """Open ``database_path``, or stay disabled when it is ``None``."""
        self._reader = None
        self._warned = False
        if database_path is None:
            return
        try:
            import geoip2.database
        except ImportError:
            self._warn(
                f"ServeConfig.geoip_database_path is set ({database_path}) but "
                "the `geoip2` package is not installed; install fastrag's "
                "`geoip` extra to enable country resolution."
            )
            return
        try:
            self._reader = geoip2.database.Reader(str(database_path))
        except OSError as exc:
            self._warn(f"Could not open GeoIP database at {database_path}: {exc}")

    def _warn(self, message: str) -> None:
        if not self._warned:
            logger.warning(message)
            self._warned = True

    def lookup(self, ip_address: str | None) -> str | None:
        """Return an ISO-3166 alpha-2 country code for ``ip_address``, or ``None``."""
        if self._reader is None or ip_address is None:
            return None
        try:
            parsed = ipaddress.ip_address(ip_address)
        except ValueError:
            return None
        if parsed.is_private or parsed.is_loopback:
            return None

        from geoip2.errors import AddressNotFoundError

        try:
            response = self._reader.country(ip_address)
        except AddressNotFoundError:
            return None
        except Exception as exc:  # noqa: BLE001 - any lookup failure degrades to None
            self._warn(f"GeoIP lookup failed for {ip_address}: {exc}")
            return None
        return response.country.iso_code

    def close(self) -> None:
        """Release the database reader. Idempotent; later lookups return ``None``."""
        if self._reader is not None:
            self._reader.close()
            self._reader = None
