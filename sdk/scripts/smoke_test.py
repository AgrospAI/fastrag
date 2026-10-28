"""Smoke-test an installed fastrag-ai distribution before it is published.

Run it against a built wheel or sdist in a throwaway environment, never the
source tree, so a file the build left out fails here rather than on a user's
machine::

    uv run --isolated --no-project --with dist/*.whl python -I scripts/smoke_test.py 0.0.1

With no extras installed, every module must import except those that need an
optional extra (``serving``, ``geoip``); those may fail only on that extra's
own packages, never on a missing ``fastrag`` module.
"""

from __future__ import annotations

import importlib
import pkgutil
import sys
from importlib.metadata import version
from pathlib import Path

import fastrag

#: Top-level packages an optional extra provides. A module may fail to import
#: on one of these and nothing else.
_EXTRA_PACKAGES = {
    "fastapi",
    "geoip2",
    "prometheus_client",
    "pydantic",
    "starlette",
    "typer",
    "uvicorn",
}


def _fail(message: str) -> None:
    sys.exit(f"smoke test failed: {message}")


def main(expected_version: str | None) -> None:
    """Check the installed distribution's version, contents, and imports."""
    package_dir = Path(fastrag.__file__).parent
    if "site-packages" not in package_dir.parts:
        _fail(f"fastrag was imported from {package_dir}, not an installed copy")

    installed = version("fastrag-ai")
    if fastrag.__version__ != installed:
        _fail(f"__version__ is {fastrag.__version__}, metadata says {installed}")
    if expected_version is not None and installed != expected_version:
        _fail(f"built version {installed} does not match {expected_version}")

    if not (package_dir / "py.typed").is_file():
        _fail("py.typed is missing")
    if any(package_dir.rglob("*.mmdb")):
        _fail("a GeoLite2 database was packaged")

    imported = 0
    for module in pkgutil.walk_packages([str(package_dir)], "fastrag."):
        try:
            importlib.import_module(module.name)
        except ModuleNotFoundError as exc:
            if (exc.name or "").partition(".")[0] not in _EXTRA_PACKAGES:
                _fail(f"importing {module.name}: {exc}")
        else:
            imported += 1

    print(f"fastrag-ai {installed}: {imported} modules imported from {package_dir}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
