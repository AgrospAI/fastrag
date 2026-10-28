"""Serving: a FastAPI app that persists chats and streams grounded answers.

The public surface is exactly :func:`create_app`, :class:`ServeConfig`,
:func:`cleanup_expired`, :class:`CleanupReport`, and :class:`CLI`. Everything
else is private.

Importing this package does not require FastAPI or Typer: only
:func:`create_app` needs FastAPI and only :class:`CLI` needs Typer, and a
missing ``serving`` extra surfaces as a
:class:`~fastrag.exceptions.ConfigurationError` naming the install command —
never a bare ``ImportError``.
"""

from __future__ import annotations

from typing import Any

from fastrag.exceptions import ConfigurationError
from fastrag.serve.cleanup import CleanupReport, cleanup_expired
from fastrag.serve.cli import CLI
from fastrag.serve.config import ServeConfig

__all__ = [
    "CLI",
    "CleanupReport",
    "ServeConfig",
    "cleanup_expired",
    "create_app",
]

_MISSING_EXTRA = (
    'create_app needs the "serving" extra: pip install "fastrag-ai[serving]"'
)


def __getattr__(name: str) -> Any:
    """Resolve the public names, importing the extras-gated ones on demand.

    ``CLI`` is the threshold for Typer, ``create_app`` for FastAPI. Every
    lazily-imported name here is a robust, direct import of its module (no
    fromlist), so a name can never recurse on itself.
    """
    if name == "create_app":
        try:
            from fastrag.serve.app import create_app
        except ImportError as error:
            raise ConfigurationError(_MISSING_EXTRA) from error
        return create_app
    if name == "CLI":
        try:
            from fastrag.serve.cli import CLI as _cli
        except ImportError as error:
            raise ConfigurationError(_MISSING_EXTRA) from error
        return _cli
    if name in __all__:
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Report the public names."""
    return sorted(__all__)
