"""Internal SQLAlchemy plumbing shared by every self-managing PostgreSQL store.

Both :class:`~fastrag.ingestion.cache.postgres.PostgresCache` and
:class:`~fastrag.query.chat.postgres.PostgresChatStore` self-manage their own
schema and deliberately never touch a shared, Alembic-managed ``Base`` (see
each store's own module docstring), so both need the identical engine
construction and error-translation shape. Private to ``fastrag`` —
the one underscore-prefixed top-level module that package-private code may
share across packages; nothing here is public API and it is not documented
in the reference.
"""

from __future__ import annotations

from collections.abc import Callable

from sqlalchemy.exc import DBAPIError, OperationalError, SQLAlchemyError
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

ErrorFactory = Callable[[SQLAlchemyError], Exception]
"""Translates one SQLAlchemy driver failure into a Fastrag exception."""

ASYNC_DRIVER = "postgresql+asyncpg"


def is_transient(error: SQLAlchemyError) -> bool:
    """Return True when ``error`` is connection-level, hence worth a retry.

    A lost connection or an ``OperationalError`` is a reason to try again; a
    malformed statement or a constraint violation is not.
    """
    if isinstance(error, DBAPIError):
        orig = error.orig
        code = getattr(orig, "sqlstate", None) or getattr(orig, "pgcode", None)
        if code in _TRANSIENT_SQLSTATES:
            return True
    return isinstance(error, OperationalError) or (
        isinstance(error, DBAPIError) and error.connection_invalidated
    )


#: Concurrency conflicts Postgres resolves by aborting one transaction, which
#: succeeds when simply run again: deadlock_detected, serialization_failure.
_TRANSIENT_SQLSTATES = frozenset({"40P01", "40001"})


class Transaction:
    """``engine.begin()``, translating every driver failure via ``error_factory``.

    Wrapping the whole block matters: a statement error is raised inside the
    body, where a plain ``engine.begin()`` would let a SQLAlchemy exception
    escape a Fastrag API.
    """

    def __init__(self, engine: AsyncEngine, error_factory: ErrorFactory) -> None:
        self._engine = engine
        self._error_factory = error_factory
        self._context = None

    async def __aenter__(self):
        self._context = self._engine.begin()
        try:
            return await self._context.__aenter__()
        except SQLAlchemyError as error:
            raise self._error_factory(error) from error

    async def __aexit__(self, exc_type, exc, traceback) -> bool:
        assert self._context is not None
        try:
            suppressed = await self._context.__aexit__(exc_type, exc, traceback)
        except SQLAlchemyError as error:
            raise self._error_factory(error) from error

        # The rollback succeeded but the statement did not: translate the error
        # that opened this block, so no SQLAlchemy exception escapes.
        if not suppressed and isinstance(exc, SQLAlchemyError):
            raise self._error_factory(exc) from exc

        return bool(suppressed)


def async_url(url: str) -> str:
    """Return ``url`` with the asyncpg driver spelled out.

    A plain ``postgresql://`` URL is what ``psql`` and every sync tool
    accept, so that is the form Fastrag takes as an argument.
    """
    if url.startswith(f"{ASYNC_DRIVER}://"):
        return url
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", f"{ASYNC_DRIVER}://", 1)
    if url.startswith("postgres://"):
        return url.replace("postgres://", f"{ASYNC_DRIVER}://", 1)
    return url


def make_engine(
    url: str, *, pool_size: int = 5, echo: bool = False, schema: str | None = None
) -> AsyncEngine:
    """Create an async engine, replacing connections closed during idle work.

    ``schema``, when given, is where tables declared without a schema live:
    every statement, DDL included, is rendered against it.
    """
    return create_async_engine(
        async_url(url),
        pool_size=pool_size,
        echo=echo,
        pool_pre_ping=True,
        pool_recycle=300,
        execution_options={"schema_translate_map": {None: schema}}
        if schema is not None
        else {},
    )
