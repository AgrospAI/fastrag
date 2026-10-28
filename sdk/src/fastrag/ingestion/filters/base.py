"""``Filter``: the composable predicate base every concrete filter builds on.

A filter is a small predicate over a :class:`~fastrag.ingestion.models.Source`
or :class:`~fastrag.models.Document` that composes with ``&``, ``|`` and ``~``
instead of being written as one growing conditional, e.g.
``html_filter & agrospai_filter``.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T_contra = TypeVar("T_contra", contravariant=True)


class Filter(ABC, Generic[T_contra]):
    """Base abstract class for composable entity filtering."""

    @property
    def header_safe(self) -> bool:
        """Whether matching uses identity/metadata only, never content/vectors.

        Custom filters default to full records. Override only for predicates
        that remain correct on the index-only cache projection.
        """
        return False

    @abstractmethod
    def matches(self, item: T_contra) -> bool:
        """Evaluate whether an entity passes the filter criteria.

        Args:
            item: Item to evaluate (e.g. a ``Source`` or ``Document``).

        Returns:
            True if the item satisfies the condition, False otherwise.
        """
        ...

    def __call__(self, item: T_contra) -> bool:
        """Evaluate the filter against ``item``. Equivalent to :meth:`matches`."""
        return self.matches(item)

    def __and__[U](self: Filter[U], other: Filter[U]) -> Filter[U]:
        """Compose two filters using logical AND."""
        return _AndFilter(self, other)

    def __or__[U](self: Filter[U], other: Filter[U]) -> Filter[U]:
        """Compose two filters using logical OR."""
        return _OrFilter(self, other)

    def __invert__(self) -> Filter[T_contra]:
        """Invert the filter using logical NOT."""
        return _NotFilter(self)


class _AndFilter(Filter[T_contra]):
    """Matches when both composed filters match."""

    def __init__(self, left: Filter[T_contra], right: Filter[T_contra]) -> None:
        self._left = left
        self._right = right

    @property
    def header_safe(self) -> bool:
        """Both predicates must accept cache headers."""
        return self._left.header_safe and self._right.header_safe

    def matches(self, item: T_contra) -> bool:
        """True when both composed filters match ``item``."""
        return self._left.matches(item) and self._right.matches(item)


class _OrFilter(Filter[T_contra]):
    """Matches when either composed filter matches."""

    def __init__(self, left: Filter[T_contra], right: Filter[T_contra]) -> None:
        self._left = left
        self._right = right

    @property
    def header_safe(self) -> bool:
        """Both predicates must accept cache headers."""
        return self._left.header_safe and self._right.header_safe

    def matches(self, item: T_contra) -> bool:
        """True when either composed filter matches ``item``."""
        return self._left.matches(item) or self._right.matches(item)


class _NotFilter(Filter[T_contra]):
    """Matches when the wrapped filter does not."""

    def __init__(self, inner: Filter[T_contra]) -> None:
        self._inner = inner

    @property
    def header_safe(self) -> bool:
        """The wrapped predicate must accept cache headers."""
        return self._inner.header_safe

    def matches(self, item: T_contra) -> bool:
        """True when the wrapped filter does not match ``item``."""
        return not self._inner.matches(item)
