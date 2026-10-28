"""Execution policy controlling operational constraints for pipeline steps."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Any, Literal

from fastrag.exceptions import ConfigurationError

#: Failure handling strategy a step's evaluator applies to a raised exception.
OnError = Literal["skip", "raise", "retry"]


class _Unset:
    """Sentinel marking a ``Policy`` field the caller did not pass."""

    __slots__ = ()

    def __repr__(self) -> str:
        return "UNSET"


_UNSET: Any = _Unset()


@dataclass(frozen=True, slots=True)
class Policy:
    """Execution policy controlling operational constraints for pipeline steps.

    Attributes:
        retries: Maximum number of execution retries on failure.
        backoff: Initial exponential backoff delay in seconds.
        max_backoff: Ceiling delay for exponential backoff in seconds.
        timeout: Maximum execution duration per item or batch in seconds.
        on_error: Failure handling strategy (``"skip"``, ``"raise"``,
            ``"retry"``).
        concurrency: Max concurrent worker tasks.
        batch_size: Processing batch size for batched transforms.
        cache_batch_size: Cache lookup window, independent of transform concurrency.
        recreate_cache: Skip cached results and overwrite them with freshly
            recomputed values.

    A step's policy is layered over its parent's (the pipeline's, or an
    enclosing router/aggregator's) with :meth:`merge`: only the fields the
    caller *explicitly* passed override the parent; the rest are inherited.
    """

    retries: int = _UNSET
    backoff: float = _UNSET
    max_backoff: float = _UNSET
    timeout: float = _UNSET
    on_error: OnError = _UNSET
    concurrency: int = _UNSET
    batch_size: int = _UNSET
    cache_batch_size: int = _UNSET
    recreate_cache: bool = _UNSET
    _explicit: frozenset[str] = field(
        default=frozenset(), init=False, repr=False, compare=False
    )

    def __post_init__(self) -> None:
        """Fill unset fields with defaults, then validate."""
        explicit: set[str] = set()
        for spec in fields(self):
            if spec.name == "_explicit":
                continue
            if getattr(self, spec.name) is _UNSET:
                default = _DEFAULTS[spec.name]
                object.__setattr__(self, spec.name, default)
            else:
                explicit.add(spec.name)
        object.__setattr__(self, "_explicit", frozenset(explicit))
        if self.concurrency < 1:
            raise ConfigurationError("Policy.concurrency must be at least 1")
        if self.retries < 0:
            raise ConfigurationError("Policy.retries cannot be negative")
        if self.batch_size < 1:
            raise ConfigurationError("Policy.batch_size must be at least 1")
        if self.cache_batch_size < 1:
            raise ConfigurationError("Policy.cache_batch_size must be at least 1")

    def merge(self, base: Policy | None) -> Policy:
        """Return this policy layered over ``base``.

        Fields this policy set explicitly win; every other field is taken
        from ``base``. The result's explicit set is the union of both, so a
        chain of merges keeps resolving correctly.

        Args:
            base: The inherited (parent) policy, or ``None`` for no parent.
        """
        if base is None or base is self:
            return self
        values = {
            name: getattr(self if name in self._explicit else base, name)
            for name in _DEFAULTS
            if name in self._explicit or name in base._explicit
        }
        return Policy(**values)

    def delay_for(self, attempt: int) -> float:
        """Return the capped exponential backoff delay before ``attempt``.

        Args:
            attempt: The retry attempt number, 1 for the first retry.

        Returns:
            ``backoff * 2 ** (attempt - 1)``, capped at ``max_backoff``.
        """
        return min(self.backoff * (2 ** (attempt - 1)), self.max_backoff)


_DEFAULTS: dict[str, Any] = {
    "retries": 3,
    "backoff": 1.0,
    "max_backoff": 10.0,
    "timeout": 30.0,
    "on_error": "skip",
    "concurrency": 8,
    "batch_size": 16,
    "cache_batch_size": 128,
    "recreate_cache": False,
}
