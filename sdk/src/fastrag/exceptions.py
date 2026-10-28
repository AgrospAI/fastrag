"""Exception hierarchy.

Every error raised by the SDK derives from :class:`FastragError`, so a caller
can catch the whole package with one ``except``.  :class:`TransientError` is a
mixin rather than a branch of the tree, so an exception can be both typed by
origin — a fetch failure, an embedding failure — and marked retryable.

:func:`describe_error` is the counterpart for *reporting* one: not every
exception that reaches a failure handler is one of ours, and a third-party
exception raised with no arguments formats as an empty string.
"""

from __future__ import annotations


class FastragError(Exception):
    """Base class for every error raised by Fastrag."""


class TransientError(FastragError):
    """Mixin marking an error worth retrying.

    Only exceptions carrying this mixin are retried by :class:`~fastrag.ingestion.policy.Policy`:
    a timeout is transient, a malformed document is not.
    """


class ConfigurationError(FastragError):
    """Invalid arguments, a missing optional dependency, or a bad template.

    Always raised as early as possible — at construction, not at first use.
    """


class SourceError(FastragError):
    """A source could not be turned into documents."""


class FetchError(SourceError):
    """A resource could not be fetched."""


class TransientFetchError(FetchError, TransientError):
    """A fetch failed for a reason that may succeed on retry (network, 5xx)."""


class ParseError(SourceError):
    """A resource could not be parsed."""


class TransientParseError(ParseError, TransientError):
    """Parsing failed because of a temporary model or network failure."""


class ChunkingError(FastragError):
    """A document could not be chunked."""


class ServiceError(FastragError):
    """A client for something outside the process failed."""


class EmbeddingError(ServiceError):
    """An embedding request failed."""


class TransientEmbeddingError(EmbeddingError, TransientError):
    """An embedding failed for a reason that may succeed on retry."""


class ChatError(ServiceError):
    """A chat completion request failed."""


class TransientChatError(ChatError, TransientError):
    """A chat completion failed for a reason that may succeed on retry."""


class ContextLengthError(ChatError):
    """The prompt exceeded the model's context window.

    Deliberately **not** a :class:`TransientError`: retrying an over-long
    prompt produces the same failure, so ``Policy`` must not retry it.
    """


class RerankError(ServiceError):
    """A rerank request failed."""


class TransientRerankError(RerankError, TransientError):
    """A rerank failed for a reason that may succeed on retry."""


class VectorStoreError(ServiceError):
    """A vector store operation failed."""


class TransientVectorStoreError(VectorStoreError, TransientError):
    """A vector store operation failed for a reason that may succeed on retry."""


class CacheError(FastragError):
    """The step cache could not be read or written."""


class TransientCacheError(CacheError, TransientError):
    """A cache operation failed on lock contention or a lost connection."""


class ChatStoreError(FastragError):
    """A chat store operation (session, chat, or message) could not be completed."""


class TransientChatStoreError(ChatStoreError, TransientError):
    """A chat store operation failed on lock contention or a lost connection."""


def describe_error(error: BaseException) -> str:
    """Format ``error`` for a human, never as an empty string.

    ``f"{error}"`` is the obvious way to render the ``error`` on an
    :class:`~fastrag.ingestion.events.ItemFailed` or
    :class:`~fastrag.ingestion.events.StepFailed`, and it renders an
    exception raised with no arguments — ``TimeoutError()``,
    ``asyncio.CancelledError()``, a bare third-party sentinel — as nothing at
    all, turning a real failure into a blank line. This falls back to the
    exception's type name in that case, and prefixes it otherwise, so a
    failure always says at minimum *what kind* of failure it was.

    .. code-block:: python

        describe_error(TimeoutError())  # -> "TimeoutError"
        describe_error(ParseError("bad xlsx"))  # -> "ParseError: bad xlsx"

    Args:
        error: The exception to describe.

    Returns:
        ``"<type>: <message>"``, or just ``"<type>"`` when the exception
        carries no message.
    """
    message = str(error).strip()
    name = type(error).__name__
    return f"{name}: {message}" if message else name
