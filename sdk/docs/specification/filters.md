> **Historical design record.** Written before implementation, as the
> pre-build design notes for composable filtering. It lives outside `docs/source/`, so
> it is never built into the published site and is not maintained in
> lockstep with the code. Most of what it describes shipped close to as
> written; where it and the code disagree, the code and `docs/source/` win —
> see `.agent/PENDING.md` section D for the pieces it names that were never
> built.

```python
class Filter(ABC, Generic[T_contra]):
    """Base abstract class for composable entity filtering."""

    @abstractmethod
    def matches(self, item: T_contra) -> bool:
        """Evaluates whether an entity passes the filter criteria.

        Args:
            item: Item to evaluate (e.g. Source or Document).

        Returns:
            True if the item satisfies the condition, False otherwise.
        """
        ...

    def __call__(self, value: T) -> bool:
        """Evaluate the filter against a value."""

    def __and__(self, other: Filter[T_contra]) -> Filter[T_contra]:
        """Composes two filters using logical AND."""

    def __or__(self, other: Filter[T_contra]) -> Filter[T_contra]:
        """Composes two filters using logical OR."""

    def __invert__(self) -> Filter[T_contra]:
        """Inverts the filter using logical NOT."""


class MediaTypeFilter(Filter[Source | Document]):
    """Filters sources or documents by MIME media type."""

    def __init__(
        self,
        include_types: Sequence[str] | None = None,
        exclude_types: Sequence[str] | None = None,
    ) -> None:
        """Initializes MediaTypeFilter.

        Args:
            include_types: Allowed MIME types (e.g. ["text/html"]).
            exclude_types: Disallowed MIME types.
        """
        ...

    def matches(self, item: Source | Document) -> bool: ...


class URLFilter(Filter[Source | Document]):
    """Filters sources or documents by URI regex patterns."""

    def __init__(
        self,
        include_patterns: Sequence[str] | None = None,
        exclude_patterns: Sequence[str] | None = None,
    ) -> None:
        """Initializes URLFilter.

        Args:
            include_patterns: Regex patterns an ID must match.
            exclude_patterns: Regex patterns an ID must not match.
        """
        ...

    def matches(self, item: Source | Document) -> bool: ...


class ContentFormatFilter(Filter[Document]):
    """Filters documents by metadata["content_format"].

    Unlike MediaTypeFilter, a document carrying no recorded format is not
    treated as unmatchable -- it is treated as ContentFormat.TEXT, since
    every document that existed before content_format was introduced is
    plain text by definition.
    """

    def __init__(
        self,
        include_formats: Sequence[ContentFormat | str] | None = None,
        exclude_formats: Sequence[ContentFormat | str] | None = None,
    ) -> None:
        """Initializes ContentFormatFilter.

        Args:
            include_formats: Allowed content formats (e.g.
                [ContentFormat.MARKDOWN]).
            exclude_formats: Disallowed content formats.
        """
        ...

    def matches(self, item: Document) -> bool: ...
```