"""``ParserRouter``: dispatch each source to the parser registered for it."""

from __future__ import annotations

from fastrag.ingestion.steps.base import RoutedStep
from fastrag.ingestion.steps.parsing.base import Parser
from fastrag.models import Document


class ParserRouter(RoutedStep[Document], Parser):
    """Route each source to the first matching ``.route`` parser, or ``.default``.

    Holds no parsing logic of its own: a source matching no route with no
    default registered is skipped with an ``ItemFailed``.

    .. code-block:: python

        from fastrag.ingestion.steps.parsing import (
            DefaultParser,
            HtmlParser,
            ParserRouter,
            html_filter,
        )

        parser = (
            ParserRouter(cache=cache)
            .route(html_filter, HtmlParser())
            .default(DefaultParser())
        )
    """

    routing_requires_content = False
