"""The ``Retriever`` contract: turning a query into documents."""

from __future__ import annotations

from abc import ABC, abstractmethod

from fastrag.models import Document


class Retriever(ABC):
    """Turns one query into the documents that answer it, best first.

    One method, one required argument. Every other knob — ``k``, filters,
    thresholds — is constructor configuration, so a retriever is a fully
    specified object that can be passed around, swapped, and benchmarked
    without its call sites knowing anything about it.
    """

    @abstractmethod
    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        """Return at most ``k`` documents relevant to ``query``, best first.

        Args:
            query: Natural-language query text.
            k: Maximum number of documents to return.
        """
