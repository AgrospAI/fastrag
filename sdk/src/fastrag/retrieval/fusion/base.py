"""Backend-independent fusion of ranked document lists."""

from abc import ABC, abstractmethod
from collections.abc import Sequence

from fastrag.models import Document


class RankFusion(ABC):
    """Combine rankings without assuming comparable backend scores."""

    @abstractmethod
    def fuse(self, rankings: Sequence[Sequence[Document]], *, k: int) -> list[Document]:
        """Return at most ``k`` unique documents, best first."""
        ...
