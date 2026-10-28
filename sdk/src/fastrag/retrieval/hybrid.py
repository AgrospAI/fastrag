"""Composable hybrid retrieval with replaceable ranking fusion."""

import asyncio
from collections.abc import Sequence

from fastrag.exceptions import ConfigurationError
from fastrag.models import Document
from fastrag.retrieval.base import Retriever
from fastrag.retrieval.fusion import RankFusion, ReciprocalRankFusion


class HybridRetriever(Retriever):
    """Fetch candidates from each retriever concurrently, then fuse their ranks."""

    def __init__(
        self,
        *,
        retrievers: Sequence[Retriever],
        fusion: RankFusion | None = None,
        candidate_k: int = 40,
    ) -> None:
        """Configure the branches, fusion algorithm and per-branch candidate count."""
        if not retrievers:
            raise ConfigurationError("HybridRetriever needs at least one retriever")
        if candidate_k < 1:
            raise ConfigurationError("candidate_k must be at least 1")
        self.retrievers = tuple(retrievers)
        self.fusion = fusion if fusion is not None else ReciprocalRankFusion()
        self.candidate_k = candidate_k

    async def retrieve(self, query: str, *, k: int = 5) -> list[Document]:
        """Fuse the candidate rankings down to ``k`` documents."""
        if k < 1:
            raise ConfigurationError("retrieval k must be at least 1")
        async with asyncio.TaskGroup() as group:
            tasks = [
                group.create_task(retriever.retrieve(query, k=max(k, self.candidate_k)))
                for retriever in self.retrievers
            ]
        return self.fusion.fuse([task.result() for task in tasks], k=k)
