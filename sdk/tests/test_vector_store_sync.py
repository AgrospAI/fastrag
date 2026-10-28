"""``VectorStore.sync``: save, then delete what the run did not yield."""

from __future__ import annotations

import io
from unittest.mock import AsyncMock, MagicMock

import pytest

from fastrag.ingestion import Pipeline
from fastrag.ingestion.events import DocumentsPruned, ItemFailed, PipelineEvent
from fastrag.ingestion.logger import ConsoleLogger
from fastrag.ingestion.models import Source
from fastrag.ingestion.steps.base import PipelineStep, SourceStep
from fastrag.ingestion.steps.embedding.base import Embedder
from fastrag.ingestion.steps.parsing import DefaultParser
from fastrag.models import Document
from fastrag.vector_store import VectorStore
from fastrag.vector_store import chroma as chroma_module
from fastrag.vector_store import milvus as milvus_module


class MemoryStore(VectorStore):
    def __init__(self, ids=()):
        super().__init__()
        self.ids = set(ids)
        self.deleted: list[str] = []

    async def ensure_ready(self, dimension):
        pass

    async def existing_ids(self, ids):
        return self.ids.intersection(ids)

    async def upsert(self, documents):
        self.ids.update(document.id for document in documents)

    async def search(self, embedding, *, k, filter=None):
        return []

    async def all_ids(self):
        return set(self.ids)

    async def delete(self, ids):
        self.deleted.extend(ids)
        self.ids.difference_update(ids)


class Embedded(PipelineStep[Document]):
    """Yields embedded documents, optionally reporting one failed item first."""

    def __init__(self, ids, *, fail=False):
        super().__init__()
        self.ids = ids
        self.fail = fail

    def config(self):
        return {}

    async def evaluate(self, context):
        if self.fail:
            await context.emit(
                ItemFailed(step=self.name, uri="memory://broken", error=OSError("x"))
            )
        for id_ in self.ids:
            yield Document(id=id_, source_id="source", content=id_, embedding=[1.0])


async def run(step: PipelineStep[Document]) -> list[PipelineEvent]:
    async with Pipeline(step) as pipeline:
        return [event async for event in pipeline]


def pruned(events: list[PipelineEvent]) -> DocumentsPruned:
    [event] = [e for e in events if isinstance(e, DocumentsPruned)]
    return event


async def test_sync_deletes_documents_the_run_did_not_yield():
    store = MemoryStore({"a", "old", "older"})

    events = await run(store.sync(Embedded(["a", "b"])))

    assert store.ids == {"a", "b"}
    assert store.deleted == ["old", "older"]
    event = pruned(events)
    assert (event.stale, event.deleted, event.failures) == (2, 2, 0)
    assert event.step == "MemoryStore.sync"
    assert event.event_name == "Indexing"


async def test_sync_keeps_stale_documents_after_a_failed_item():
    store = MemoryStore({"a", "old"})

    events = await run(store.sync(Embedded(["a", "b"], fail=True)))

    assert store.ids == {"a", "b", "old"}
    assert store.deleted == []
    event = pruned(events)
    assert (event.stale, event.deleted, event.failures) == (1, 0, 1)


async def test_sync_deletes_nothing_when_the_run_stops_early():
    store = MemoryStore({"old"})

    async for _ in store.sync(Embedded(["a", "b"])):
        break

    assert "old" in store.ids


async def test_sync_requires_a_store_that_lists_and_deletes():
    class AddOnlyStore(MemoryStore):
        all_ids = VectorStore.all_ids
        delete = VectorStore.delete

    with pytest.raises(NotImplementedError, match="AddOnlyStore does not support"):
        await AddOnlyStore().sync(Embedded(["a"])).run()


async def test_sync_keeps_documents_skip_existing_reports_as_cached():
    """Already indexed chunks bypass embedding but still count as yielded."""

    class TextSource(SourceStep):
        def config(self):
            return {}

        async def evaluate(self, context):
            for name in ("one", "two"):
                yield Source(
                    id=name,
                    uri=f"memory://{name}",
                    content=name.encode(),
                    content_type="text/plain",
                    content_hash=name,
                )

    class TestEmbedder(Embedder):
        def config(self):
            return {}

        @property
        def dimension(self):
            return 1

        @property
        def model_id(self):
            return "test"

        async def embed(self, texts):
            return [[1.0] for _ in texts]

    store = MemoryStore({"old"})
    indexed = store.sync(
        TestEmbedder()(DefaultParser()(TextSource())), skip_existing=True
    )

    first = pruned(await run(indexed))
    assert (first.stale, first.deleted) == (1, 1)
    assert len(store.ids) == 2 and "old" not in store.ids
    indexed_ids = set(store.ids)

    second = pruned(await run(indexed))
    assert (second.stale, second.deleted) == (0, 0)
    assert store.ids == indexed_ids


async def test_console_logger_reports_the_sync_outcome():
    store = MemoryStore({"a", "old"})
    stream = io.StringIO()

    await ConsoleLogger(Pipeline(store.sync(Embedded(["a"]))), stream=stream).run()

    assert "Sync: MemoryStore.sync: deleted 1 stale documents" in stream.getvalue()

    stream = io.StringIO()
    store.ids.add("old")
    await ConsoleLogger(
        Pipeline(store.sync(Embedded(["a"], fail=True))), stream=stream
    ).run()

    assert (
        "Sync: MemoryStore.sync: kept 1 stale documents because 1 items failed"
        in stream.getvalue()
    )


async def test_chroma_lists_and_deletes_ids(tmp_path, monkeypatch):
    pytest.importorskip("chromadb")
    from fastrag.vector_store import ChromaVectorStore

    monkeypatch.setattr(chroma_module, "ID_PAGE_SIZE", 2)
    async with ChromaVectorStore(path=str(tmp_path)) as store:
        assert await store.all_ids() == set()
        await store.ensure_ready(1)
        await store.upsert(
            [
                Document(id=id_, source_id="s", content=id_, embedding=[1.0])
                for id_ in ("a", "b", "c")
            ]
        )
        assert await store.all_ids() == {"a", "b", "c"}
        await store.delete(["a", "missing"])
        assert await store.all_ids() == {"b", "c"}


async def test_milvus_pages_ids_in_primary_key_order(monkeypatch):
    monkeypatch.setattr(milvus_module, "ID_PAGE_SIZE", 2)
    store = milvus_module.MilvusVectorStore(url="http://milvus.invalid", collection="c")
    client = MagicMock()
    client.has_collection = AsyncMock(return_value=True)
    client.query = AsyncMock(
        side_effect=[[{"id": "a"}, {"id": "b"}], [{"id": "c"}, {"id": "d"}], []]
    )
    client.delete = AsyncMock()
    store._client = client

    assert await store.all_ids() == {"a", "b", "c", "d"}
    filters = [call.kwargs["filter"] for call in client.query.call_args_list]
    assert filters == ['id != ""', 'id > "b"', 'id > "d"']
    assert client.query.call_args.kwargs["iterator"] == "True"
    assert client.query.call_args.kwargs["consistency_level"] == "Strong"

    await store.delete(["a", "b"])
    client.delete.assert_awaited_once_with(collection_name="c", ids=["a", "b"])

    client.has_collection = AsyncMock(return_value=False)
    assert await store.all_ids() == set()
