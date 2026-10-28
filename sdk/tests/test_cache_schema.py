"""Bounded identifiers and non-destructive schema-name upgrades."""

import hashlib

import pytest
from conftest import DATABASE_URL, _drop_schema
from sqlalchemy import event, inspect, text
from sqlalchemy.dialects.postgresql import dialect
from sqlalchemy.schema import CreateIndex, CreateTable
from sqlalchemy.sql.elements import conv

from fastrag._sql import make_engine
from fastrag.exceptions import CacheError
from fastrag.ingestion.cache import PostgresCache
from fastrag.ingestion.cache.schema import CacheBase, SourceRecord
from fastrag.ingestion.models import Source
from fastrag.models import Document
from fastrag.query.chat.schema import build_schema as build_chat_schema

COLLISION_SCHEMA = "fastrag_test_1be9af14c1424f56b5cee8fe816e9f83"
SCHEMAS = [
    COLLISION_SCHEMA,
    "fastrag_cache",
    *["s" + hashlib.sha256(str(seed).encode()).hexdigest()[:62] for seed in range(32)],
    "é" * 31 + "s",
    "界" * 21,
]


@pytest.mark.parametrize("schema", SCHEMAS)
@pytest.mark.parametrize("builder", [lambda _schema: CacheBase, build_chat_schema])
def test_compiled_relation_identifiers_are_unique_bounded_and_schema_independent(
    schema, builder
):
    metadata = builder(schema).metadata
    preparer = dialect().identifier_preparer
    relations = []
    for table in metadata.tables.values():
        assert str(CreateTable(table).compile(dialect=dialect()))
        relations.extend([table.name, str(table.primary_key.name)])
        for index in table.indexes:
            assert str(CreateIndex(index).compile(dialect=dialect()))
            compiled = preparer.format_index(index)
            assert compiled == index.name
            assert len(compiled.encode()) <= 63
            relations.append(compiled)
    assert len(relations) == len(set(relations))
    reference = builder("short").metadata
    assert {i.name for t in metadata.tables.values() for i in t.indexes} == {
        i.name for t in reference.tables.values() for i in t.indexes
    }


async def _index_names(store):
    async with store._engine.connect() as connection:
        return await connection.run_sync(
            lambda sync: {
                (table.name, index["name"])
                for table in CacheBase.metadata.tables.values()
                for index in inspect(sync).get_indexes(
                    table.name, schema=store._schema_name
                )
            }
        )


async def test_exact_collision_schema_create_reopen_preserves_payloads():
    # Refuse to use an existing schema: teardown owns only what this test creates.
    store = PostgresCache(url=DATABASE_URL, schema=COLLISION_SCHEMA)
    probe = make_engine(DATABASE_URL)
    try:
        async with probe.connect() as connection:
            assert not await connection.scalar(
                text(
                    "SELECT EXISTS (SELECT 1 FROM pg_namespace WHERE nspname=:schema)"
                ),
                {"schema": COLLISION_SCHEMA},
            )
    finally:
        await probe.dispose()
    try:
        async with store:
            source = Source(
                "s", "https://example.test", b"payload", "text/plain", "hash"
            )
            doc = Document(
                "d", "s", "document", embedding=[0.1, 0.2], metadata={"x": 1}
            )
            await store.put_source_entries("source", [source])
            await store.put_document_entries("document", [doc])
            await store.put_document_entries("empty", [])
            names = await _index_names(store)
            assert len(names) == 7
        async with PostgresCache(url=DATABASE_URL, schema=COLLISION_SCHEMA) as reopened:
            assert await _index_names(reopened) == names
            assert (await reopened.get_source_entries("source"))[
                0
            ].content == b"payload"
            hit = (await reopened.get_document_entries("document"))[0]
            assert (hit.content, hit.embedding, hit.metadata, hit.cached) == (
                "document",
                [0.1, 0.2],
                {"x": 1},
                True,
            )
            assert await reopened.get_document_entries("empty") == []
            assert await reopened.get_document_entries("absent") is None
    finally:
        await store.__aexit__(None, None, None)
        await _drop_schema(COLLISION_SCHEMA)


async def test_legacy_index_names_ready_and_missing_coverage_repaired(cache):
    await cache.put_document_entries("payload", [Document("d", "s", "keep")])
    prep = cache._engine.dialect.identifier_preparer
    schema = prep.quote_schema(cache._schema_name)
    async with cache._engine.begin() as connection:
        for table in CacheBase.metadata.tables.values():
            for index in table.indexes:
                legacy = f"ix_{cache._schema_name}_{table.name}_{next(iter(index.columns)).name}"
                # SQLAlchemy's legacy convention is truncated at compilation time.
                legacy = prep.truncate_and_render_index_name(conv(legacy))
                await connection.execute(
                    text(
                        f"ALTER INDEX {schema}.{prep.format_index(index)} RENAME TO {legacy}"
                    )
                )
    before = await _index_names(cache)
    statements = []

    def record(*args):
        statements.append(args[2])

    event.listen(cache._engine.sync_engine, "before_cursor_execute", record)
    try:
        await cache.__aenter__()
        assert len(statements) == 1
    finally:
        event.remove(cache._engine.sync_engine, "before_cursor_execute", record)
    assert await _index_names(cache) == before
    async with PostgresCache(url=DATABASE_URL, schema=cache._schema_name) as reopened:
        assert await _index_names(reopened) == before
        assert (await reopened.get_document_entries("payload"))[0].content == "keep"
    async with cache._engine.begin() as connection:
        table, name = next(pair for pair in before if pair[0] == "sources")
        await connection.execute(text(f"DROP INDEX {schema}.{prep.quote(name)}"))
    await cache.__aenter__()
    after = await _index_names(cache)
    assert len(after) == len(before)
    assert before - {(table, name)} <= after
    assert ("sources", "ix_sources_last_accessed") in after
    assert (await cache.get_document_entries("payload"))[0].content == "keep"


@pytest.mark.parametrize(
    "expression",
    ["last_accessed) WHERE false", "(coalesce(last_accessed, last_accessed)))"],
)
async def test_partial_or_expression_index_is_not_readiness_coverage(cache, expression):
    index = next(iter(SourceRecord.__table__.indexes))
    prep = cache._engine.dialect.identifier_preparer
    schema = prep.quote_schema(cache._schema_name)
    async with cache._engine.begin() as connection:
        await connection.run_sync(index.drop)
        await connection.execute(
            text(f"CREATE INDEX legacy_incomplete ON {schema}.sources ({expression}")
        )
        assert not await cache._schema_ready(connection)
    await cache.__aenter__()
    assert ("sources", "ix_sources_last_accessed") in await _index_names(cache)


async def test_conflicting_relation_name_is_not_silently_accepted(cache):
    index = next(iter(SourceRecord.__table__.indexes))
    prep = cache._engine.dialect.identifier_preparer
    schema = prep.quote_schema(cache._schema_name)
    async with cache._engine.begin() as connection:
        await connection.run_sync(index.drop)
        await connection.execute(
            text(f"CREATE INDEX ix_sources_last_accessed ON {schema}.sources (uri)")
        )
    with pytest.raises(CacheError):
        await cache.__aenter__()
