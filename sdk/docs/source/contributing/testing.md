# Testing

The SDK combines deterministic local contract tests with service-backed pipeline
and transport tests. Collect the current inventory rather than relying on a
fixed test count:

```bash
uv run --group dev --all-extras python -m pytest --collect-only -q
```

## Isolated service gate

The SDK-owned test stack has independent volumes and loopback PostgreSQL/Milvus
ports. It avoids the parent project's services and shared production collections.

```bash
uv sync --all-groups --all-extras --locked
uv run python scripts/test_services.py start
uv run python scripts/test_services.py probe
uv run python scripts/test_services.py test
uv run python scripts/test_services.py stop
```

`start` waits for healthy services with a deadline; `probe` performs actual isolated
writes/reads and cleanup. `stop` preserves test volumes. The test command runs the
entire suite with `FASTRAG_DATABASE_URL` and `FASTRAG_MILVUS_URI` targeting that
project, without blanket skips. Tests create unique schemas/collections and clean
them at teardown. Provider-specific tests use local fakes unless they explicitly
exercise the installed local Model2Vec weights.

For direct execution against that already healthy stack:

```bash
FASTRAG_DATABASE_URL=postgresql://postgres@127.0.0.1:55433/fastrag_sdk_test \
FASTRAG_MILVUS_URI=http://127.0.0.1:19531 \
uv run --group dev --all-extras python -m pytest -q --tb=short
```

`task setup` followed by `task test` installs the dependencies, starts the isolated
SDK stack (Docker required), and runs the entire suite. It does not require `.env`.
Services remain running for repeat tests; stop them with the command above.
Compose `task e2e` runs the same discovered tests with its endpoint settings.
Reachability failures are failures with
setup guidance, not evidence of passing behavior. Installed extras and first-use
model/tokenizer caches are prerequisites; `uv sync` alone does not prove every
remote provider works.

Parsed document IDs include their source identity, so identical text from two
URLs retains separate provenance in chunks and cache entries. Parser cache keys
are versioned to regenerate entries created with the earlier content-only IDs.
Reindex existing collections when adopting this identity change.

## Coverage map

The current files divide by contracts and independently testable subsystems:

| Area | Coverage |
| --- | --- |
| Sources/rendering | Local fixture HTTP sitemap/web/file sources, glob selection, redirects, revalidation, dynamic/browser rendering and source aggregation. |
| Parsing/capabilities | Text/HTML/Docusaurus/Docling outputs, parser representation promises, direct/routed compatibility and failure isolation. |
| Normalization | Markdown/RST/HTML exact original Unicode locations, hierarchy, malformed markup, nested structures, exclusions, custom adapters and complete source coverage. |
| Structural chunking | Final-input budgets, exact boundaries, code/table/list fidelity, overlap, metadata/ID determinism, map corruption detection and parent context. |
| Semantic splitting | Legacy chat compatibility, validated chat boundary proposals and deterministic fallback; injected embedding vectors, thresholds, degeneracies and source preservation. |
| Generated/performance | Seeded cross-format source occupancy oracles, nonadditive/nonmonotonic tokenizers, large-node/large-code regressions, call counts and cache hydration. |
| Cache/pipeline/loggers | Fresh/full/header/routed replay, within-parent source order, concurrent-parent completion order, empty buckets, schema/index readiness and content/preview/event output. |
| Embedding/vector transport | Actual local Model2Vec, structural embedding input, Milvus save/search and lossless bounded metadata round trips. |
| Retrieval/query | Dense/BM25/hybrid/reranked results, fusion, prompting and generated/streamed answers with deterministic providers. |
| Serving/CLI | HTTP/SSE/auth/guards/history/chat persistence, expiry, scripted client state and SSE parsing. |
| Benchmark/evaluation | Typed retrieval/generation metrics, datasets, comparison, reproducible same-corpus baseline/structural/refined retrieval and parent-expanded context costs. |
| Published recipes | Execute standalone structural budget, parent-context and custom-adapter guide snippets against the public SDK. |

The fixture retrieval evaluation is a transparent lexical-vector experiment,
not a real-model quality claim. Reproduce it with:

```bash
uv run python scripts/evaluate_structure.py --help
uv run python scripts/profile_structure.py --help
```

Reports retain poorer/tied scores and disclose parent scope differences. Performance
measurements are descriptive; regression tests prefer exact invariants and operation
counts over brittle wall-clock comparisons.

## Local and external validation

Focused normalization/structure/refinement/generated tests run without database
services or provider endpoints. Cache/store/pipeline/example integrations use the
isolated services. Examples add actual script-main, server health/request/shutdown,
scripted CLI and benchmark behavior; importing or compiling a script is insufficient.
Keep exact commands and outcomes in validation reports and distinguish deterministic
compatible provider fixtures from real external inference. Neither fixtures nor a
successful docs build proves production endpoint availability or model quality.

## Adding tests and building documentation

Use `tests/conftest.py` fixture-owned schemas/collections, local HTTP origins and
provider fakes. Assert observable contracts: retained text, independent provenance,
query results, event semantics and destination writes. Choose focused files for
new subsystems rather than obscuring failures in a monolithic layer file. Do not
print credentials or modify production destinations in tests.

```bash
uv run ruff check src tests examples docs app scripts
uv run ruff format --check src tests examples docs app scripts
uv run ty check src examples app scripts
uv run --group docs sphinx-build -b html -n -W --keep-going docs/source docs/build/html
```

The strict build resolves public references, generates API pages and checks
published imports. The standalone recipes test additionally executes their source;
Sphinx rendering alone would not catch behavior errors inside a Python fence.

## See also

{doc}`setup` for service prerequisites; {doc}`architecture` for layering;
{doc}`documentation` for guide/reference ownership and publication checks.
