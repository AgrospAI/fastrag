# Development setup

Everything below assumes a clone of the repository and runs from the `sdk/`
directory — the package's own root, not the monorepo's. `ui/` and
`dashboard/` are separate projects with their own setup.

```bash
git clone https://github.com/AgrospAI/fastrag.git
cd fastrag/sdk
uv sync --all-groups --all-extras   # or: task setup
```

`uv sync` installs `fastrag` itself in editable mode, so changes under `src/`
are picked up without reinstalling.

`task setup` also installs the git hooks from the repository's
`.pre-commit-config.yaml` (`task hooks` installs only those). On every commit
that touches the SDK they run `ruff check --fix` and `ruff format` on the
staged files and `ty check` on the package, with the versions pinned in
`uv.lock`. A hook that fixes a file stops the commit; stage the fix and commit
again.

## The Compose stack

The SDK ships its own stack so it does not collide with the root stack —
Traefik forwards host port **5433** to PostgreSQL's internal port 5432:

```bash
cp .env.example .env
docker compose up -d
docker compose ps          # STATUS must read "healthy"
```

`docker-compose.yaml` at the `sdk/` root includes three files from
`compose/` — PostgreSQL, Milvus (with its etcd/MinIO backing services and the
Attu console on `:8000`), and Ollama, which is also pre-pulled with
`FASTRAG_CHAT_MODEL` by a one-shot `ollama-pull` service — plus a `docs`
service that serves the live-reloading documentation on `:8001`. Those are
the services the numbered `examples/` need. `.env.example` documents service settings. The numbered scripts accept execution-time endpoint/input/schema/collection
overrides; see {doc}`/guide/examples`. AgrospAI fetching has explicit
CLI/environment configuration.

pgAdmin is available as an opt-in: it is commented out in
`compose/docker-compose.postgres.yaml` (together with its named volume).
Uncomment both to get it on `http://localhost:5050` — `compose/servers.json`
and `compose/pgpass` preconfigure the connection to the stack's PostgreSQL.
`compose/` holds nothing else: the backends this SDK has no code for
(Chroma, Redis, RustFS, SeaweedFS, standalone MinIO) were removed.

The cache and chat-store schemas are self-managed — `PostgresCache` and
`PostgresChatStore` create their own tables on `__aenter__` — so there is no
migration step for the new tree.

## Day to day

```bash
task lint        # ruff check + ruff format --check, whole repo (src tests examples docs)
task format      # ruff check --fix + ruff format
task typecheck   # ty check src examples
task index       # uv run python examples/05_indexing.py
task serve       # uv run python examples/14_serving.py
task ask         # uv run python examples/15_asking.py
task docs-serve  # live-reloading docs on :8001
task test        # starts isolated SDK services, then runs pytest on the host
task e2e         # the same suite inside Compose — Docker is the only prerequisite
```

Run the numbered `examples/*.py` against the same stack — see
{doc}`/guide/examples`.

## Running the docs locally

```bash
task docs           # one-shot build into docs/build/html, warnings are errors
task docs-serve      # live reload on http://localhost:8001, watching src/ too
task docs-linkcheck  # verify every external link resolves
```

`task docs-docker` builds and serves the documentation image on `:8080` —
useful when a build difference is suspected between a local Sphinx install and
the container's. CI does not use that image: its docs job runs `task
docs-check` directly and uploads `docs/build/html` as a workflow artifact.

For other Sphinx targets, run `task` from `docs/` to list them, then use
`task html`, `task linkcheck`, or another target. `docs/Taskfile.yaml` accepts
`SPHINXBUILD`, `SPHINXOPTS`, `O`, `SOURCEDIR`, and `BUILDDIR` overrides;
for example, `task html -- -W --keep-going` passes extra options to Sphinx.

## See also

{doc}`architecture` for the package map; {doc}`documentation` for where a new
paragraph belongs before you write it.


## Isolated example workflow

Use the SDK-owned service stack and original-main smoke runner to validate all 20
scripts with fixture sources and deterministic protocol providers:

```bash
.venv/bin/python scripts/test_services.py start
.venv/bin/python scripts/test_services.py probe
.venv/bin/python scripts/example_smoke.py
```

This uses PostgreSQL on loopback 55433 and Milvus on 19531, unique owned namespaces
and automatic teardown; it does not modify the parent project's services. Model2Vec
weights and optional Playwright browsers must already be installed. Provider fixture
responses prove protocol behavior, not model quality. See {doc}`/guide/examples`
for individual script selection, browser mode and all runtime override names.
