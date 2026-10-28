# Contributing

Working on `fastrag` itself, rather than building with it — if you are
looking for how to _use_ the package, {doc}`/guide/index` is the other door.

```{toctree}
:maxdepth: 1

setup
architecture
cache-performance
cache-retrieval
conventions
documentation
serving-internals
frontend-types
testing
decisions
```

## Where to start

1. **{doc}`setup`** — clone, bring up the Compose stack, apply the schema,
   run the day-to-day `task` commands.
2. **{doc}`architecture`** — the package map, the dependency direction
   between layers, and the `_`-prefixed private-module convention, so you
   know where a change belongs before you make it.
3. **{doc}`conventions`** — the rules that hold regardless of which module
   you touch: why the SDK has no CLI, which methods are `async`, how
   naming works, what counts as public API.

Three narrower pages round it out: **{doc}`serving-internals`** for the parts
of `fastrag.serve` that only a contributor needs — the layout, the
advisory-lock design, the append-not-truncate regenerate —
**{doc}`frontend-types`** for how the OpenAPI document and the generated
TypeScript that `dashboard/` and `ui/registry/` consume stay in sync with
those same Pydantic models, and **{doc}`testing`** for what is tested today
and how to add to it.

**{doc}`documentation`** is the standard this whole site is written against:
read it before adding a page, and use it to settle where new content
belongs. **{doc}`decisions`** is the append-only log of choices that already
have an answer — check it before reopening one.

`BACKLOG.md` in the repository root, not part of this site, tracks everything
that is intentionally not built yet; a docs page describes only what ships.

## See also

`CONTRIBUTING.md` in the repository root for the short version: clone,
install, test, and where the rest of this section lives.
