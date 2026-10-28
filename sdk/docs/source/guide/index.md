# Guide

A `fastrag` pipeline has four stages: **ingest** a source into documents,
**index** them into a vector collection, **retrieve** the ones relevant to a
question, and **query** an answer out of them. **Benchmark** measures the
whole chain against a labelled dataset, and every stage is built from typed,
swappable pieces you construct and pass in — nothing is configured by string
or discovered by import.

Read {doc}`concepts` first regardless of what you came here to build — it
names the four kinds of thing every other page assumes you already
recognise. After that, the pages below follow the pipeline in order, and
each stands on its own once you know the vocabulary:

```{toctree}
:maxdepth: 1

install
quickstart
concepts
ingestion
structure
caching
indexing
retrieval
querying
benchmarking
configuration
customizing
examples
serving/index
```

- **New to the package?** {doc}`install` then {doc}`quickstart` gets a real
  question answered in about 40 lines.
- **Building an ingestion pipeline?** {doc}`ingestion` → {doc}`caching` →
  {doc}`indexing`, in that order — each stage's cache and identity rules
  depend on the one before it.
- **Already indexed, now querying?** {doc}`retrieval` → {doc}`querying`.
- **Comparing configurations?** {doc}`benchmarking` turns "I think 500 is
  better than 800" into a table.
- **Exposing this over HTTP?** {doc}`serving/index` is a separate audience —
  it assumes no Python on the other end at all.
- **None of the built-ins fit?** {doc}`customizing` is the extension story
  for every kind in {doc}`concepts`.

## See also

{doc}`/reference/index` for the generated API; {doc}`/contributing/index` if
the piece you need does not exist yet and you are thinking about adding it.
