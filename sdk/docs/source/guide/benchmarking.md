# Benchmarking

Score a retrieval or query pipeline against a labelled QA dataset, and
compare runs. `fastrag.benchmark` loads a `QADataset`, runs every sample
through a `Benchmark` against a `Retriever` (retrieval metrics) or a
`QueryEngine` (generation and operational metrics too), and puts several runs
side by side with `Comparison`.

`examples/16_benchmarking.py` demonstrates a chunk-size sweep, scored and saved
in a loop. Give each configuration its own collection when comparing runs: a
cache fingerprint does not change a store's collection automatically. The script
opens `<FASTRAG_COLLECTION>_recursive_<size>` explicitly per configuration and
uses its matching retriever. Runtime size/dataset/result overrides and the owned
local smoke workflow are documented in {doc}`examples`:

```{literalinclude} /../../examples/16_benchmarking.py
:language: python
:start-after: "docs:benchmarking-run-start"
:end-before: "docs:benchmarking-run-end"
:dedent:
```

`metrics=[...]` takes the typed `Metric` objects directly (`Recall(k=5)`,
`Precision(k=5)`, …) — there is no string-shorthand form. Metrics read a
retrieved document's `score`/`rank`/`source_uri` from its `metadata` (the
keys {doc}`retrieval`'s `VectorStore.search` stamps).

## Datasets

`QADataset` and `EvalSample` model one question with its expected contexts and
answers. `from_records` builds one from rows; loading `.yaml` / `.parquet`
files and `from_huggingface` raise `ConfigurationError` naming
`pip install "fastrag-ai[datasets]"` until that extra's loaders ship (see
`BACKLOG.md`).

## Targets and metrics

- **`BenchmarkTarget = Retriever | QueryEngine`.** A bare `Retriever` is
  scored with retrieval metrics; a `QueryEngine` adds generation metrics
  (`Faithfulness`, `AnswerRelevancy`, `Correctness`, `TokenOverlap`) and
  operational ones (`Latency`, `TokenUsage`, `Cost`). A judge-backed metric
  refuses a bare-`Retriever` target.
- **`Comparison`** merges several saved `BenchmarkResult`s into one table.

## See also

`examples/16_benchmarking.py` shows indexing, scoring, saving and comparison;
{doc}`structure` explains the structural alternative and its budget guarantees.
