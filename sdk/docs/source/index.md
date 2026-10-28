# fastrag

A Python SDK for the whole life of a RAG system: ingest sources, keep a
vector collection in sync with them as those sources change, query it, and
benchmark the result. `pip install fastrag-ai` gets you typed, swappable pieces
— fetchers, parsers, chunkers, embedders, vector stores, rerankers — composed
into a pipeline you own, not a framework you configure.

```{note}
Want to see a RAG chatbot answering questions end to end — backend,
dashboard, and the widget it's serving — before writing any Python? The
repository root's `README.md` brings up the whole composed product
(`docker compose up -d`, no install beyond Docker) in a few commands. The
guide below is for building your own pipeline with the SDK directly.
```

::::{grid} 1 2 2 2
:gutter: 3

:::{grid-item-card} {octicon}`rocket` Build a pipeline
:link: guide/index
:link-type: doc

Install the package, index a source, ask it a question. Start here.
:::

:::{grid-item-card} {octicon}`plug` Talk to it over HTTP
:link: guide/serving/index
:link-type: doc

`fastrag.serve` is a FastAPI app — `create_app(engine=, chat_store=)` — that
streams grounded answers over SSE, for a client with no Python at all.
:::

:::{grid-item-card} {octicon}`book` Look up a symbol
:link: reference/index
:link-type: doc

Every public class and function, generated from its docstring.
:::

:::{grid-item-card} {octicon}`git-branch` Work on fastrag itself
:link: contributing/index
:link-type: doc

Clone it, run the Compose stack, and find your way around the source.
:::
::::

```{toctree}
:maxdepth: 2
:hidden:

guide/index
guide/serving/index
reference/index
contributing/index
```
