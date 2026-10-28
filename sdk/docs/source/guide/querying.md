# Querying

Turn a retrieved set of documents and a question into a generated, grounded
answer. `fastrag.query` composes a `Retriever`, a `PromptTemplate`, and a
`ChatModel` into one `QueryEngine` with an `ask` and a `stream` surface.
`examples/12_querying.py` and `examples/13_streaming.py` run the same three
pieces without the `QueryEngine` wrapper — retrieve, render, then
`llm.invoke`/`llm.stream` directly — so there is no standalone runnable
example of `QueryEngine.ask` itself to point `literalinclude` at; the
snippet below is illustrative:

```python
import asyncio

from fastrag.query import PromptTemplate, QueryEngine
from fastrag.query.llm import OpenAIChatModel


async def main():
    engine = QueryEngine(
        retriever=retriever,  # any fastrag.retrieval Retriever
        llm=OpenAIChatModel(
            model="llama3.2:3b",
            api_key="ollama",
            base_url="http://localhost:11434/v1",
        ),
        prompt=PromptTemplate(
            system="You are an expert on data spaces. Cite sources as [n].",
            template="Documents:\n{context}\n\nQuestion: {question}\n\nAnswer:",
        ),
    )

    answer = await engine.ask("What is AgrospAI?")
    print(answer.text)
    for source in answer.sources:
        print(source.metadata.get("source_uri"))


asyncio.run(main())
```

## ask vs stream

- **`ask(question)`** — returns an `Answer` with `text` and `sources` (the
  retrieved documents cited, in order).
- **`stream(question)`** — yields a `SourcesChunk` first, then `TokenChunk`s
  as the model generates, then one `DoneChunk`. This is the surface the
  serving layer uses for SSE.

## The prompt

`PromptTemplate.template` must contain both `{context}` and `{question}`; it
is validated at construction (a missing placeholder raises
`ConfigurationError` rather than silently answering from the model's own
knowledge). `render(context=..., question=...)` numbers the documents by
source and builds the messages: each page gets one `[n]`, followed by its
retrieved chunks in page order, so an answer cites a page once rather than
once per chunk. Each page of a PDF is its own source, linked by the
`#page=N` URL its parser records as `source_url`. `group_sources(context)`
returns those groups, `[n]` being `groups[n - 1]`. `DEFAULT_RAG_PROMPT` is the shipped default. It asks for
citations as `[n]`, once per paragraph, and for one language version when
several documents say the same thing.

## Chat models

`ChatModel` is the message-to-answer contract; `OpenAIChatModel` speaks any
OpenAI-protocol endpoint (OpenAI, Ollama, vLLM, llama.cpp), selected by
`base_url`. `ChatModel.invoke(messages)` returns `str`; `stream(messages)`
yields tokens.

## See also

{doc}`retrieval` for the `Retriever` you pass in; {doc}`serving/index` for
what `QueryEngine.stream` feeds; `examples/12_querying.py` and
`examples/13_streaming.py` for the retrieve/render/invoke steps
`QueryEngine.ask`/`stream` wrap, run end to end.
