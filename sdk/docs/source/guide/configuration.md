# Configuration

Every credential, endpoint, and model identifier is a constructor argument.
Nothing under `fastrag` reads an environment variable or a `.env` file on
your behalf — {doc}`serving/configuration` is the same rule applied to the
serving app's own config object.

```python
import os

embedder = OpenAIEmbedder(
    model="text-embedding-3-small",
    api_key=os.environ["OPENAI_API_KEY"],
)
```

Reading configuration from the environment is the caller's job, done at the
call site — which is also why it stays visible: nothing is hidden behind a
settings object constructed somewhere else in the program.

## Extra model request parameters

`OpenAIVisionParser`, `OpenAIChatModel`, `OpenAIEmbedder`, and
`CohereReranker` accept `extra_body`, a mapping of additional JSON fields
merged into every model request. These fields override matching defaults.
For example, disable thinking for your Qwen vision model:

```python
from fastrag.ingestion.steps.parsing import OpenAIVisionParser

vision = OpenAIVisionParser(
    model="Qwen3.8",
    api_key=os.environ["VISION_API_KEY"],
    base_url=os.environ["VISION_BASE_URL"],
    extra_body={
        "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False},
    },
)
```

The fields are sent directly in the JSON body, without an `extra_body`
wrapper. Chat parameters apply to both complete and streamed answers.
Vision parsing and embedding include these parameters in their cache
identity, so different settings do not reuse the same cached output.
`Model2VecEmbedder` runs locally and has no HTTP request body.

## `Policy`: retries, timeouts, concurrency

```python
from fastrag.ingestion import Policy

Policy(retries=2, timeout=60.0, concurrency=8, batch_size=64, on_error="skip")
```

Every ingestion step runs inside a shared executor, so `Policy` is inherited
by every step, including custom ones ({doc}`customizing`). `on_error="skip"`
is the default because one malformed document in ten thousand should not
abort a long index — a skipped item is reported as an `ItemFailed` event and
shows up in the `ConsoleLogger`'s `errors` column, never dropped silently.
Retries apply only to exceptions marked
{py:class}`~fastrag.exceptions.TransientError` — a timeout is worth retrying,
a malformed document never is, regardless of `retries`.

## Events

```{literalinclude} /../../examples/06_logging.py
:language: python
:start-after: "docs:configuration-events-start"
:end-before: "docs:configuration-events-end"
:dedent:
```

Progress reporting is an event stream, not a logging callback. Evaluating a
`Pipeline` yields lifecycle events (`StepStarted`, `StepCompleted`,
`ItemFailed`) and domain events that carry the item they describe
(`DocumentFetched`, `DocumentParsed`, `DocumentChunked`, `DocumentEmbedded`,
`DocumentSaved`) — each with a trustworthy `cached` flag read off the item
itself. `ConsoleLogger` renders a per-fetcher table of new/cached/errors.

## Exceptions

Every error `fastrag` raises derives from
{py:exc}`~fastrag.exceptions.FastragError`, so one `except` catches the
whole package. {py:exc}`~fastrag.exceptions.TransientError` is a mixin
rather than a branch of the tree — an exception is typed by _where_ it
happened (`FetchError`, `EmbeddingError`, `ChatError`, `ChatStoreError`,
…) independently of _whether retrying could help_, so a
`TransientFetchError` is both a `FetchError` and retryable, while a plain
`FetchError` is not. {py:exc}`~fastrag.exceptions.ConfigurationError` is
always raised as early as possible — at construction, never at first use —
which is why a missing extra or a bad `PromptTemplate` fails immediately
rather than mid-run.

## See also

{doc}`ingestion` for the steps `Policy` wraps; {doc}`customizing` for
writing a class that inherits all of this for free.
