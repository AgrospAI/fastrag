# Documentation standard

Four rules govern where a sentence about `fastrag` goes. Each is mechanical
enough to settle an argument about where a paragraph belongs.

## Audience routing

Three audiences, resolved by one question — _what does the reader need to
change?_

| Audience         | Has                                      | Needs                                         | Home                                         |
| ---------------- | ---------------------------------------- | --------------------------------------------- | -------------------------------------------- |
| **User**         | `pip install fastrag-ai`; public names only | To build a pipeline and swap its parts        | {doc}`/guide/index`, {doc}`/reference/index` |
| **API consumer** | An HTTP endpoint; no Python at all       | Endpoints, SSE framing, status codes          | {doc}`/guide/serving/index`                  |
| **Contributor**  | A clone and a Compose stack              | Layering, invariants, why it is not otherwise | {doc}`/contributing/index`                   |

Routing test, applied per paragraph:

- Does a reader need this to **use** an installed `fastrag`? → `guide/`
- Is it the signature, contract, or choice-between-siblings of one symbol? →
  **docstring**, surfaced in `reference/`
- Does it name a `_`-prefixed module, a table's DDL, a lock key, a test
  strategy, or a rejected alternative? → `contributing/`

Anything that fails all three is roadmap, and belongs in `BACKLOG.md`, not on
the site: published docs describe only what ships.

**A worked example.** "Start PostgreSQL" splits cleanly. {doc}`/guide/install`
states the requirement — a reachable PostgreSQL, a DSN, and an applied schema
— and nothing about how the reader gets one. {doc}`/contributing/setup` owns
_this repo's_ stack: `docker compose up -d`, port 5433 to dodge the root
stack's collision, `.env.example`. Same subject, two pages, no overlap,
because the two readers can do different things.

## Docstrings own symbols; Markdown owns everything larger

| Docstrings own                            | Markdown owns                                      |
| ----------------------------------------- | -------------------------------------------------- |
| What one symbol is and guarantees         | Tasks spanning several symbols                     |
| Arguments, returns, raises                | Concepts and vocabulary                            |
| **When to choose this over its siblings** | Comparison across a whole subsystem                |
| One short runnable snippet                | Long, staged, or multi-file examples               |
| Contract an implementer must honour       | Rationale, rejected alternatives, decision history |

Two consequences:

1. **Markdown never restates a signature, a default, or a field list.** It
   cross-references: `` {py:class}`~fastrag.ingestion.policy.Policy` ``. A
   default copied into prose is a default that will be wrong within two
   releases, and `nitpicky` catches a broken reference while it cannot catch a
   stale number.
2. **A docstring never explains the pipeline.** If the sentence needs two
   class names to make sense, it is a guide sentence.

## The summary ladder

Summaries are the highest-priority element of the docs, because they are the
only part most readers see. Four levels; each must stand alone, and each must
say something the level below does not repeat.

| Level | Where                  | Budget                      | Must answer                                |
| ----- | ---------------------- | --------------------------- | ------------------------------------------ |
| 1     | `README.md`            | ≤ 60 words + one code block | What is it, who is it for, how do I start  |
| 2     | `docs/source/index.md` | ≤ 80 words + 4 cards        | Which of these four things am I here to do |
| 3     | Page lede              | 1–3 sentences               | What will I be able to do after this page  |
| 4     | Docstring summary line | One sentence, ≤ 79 chars    | What is this symbol, in isolation          |

Rules that apply at every level:

- **Inverted pyramid.** The first sentence carries the conclusion. Context,
  qualifications, and edge cases follow it, never precede it.
- **No heading immediately follows a heading.** Every heading is followed by
  prose that makes the section useful to someone who stops reading there.
- **No "In this page you will find…" lists.** They restate the table of
  contents, which the theme already renders in the sidebar.
- **Level 4 is the API index.** Autosummary tables render the first line and
  nothing else, so the summary line must survive being read on its own, next
  to forty siblings.

## Page and docstring templates

Every guide page:

```markdown
# <Task, as a noun phrase>

<Lede: 1–3 sentences. What this gets you, and when to use it.>

<Smallest complete runnable example.>

<Sections that vary it: the axis, then why you would move along it.>

## See also

<Links to the neighbouring task and the reference page.>
```

Every public symbol:

```python
def upsert(self, documents: Sequence[Document]) -> None:
    """One-line summary, imperative, ends in a period.

    Body: the contract. What an implementer must guarantee, what a caller may
    rely on, and what breaks if either is violated. For an implementation
    class, the body says when to choose it over its siblings.

    Args:
        documents: What it is, not what its type is — the annotation already
            says the type, and ``autodoc_typehints = "description"`` renders it.

    Raises:
        VectorStoreError: The condition, stated as a condition.

    .. code-block:: python

        store.upsert(chunks)   # runnable as written
    """
```

Dataclass fields take attribute docstrings, the pattern
`RecursiveCharacterTextSplitter` already sets —
they render as field documentation and stay next to the field.

## Checks a human still has to make

Two things a build cannot verify for itself:

- **The lede test.** Every page's first paragraph, read alone with no title,
  tells you whether to keep reading.
- **The sibling test.** Every implementation class docstring says when to
  pick it over its siblings. `RecursiveCharacterTextSplitter` passes; use it as the model.

## What the build enforces

Every rule below runs on each push and pull request, in
`.github/workflows/sdk.yml`.

```bash
# The build is clean, warnings are errors, cross-references resolve.
task docs-check

# No prose ships inside the package or the wheel. The wheel side is a
# zipfile call, not a grep: `python -m zipfile -l` pads its columns past the
# filename, so `grep '\.md$'` never matches even when a .md file is there.
find src -name '*.md' | grep . && exit 1
uv build --wheel -o dist/
python3 -c "import glob, zipfile; \
    names = zipfile.ZipFile(glob.glob('dist/*.whl')[0]).namelist(); \
    assert not [n for n in names if n.endswith('.md')]; \
    assert any(n.endswith('py.typed') for n in names)"

# No user page names a private module.
grep -rnE 'fastrag(\.[a-z_]+)*\._[a-z_]+|`_[a-z_]+\.py`' docs/source/guide && exit 1

# Every symbol carries a docstring in the house style.
task lint
```
