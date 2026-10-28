"""Sphinx configuration for the fastrag documentation site."""

import importlib.metadata
import inspect
import sys
from pathlib import Path

# Private, but there is no public alternative: this sentinel is how autodoc
# marks a `__slots__`-declared attribute internally. See `skip_framework_members`.
from sphinx.ext.autodoc._sentinels import SLOTS_ATTR

# Autodoc imports the package, so src/ must be on the path. Resolved against
# this file rather than the working directory, so a build launched from the
# repository root finds the same tree as one launched from docs/.
sys.path.insert(0, str((Path(__file__).parents[2] / "src").resolve()))

project = "fastrag"
author = "AgrospAI"
copyright = "2026, AgrospAI"
# Not `from importlib.metadata import version`: every top-level name here is
# read as a Sphinx setting, and `version` is one.
release = importlib.metadata.version("fastrag-ai")

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.intersphinx",
    "sphinx.ext.viewcode",
    "myst_parser",
    "sphinx_copybutton",
    "sphinx_design",
    "sphinxcontrib.mermaid",
]

# -- Autodoc -----------------------------------------------------------
# "description" moves type hints out of the signature and into the
# parameter list, which keeps signatures readable once they carry
# generics and unions.
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    # Dataclass fields whose name and type say everything carry no docstring on
    # purpose, and they are still part of the public surface.
    "undoc-members": True,
    # A concrete step's own constructor args are only part of the story —
    # `run()`, `cache`, `policy` and friends live on its public base
    # (`PipelineStep`/`SourceStep`/`DocumentStep`), so without this a
    # concrete class like `WebCrawler` would document its own fields and
    # nothing it actually inherits.
    "inherited-members": True,
    "show-inheritance": True,
}
autosummary_generate = True

# ...which together pull in everything the third-party base classes contribute:
# pydantic's whole model API on Settings, SQLAlchemy's registry on Base,
# BaseException's on twelve exception types. Skipped by `skip_framework_members`
# below — by defining module where the member has one, and by name where it does
# not (a plain class attribute knows nothing about who declared it).
FRAMEWORK_MEMBERS = frozenset(
    {
        # pydantic.BaseModel
        "model_computed_fields",
        "model_config",
        "model_fields",
        # BaseException
        "add_note",
        "args",
        "with_traceback",
    }
)

#: Modules a member reports when it cannot say who declared it. See
#: `declaring_module`.
NON_ANSWERS = ("builtins", "sphinx.")

#: Names a wrapper package's ``__init__.py`` re-exports from a deeper module
#: that documents them as its own, mapped to the wrapper page(s) that must not
#: register them a second time. The canonical (deepest-defining) page wins;
#: see `skip_framework_members`. ``QueryOutcome`` is re-exported from two
#: wrappers — `fastrag.benchmark` and `fastrag.benchmark.metrics` — both
#: one hop closer to the surface than its home in
#: `fastrag.benchmark.metrics.base`.
REEXPORT_WRAPPER_DOCNAMES: dict[str, frozenset[str]] = {
    # ``fastrag`` (the top-level package, task 024) re-exports these three
    # from ``fastrag.ingestion``, which already re-exports them from their
    # own canonical modules — the same one-wrapper shape as the pre-queue-4
    # Index/SyncReport entries had.
    "Pipeline": frozenset({"reference/generated/fastrag"}),
    "Policy": frozenset({"reference/generated/fastrag"}),
    "Source": frozenset({"reference/generated/fastrag"}),
    "ContentFormat": frozenset({"reference/generated/fastrag"}),
    "QueryOutcome": frozenset(
        {
            "reference/generated/fastrag.benchmark",
            "reference/generated/fastrag.benchmark.metrics",
        }
    ),
    # ``fastrag.ingestion.steps.base`` is re-exported twice on the way to the
    # surface: once by ``fastrag.ingestion.steps`` (its own package), again by
    # ``fastrag.ingestion`` (the whole new package's top level).
    "PipelineStep": frozenset(
        {
            "reference/generated/fastrag.ingestion",
            "reference/generated/fastrag.ingestion.steps",
        }
    ),
    "ExecutionContext": frozenset(
        {
            "reference/generated/fastrag.ingestion",
            "reference/generated/fastrag.ingestion.steps",
        }
    ),
    "SourceStep": frozenset(
        {
            "reference/generated/fastrag.ingestion",
            "reference/generated/fastrag.ingestion.steps",
        }
    ),
    "DocumentStep": frozenset(
        {
            "reference/generated/fastrag.ingestion",
            "reference/generated/fastrag.ingestion.steps",
        }
    ),
    "RoutedStep": frozenset(
        {
            "reference/generated/fastrag.ingestion",
            "reference/generated/fastrag.ingestion.steps",
        }
    ),
    # ``fastrag.ingestion.cache`` re-exports from ``.base``/``.postgres``.
    "CacheStore": frozenset({"reference/generated/fastrag.ingestion.cache"}),
    "SourceFreshnessValidator": frozenset(
        {"reference/generated/fastrag.ingestion.cache"}
    ),
    # ``fastrag.retrieval.reranking`` is re-exported a second time by
    # ``fastrag.retrieval`` (task 025), the same double-hop shape as
    # PipelineStep/ExecutionContext/SourceStep/DocumentStep/RoutedStep above.
    "Reranker": frozenset(
        {
            "reference/generated/fastrag.retrieval",
            "reference/generated/fastrag.retrieval.reranking",
        }
    ),
    "CohereReranker": frozenset(
        {
            "reference/generated/fastrag.retrieval",
            "reference/generated/fastrag.retrieval.reranking",
        }
    ),
    # ``fastrag.query.prompts.templates.PromptTemplate`` is re-exported twice
    # on the way to the surface: once by ``fastrag.query.prompts`` (its own
    # package), again directly by ``fastrag.query`` (task 026) — the same
    # double-hop shape as Reranker/CohereReranker above.
    "RankFusion": frozenset(
        {
            "reference/generated/fastrag.retrieval",
            "reference/generated/fastrag.retrieval.fusion",
        }
    ),
    "ReciprocalRankFusion": frozenset(
        {
            "reference/generated/fastrag.retrieval",
            "reference/generated/fastrag.retrieval.fusion",
        }
    ),
    "PromptTemplate": frozenset(
        {
            "reference/generated/fastrag.query",
            "reference/generated/fastrag.query.prompts",
        }
    ),
}

# -- Napoleon (Google-style docstrings) --------------------------------
napoleon_google_docstring = True
napoleon_numpy_docstring = False
napoleon_use_param = True
napoleon_use_rtype = True

# -- Intersphinx -------------------------------------------------------
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "sqlalchemy": ("https://docs.sqlalchemy.org/en/20/", None),
    "alembic": ("https://alembic.sqlalchemy.org/en/latest/", None),
    # Publishes an objects.inv (unlike docling_core, its sibling package,
    # which does not) — resolves `InputFormat` in DoclingParser's signature.
    "docling": ("https://docling-project.github.io/docling/", None),
}

# -- MyST --------------------------------------------------------------
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "fieldlist",
    "attrs_inline",
    "substitution",
]
myst_heading_anchors = 3

source_suffix = {
    ".md": "markdown",
    ".rst": "restructuredtext",
}

templates_path = ["_templates"]

# -- Cross-references --------------------------------------------------
# Nitpicky, so a renamed symbol turns every stale :class: in a docstring into a
# build failure rather than into plain text nobody notices.
nitpicky = True
nitpick_ignore_regex = [
    # Private fallback implementation annotations have no public API target;
    # users supply the exported parser/renderer classes documented on this page.
    (r"py:class", r"^_AIParser$"),
    (r"py:class", r"^_AIRenderer$"),
    # Types belonging to dependencies that publish no object inventory. Not
    # fastrag's to link, and unresolvable however the docstring spells them.
    (r"py:.*", r"pydantic\..*"),
    # PEP 695 type parameters, which are scoped to one signature and so have no
    # target to point at. `autodoc_typehints = "description"` spells these out
    # both bare (same-module context) and fully qualified (cross-module, e.g.
    # an inherited-member page rendering `fastrag.ingestion`'s generic steps),
    # so both forms need covering.
    (r"py:.*", r"T"),
    (r"py:.*", r".*\.steps\.base\.T"),
    (r"py:.*", r".*\.steps\.base\.R"),
    (r"py:.*", r"T_contra"),
    (r"py:.*", r".*\.filters\.base\.T_contra"),
    # ``fastrag.query.chat.base.Page`` is generic over a plain
    # ``typing.TypeVar("T")``, not a PEP 695 parameter, but it has the same
    # "scoped to one signature, no target to point at" problem as the ones
    # above.
    (r"py:.*", r".*\.chat\.base\.T"),
    # FastAPI publishes no object inventory, so its types cannot resolve
    # however the docstring spells them.
    (r"py:.*", r"fastapi\..*"),
    # ``pathlib.Path`` in a public signature (the ask CLI's token-file
    # parameter). The intersphinx python inventory resolves it — but the two
    # warnings here are emitted while the module stub's own annotations are
    # being rendered under `ref.class`, where the cross-reference to the
    # `pathlib` domain object is not yet in scope. It links fine from the
    # CLI class row; pinning it as a nitpick ignore stops the stub build's
    # spurious failure.
    (r"py:class", r"^Path$"),
    # `docling.datamodel.base_models.InputFormat`, in `DoclingParser`'s
    # `allowed_formats`/`format_options` parameters. Same "not yet in scope
    # while the module stub's own annotations are rendered" shape as `Path`
    # above — the `docling` intersphinx mapping is still worth having for
    # every reference that resolves outside the stub build.
    (r"py:class", r"^InputFormat$"),
    # `fastrag.ingestion.steps.parsing.docling.OutputFormat` is a
    # `Literal["markdown", "docling"]` alias, not a class — there is no
    # `py:class` target for `autodoc_typehints = "description"` to link it
    # to however the docstring spells it, so this is fastrag's own name and
    # still unresolvable, the same shape as the PEP 695 type parameters above.
    (r"py:class", r"^OutputFormat$"),
    # `docling_core.transforms.chunker.BaseChunker`, the type of
    # `DoclingChunker`'s `chunker` parameter. `docling_core` is a separate
    # package from `docling` (see `intersphinx_mapping`) and publishes no
    # object inventory of its own.
    (r"py:class", r"^BaseChunker$"),
    # `httpx.AsyncBaseTransport`, the type of the ask CLI's `transport`
    # parameter. httpx's docs are built with mkdocs, not Sphinx, so there is
    # no object inventory to add to `intersphinx_mapping`.
    (r"py:class", r"^httpx\.AsyncBaseTransport$"),
    # `fastrag.ingestion.steps.chunking.docling.HierarchicalChunker` and
    # `fastrag.ingestion.steps.parsing.docling.DocumentStream` are the same
    # optional-import sentinel shape as `InputFormat`/`OutputFormat`/
    # `BaseChunker` above: a module-level name typed `Any` and re-assigned
    # from `docling_core` (no object inventory of its own) when the extra is
    # installed, `None` otherwise. Unlike the typehint-description warnings
    # above, this one comes from autosummary's own "Module Attributes" table,
    # which links the fully qualified local name rather than the bare one.
    (r"py:obj", r"^fastrag\.ingestion\.steps\.chunking\.docling\.HierarchicalChunker$"),
    (r"py:obj", r"^fastrag\.ingestion\.steps\.parsing\.docling\.DocumentStream$"),
]

# `Policy` names two genuinely different classes — `fastrag.policy.Policy`
# (old tree) and `fastrag.ingestion.policy.Policy` (new) — both re-exported
# one hop from their own module, per `fastrag.ingestion`'s own module
# docstring on why the split exists. Every signature naming either one, once
# `autodoc_typehints = "description"` spells it out, cross-references the
# type's bare name — never a fully qualified path a docstring could pin to
# one of the two — so this is an irreducible, permanent ambiguity, not a
# missing qualifier to add. `nitpick_ignore_regex` cannot silence it: Sphinx
# raises "more than one target found" (`ref.python`) through a different path
# than the "target not found" nitpicky checks. This suppresses that one
# message type globally, which is safe here because every other warning this
# build produces is still caught by the checks above.
suppress_warnings = ["ref.python"]

# -- Link checking -------------------------------------------------------
# `task docs-linkcheck` runs in CI with no network path to localhost and no
# control over a third party's uptime — both appear throughout the guide as
# the addresses examples/ actually talks to, not as documentation about them.
linkcheck_ignore = [
    r"^https?://localhost(:\d+)?/",
    r"^https?://127\.0\.0\.1(:\d+)?/",
    r"^https?://agrospai\.udl\.cat/",
]

# -- HTML --------------------------------------------------------------
html_theme = "pydata_sphinx_theme"
html_title = "fastrag"
html_static_path = ["_static"]
html_css_files = ["custom.css"]

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/AgrospAI/fastrag",
            "icon": "fa-brands fa-square-github",
            "type": "fontawesome",
        },
    ],
    # The reference pages are long and flat — one page per module, every member
    # a sibling — so the in-page table of contents is the only way to reach a
    # class without scrolling. Two levels shows those members.
    "show_toc_level": 2,
    "use_edit_page_button": True,
    "navigation_with_keys": False,
}

# Feeds the "Edit this page" button. `doc_path` is relative to the repository
# root, not to this file, because that is what the GitHub URL is built from.
html_context = {
    "github_user": "AgrospAI",
    "github_repo": "fastrag",
    "github_version": "main",
    "doc_path": "sdk/docs/source",
}


def declaring_module(obj):
    """Return the module that declared ``obj``, or ``None`` if it cannot tell."""
    # A mapped column reports sqlalchemy as its module, so ask it which class it
    # belongs to first — otherwise every CacheEntry column looks third-party.
    owner = getattr(obj, "class_", None)
    if owner is not None:
        return getattr(owner, "__module__", None)

    target = getattr(obj, "fget", obj)  # properties
    target = getattr(target, "__func__", target)  # bound and class methods
    if not (inspect.isroutine(target) or inspect.isclass(target)):
        # A plain value. `retries = 2` says nothing about who declared it, but
        # one a framework built — Base.metadata — gives itself away by type.
        target = type(target)

    module = getattr(target, "__module__", None)
    if module is None or module.startswith(NON_ANSWERS):
        # A default value reports builtins, and an annotation with no value at
        # all reports one of autodoc's sentinels. Neither is an answer, so
        # neither is a reason to skip.
        return None
    return module


def skip_framework_members(app, what, name, obj, skip, options):
    """Keep the base classes' own API out of fastrag's reference."""
    if name in FRAMEWORK_MEMBERS:
        return True

    # A frozen, slotted dataclass (``fastrag.ingestion``' models/events/policy/
    # step base) gives every field two documented identities: autodoc's own
    # `__slots__` handling (this `SLOTS_ATTR` sentinel) and the `py:attribute`
    # napoleon's "Attributes:" docstring section registers directly. A
    # non-slotted dataclass (the old tree's) only ever gets the second one —
    # `dir()` finds nothing at the class level for an unset field without
    # `__slots__`, so autodoc's own scan never fires. Skip the sentinel one and
    # keep napoleon's, which is the one carrying the written description.
    if obj is SLOTS_ATTR:
        return True

    # These normalization construction helpers are deliberately not re-exported.
    if (
        name in {"finish", "line_offsets"}
        and getattr(obj, "__module__", "") == "fastrag.normalization.models"
    ):
        return True

    module = declaring_module(obj)
    if module is not None and not module.startswith("fastrag"):
        return True

    # A wrapper __init__.py re-exports a name a deeper module already
    # documents as its own. Registering the same object from two pages makes
    # nitpicky builds fail; the canonical (deepest-defining) page wins.
    wrapper_docnames = REEXPORT_WRAPPER_DOCNAMES.get(name)
    if wrapper_docnames and app.env.current_document.docname in wrapper_docnames:
        return True

    return skip


def resolve_public_bases(app, name, obj, options, bases):
    """Rewrite the "Bases:" line to name only documented classes.

    No concrete class in the current tree inherits from a private base today
    (the fetchers that once did, e.g. ``WebCrawler``/``SitemapCrawler``, hold
    their shared ``_HttpTransport`` as an attribute instead), so this hook is
    a no-op in practice right now — kept because the next private mixin a
    step family grows would otherwise get advertised in its own "Bases:"
    line as a link to nowhere. If one class ever does inherit a private
    base again, this walks up to its nearest public ancestor and reports
    that instead.
    """
    resolved = []
    for base in bases:
        while getattr(base, "__name__", "").startswith("_"):
            base = base.__bases__[0]
        if base not in resolved:
            resolved.append(base)
    bases[:] = resolved


def setup(app):
    """Register the autodoc hooks."""
    app.connect("autodoc-skip-member", skip_framework_members)
    app.connect("autodoc-process-bases", resolve_public_bases)
