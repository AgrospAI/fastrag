"""The deployable FastRAG backend: one pipeline, two entry points, one image.

This package is not part of the ``fastrag`` library (``src/fastrag/``) — it is
a *user* of it, the same way ``examples/`` is, except packaged as the image
task 006 ships. ``pipeline.py`` is the single file a user edits to change what
gets scraped, parsed, chunked, embedded, retrieved and answered; ``ingest.py``
and ``serve.py`` are thin entry points that both build from it. Nothing here
is imported by ``fastrag`` itself, and nothing under ``src/fastrag/`` ever
reads ``os.environ`` — that happens exactly once, in ``config.py``, read by
these entry points and nowhere else.
"""
