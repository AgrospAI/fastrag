"""Bump the fastrag-ai version and roll the changelog for a release.

``pyproject.toml`` is the one place the version is written. This script changes
it through ``uv version``, which re-locks ``uv.lock``, then updates the checked-in
files derived from it: the ``CHANGELOG.md`` headings and links, and
``info.version`` in ``openapi.json``. Run it through ``task bump``::

    task bump               # patch: 0.0.1 => 0.0.2 (``fix`` is an alias)
    task bump -- minor      # 0.0.1 => 0.1.0
    task bump -- major      # 0.0.1 => 1.0.0
    task bump -- 0.2.0rc1   # an explicit version

Merging the result into ``main`` releases it: the Release SDK workflow tags
``v<version>`` and publishes it. A version whose tag already exists is refused,
because that workflow would never release it; the retired CLI's ``v0.1.1`` to
``v0.1.8`` tags are taken.
"""

from __future__ import annotations

import datetime
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import NoReturn

SDK_DIR = Path(__file__).resolve().parent.parent
CHANGELOG_PATH = SDK_DIR / "CHANGELOG.md"
OPENAPI_PATH = SDK_DIR / "openapi.json"

#: The ``uv version --bump`` component for each accepted argument.
_PARTS = {"fix": "patch", "patch": "patch", "minor": "minor", "major": "major"}

_UNRELEASED_HEADING = "## [Unreleased]\n"
_UNRELEASED_LINK = re.compile(
    r"^\[Unreleased\]: (?P<repo>\S+)/compare/\S+\.\.\.HEAD$", re.MULTILINE
)


def _fail(message: str) -> NoReturn:
    sys.exit(f"bump failed: {message}")


def _uv_version(*args: str) -> str:
    """Run ``uv version`` in the SDK and return what it prints."""
    command = [os.environ.get("UV", "uv"), "version", *args]
    result = subprocess.run(command, cwd=SDK_DIR, stdout=subprocess.PIPE, text=True)
    if result.returncode != 0:
        _fail(f"`{' '.join(command)}` exited with status {result.returncode}")
    return result.stdout.strip()


def _tag_exists(tag: str) -> bool:
    result = subprocess.run(
        ["git", "rev-parse", "--quiet", "--verify", f"refs/tags/{tag}"],
        cwd=SDK_DIR,
        capture_output=True,
    )
    return result.returncode == 0


def _roll_changelog(text: str, previous: str, version: str, today: str) -> str:
    """Move the ``Unreleased`` entries under ``version`` and point the links at its tag."""
    link = _UNRELEASED_LINK.search(text)
    if _UNRELEASED_HEADING not in text or link is None:
        _fail(
            f"{CHANGELOG_PATH.name} has no `## [Unreleased]` heading and `[Unreleased]: ...` link"
        )

    entries = text.split(_UNRELEASED_HEADING, 1)[1].split("\n## [", 1)[0]
    if not entries.strip():
        print(
            f"warning: {CHANGELOG_PATH.name} has no Unreleased entries for {version}",
            file=sys.stderr,
        )

    repo = link["repo"]
    links = f"[Unreleased]: {repo}/compare/v{version}...HEAD\n[{version}]: {repo}/compare/v{previous}...v{version}"
    text = text[: link.start()] + links + text[link.end() :]
    return text.replace(
        _UNRELEASED_HEADING, f"{_UNRELEASED_HEADING}\n## [{version}] - {today}\n", 1
    )


def _set_openapi_version(version: str) -> None:
    # Written the way scripts/generate_openapi.py writes it, so regenerating
    # the document afterwards leaves this line alone.
    schema = json.loads(OPENAPI_PATH.read_text())
    schema["info"]["version"] = version
    OPENAPI_PATH.write_text(json.dumps(schema, indent=2, sort_keys=True) + "\n")


def main(args: list[str]) -> None:
    """Bump by ``args[0]`` (``patch`` by default, or an explicit version)."""
    if len(args) > 1:
        _fail("expected one argument: patch (default), minor, major, or a version")
    target = args[0] if args else "patch"
    bump = ["--bump", _PARTS[target]] if target in _PARTS else [target]

    previous = _uv_version("--short")
    version = _uv_version(*bump, "--dry-run", "--short")
    if version == previous:
        _fail(f"fastrag-ai is already at {version}")
    tag = f"v{version}"
    if _tag_exists(tag):
        _fail(
            f"tag {tag} already exists, so merging {version} would not release it. "
            "Bump past it, e.g. `task bump -- minor` or `task bump -- <version>`."
        )
    changelog = _roll_changelog(
        CHANGELOG_PATH.read_text(), previous, version, datetime.date.today().isoformat()
    )

    # Re-locks uv.lock but leaves the environment alone: a sync here would cut
    # it down to the default groups. The next `uv run` installs the new version.
    _uv_version(*bump, "--no-sync")
    CHANGELOG_PATH.write_text(changelog)
    _set_openapi_version(version)

    print(f"fastrag-ai {previous} => {version}")
    print(
        f"Review CHANGELOG.md, then merge into main to publish {version} and tag {tag}."
    )


if __name__ == "__main__":
    main(sys.argv[1:])
