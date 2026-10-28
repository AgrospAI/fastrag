"""SDK-owned local service validation: start/probe/test/stop.

Stop preserves volumes. Postgres trust auth is confined to this test network
and loopback. Internal storage credentials are generated privately.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import secrets
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
ENV_FILE = ROOT / ".test-services.env"
COMPOSE = [
    "docker",
    "compose",
    "--project-name",
    "fastrag-sdk-tests",
    "--env-file",
    str(ENV_FILE),
    "--file",
    str(ROOT / "compose/docker-compose.test.yaml"),
]
PROBE_TESTS = [
    "tests/test_cache.py::test_bulk_reads_and_coalescing_preserve_cache_contract",
    "tests/test_retrieval.py::test_vector_store_save_upserts_every_document_and_caches_on_rerun",
]


def main() -> int:
    """Use isolated endpoints with bounded service/test subprocesses."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "probe", "test", "stop"))
    args = parser.parse_args()
    if args.action == "start" and not ENV_FILE.exists():
        descriptor = os.open(ENV_FILE, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(descriptor, "w") as stream:
            stream.write(f"SDK_TEST_STORAGE_USER={secrets.token_hex(8)}\n")
            stream.write(f"SDK_TEST_STORAGE_PASSWORD={secrets.token_hex(24)}\n")
    env = os.environ.copy()
    env.pop("SDK_TEST_STORAGE_USER", None)
    env.pop("SDK_TEST_STORAGE_PASSWORD", None)
    env["FASTRAG_DATABASE_URL"] = (
        "postgresql://postgres@127.0.0.1:55433/fastrag_sdk_test"
    )
    env["FASTRAG_MILVUS_URI"] = "http://127.0.0.1:19531"
    if args.action in ("start", "stop"):
        if not ENV_FILE.exists():
            parser.error("Run start first to create the SDK-local environment")
        command = COMPOSE + (
            ["up", "--detach", "--wait", "--wait-timeout", "150"]
            if args.action == "start"
            else ["down", "--timeout", "20"]
        )
        timeout = 240 if args.action == "start" else 60
    else:
        command = [sys.executable, "-m", "pytest", "-q"]
        if args.action == "probe":
            command += PROBE_TESTS
        timeout = 300
    try:
        return subprocess.run(command, cwd=ROOT, env=env, timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        print(f"{args.action} exceeded its {timeout}s ceiling", file=sys.stderr)
        return 124


if __name__ == "__main__":
    raise SystemExit(main())
