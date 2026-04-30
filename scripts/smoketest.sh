#!/usr/bin/env bash
# Run BEFORE a real GPU job to catch dumb failures (imports, env vars, paths,
# stale containers) on cheap CPU time only.
#
# Steps:
#   1. unit tests for the pure-Python pipeline / config
#   2. modal deploy (pushes any code changes; ~2 s if image cached)
#   3. modal run probe_worker — exercises imports + R2 + train.py path inside a
#      fresh CPU container on Modal
#
# Exits non-zero on any failure. Usage:
#   scripts/smoketest.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [ ! -f poc/.env ]; then
    echo "ERR: poc/.env missing — abort" >&2
    exit 1
fi

# Load env vars (cert bundle + R2 creds for local app code, modal CLI auth).
set -a
# shellcheck disable=SC1091
source poc/.env
set +a

PY=${PYTHON:-.venv/bin/python}
MODAL=${MODAL:-.venv/bin/modal}

echo "=== 1/3 unit tests ==="
"$PY" -m pytest tests/poc -q

echo
echo "=== 2/3 modal deploy ==="
"$MODAL" deploy poc/worker/modal_app.py

echo
echo "=== 3/3 modal run probe_worker (fresh CPU container) ==="
"$MODAL" run poc/worker/modal_app.py::probe_worker

echo
echo "smoketest passed — safe to submit a real GPU job"
