"""Re-process a bundle that's already in R2 (skip the upload).

Use this for fast iteration on the worker (texture / 3DGS / pose-handling
changes) — no need to re-capture on the phone every time. The bundle stays
in R2 from the original upload; we just create a new run row + spawn a
fresh Modal job pointing at the same input_bundle_key.

Usage:
    python scripts/rerun_bundle.py --last-bundle
        # auto-pick the most recent bundle run

    python scripts/rerun_bundle.py --run <existing_run_id>
        # rerun a specific previous run by id

    python scripts/rerun_bundle.py --key inputs/<id>/file.zip
        # rerun against an explicit R2 key

    python scripts/rerun_bundle.py --last-bundle --options '{"splat_enabled": false}'
        # tweak server-side options without changing code

After spawning, prints the new run_id + the viewer URLs. Watch progress at:
    https://arielpollack--lingbot-map-web-fastapi-app.modal.run/api/runs/<id>
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from uuid import uuid4

import modal


WEB_DICT_NAME = "lingbot-map-runs"
APP_NAME = "lingbot-map-poc"
FN_NAME = "process_video"
WEB_BASE = "https://arielpollack--lingbot-map-web-fastapi-app.modal.run"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _resolve_input_key(args: argparse.Namespace, runs_dict: modal.Dict) -> str:
    if args.key:
        return args.key
    if args.run:
        if args.run not in runs_dict:
            sys.exit(f"run {args.run} not found in dict")
        run = runs_dict[args.run]
        key = run.get("input_key") or ""
        if not key:
            sys.exit(f"run {args.run} has no input_key")
        return key
    if args.last_bundle:
        candidates = []
        for rid in list(runs_dict.keys()):
            try:
                run = runs_dict[rid]
            except Exception:
                continue
            key = run.get("input_key") or ""
            if not key.endswith(".zip"):
                continue
            candidates.append((run.get("created_at", ""), rid, key))
        if not candidates:
            sys.exit("no bundle runs (.zip inputs) found in dict")
        candidates.sort(reverse=True)
        created_at, rid, key = candidates[0]
        print(f"using last bundle: run={rid} key={key} created={created_at}")
        return key
    sys.exit("specify --key, --run <id>, or --last-bundle")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument("--last-bundle", action="store_true",
                     help="rerun the most recent bundle upload")
    src.add_argument("--run", help="rerun a specific previous run by id")
    src.add_argument("--key", help="rerun against an explicit R2 input key")
    parser.add_argument("--options", default="{}",
                        help='JSON dict of options to forward to the worker '
                             '(e.g. \'{"splat_enabled": false}\')')
    args = parser.parse_args()

    options = json.loads(args.options)
    if not isinstance(options, dict):
        sys.exit("--options must be a JSON object")

    runs_dict = modal.Dict.from_name(WEB_DICT_NAME)
    fn = modal.Function.from_name(APP_NAME, FN_NAME)

    input_key = _resolve_input_key(args, runs_dict)
    new_run_id = uuid4().hex
    output_prefix = f"runs/{new_run_id}/"

    payload = {
        "run_id": new_run_id,
        "input_bundle_key": input_key,
        "output_prefix": output_prefix,
        "options": options,
    }

    # Register the run in the dict BEFORE spawning so the FastAPI poll
    # endpoint (/api/runs/<id>) can find it immediately.
    filename = input_key.split("/")[-1]
    runs_dict[new_run_id] = {
        "id": new_run_id,
        "filename": filename,
        "input_key": input_key,
        "output_prefix": output_prefix,
        "status": "submitted",
        "source": "rerun",
        "job_id": None,
        "result": None,
        "error": None,
        "created_at": _now_iso(),
        "updated_at": _now_iso(),
    }

    call = fn.spawn(payload)
    job_id = call.object_id

    # Patch the row with the modal job id so polling correlates.
    row = dict(runs_dict[new_run_id])
    row["job_id"] = job_id
    row["updated_at"] = _now_iso()
    runs_dict[new_run_id] = row

    print()
    print(f"  run_id    {new_run_id}")
    print(f"  job_id    {job_id}")
    print(f"  input     {input_key}")
    print(f"  options   {options}")
    print()
    print(f"  status    {WEB_BASE}/api/runs/{new_run_id}")
    print(f"  splat     {WEB_BASE}/splat?run={new_run_id}")
    print(f"  mesh      {WEB_BASE}/mesh?run={new_run_id}")
    print(f"  points    {WEB_BASE}/viewer?run={new_run_id}")
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
