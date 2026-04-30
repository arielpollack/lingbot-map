from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def list_runs(self) -> list[dict[str, Any]]:
        data = self._read()
        return sorted(data.values(), key=lambda run: run["created_at"], reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any]:
        data = self._read()
        if run_id not in data:
            raise KeyError(run_id)
        return data[run_id]

    def create_run(self, filename: str, input_key: str) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            run_id = uuid4().hex
            run = {
                "id": run_id,
                "filename": filename,
                "input_key": input_key,
                "output_prefix": f"runs/{run_id}/",
                "status": "uploaded",
                "job_id": None,
                "result": None,
                "error": None,
                "created_at": _now_iso(),
                "updated_at": _now_iso(),
            }
            data[run_id] = run
            self._write(data)
            return run

    def update_run(self, run_id: str, **updates: Any) -> dict[str, Any]:
        with self._lock:
            data = self._read()
            if run_id not in data:
                raise KeyError(run_id)
            data[run_id].update(updates)
            data[run_id]["updated_at"] = _now_iso()
            self._write(data)
            return data[run_id]

    def _read(self) -> dict[str, dict[str, Any]]:
        if not self.path.exists():
            return {}
        with self.path.open("r", encoding="utf-8") as file:
            return json.load(file)

    def _write(self, data: dict[str, dict[str, Any]]) -> None:
        tmp_path = self.path.with_name(f".{self.path.name}.{uuid4().hex}.tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
        tmp_path.replace(self.path)


class RunStoreDict:
    """Same surface as RunStore but backed by a `modal.Dict` so the FastAPI
    asgi_app can persist run records across containers without a Volume.

    Each run is stored under its run_id as a single dict value. Iteration
    happens via `keys()` / `values()`. Modal Dict ops are network round-trips,
    so we cache the value on read inside a single request — but that's the
    caller's responsibility; for the simple POC handler each request just
    reads/writes the records it touches.
    """

    def __init__(self, modal_dict):
        self._d = modal_dict

    def list_runs(self) -> list[dict[str, Any]]:
        runs = list(self._d.values())
        return sorted(runs, key=lambda r: r.get("created_at", ""), reverse=True)

    def get_run(self, run_id: str) -> dict[str, Any]:
        if run_id not in self._d:
            raise KeyError(run_id)
        return self._d[run_id]

    def create_run(self, filename: str, input_key: str) -> dict[str, Any]:
        run_id = uuid4().hex
        run = {
            "id": run_id,
            "filename": filename,
            "input_key": input_key,
            "output_prefix": f"runs/{run_id}/",
            "status": "uploaded",
            "job_id": None,
            "result": None,
            "error": None,
            "created_at": _now_iso(),
            "updated_at": _now_iso(),
        }
        self._d[run_id] = run
        return run

    def update_run(self, run_id: str, **updates: Any) -> dict[str, Any]:
        if run_id not in self._d:
            raise KeyError(run_id)
        run = dict(self._d[run_id])  # copy to mutate then write back
        run.update(updates)
        run["updated_at"] = _now_iso()
        self._d[run_id] = run
        return run
