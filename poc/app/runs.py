from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any
from uuid import uuid4


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class RunStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
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
        tmp_path = self.path.with_suffix(".tmp")
        with tmp_path.open("w", encoding="utf-8") as file:
            json.dump(data, file, indent=2, sort_keys=True)
        tmp_path.replace(self.path)
