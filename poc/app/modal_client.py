"""Submit jobs to the deployed Modal app and poll for status.

Returns dicts with the same shape as the old RunpodClient so main.py barely
changes:
    submit_job(payload) -> {"id": <job_id>, "status": "IN_QUEUE"}
    get_status(job_id)  -> {
        "status": "COMPLETED" | "IN_PROGRESS" | "FAILED" | "EXPIRED",
        "output"?: dict,
        "error"?: str,
    }
"""

from __future__ import annotations

from typing import Any

import modal


class ModalClient:
    def __init__(self, app_name: str, function_name: str):
        self.app_name = app_name
        self.function_name = function_name
        self._fn: modal.Function | None = None

    def _function(self) -> modal.Function:
        if self._fn is None:
            self._fn = modal.Function.from_name(self.app_name, self.function_name)
        return self._fn

    def submit_job(self, input_payload: dict[str, Any]) -> dict[str, Any]:
        call = self._function().spawn(input_payload)
        return {"id": call.object_id, "status": "IN_QUEUE"}

    def get_status(self, job_id: str) -> dict[str, Any]:
        try:
            call = modal.FunctionCall.from_id(job_id)
        except Exception as exc:
            # Modal returns a NotFoundError for reaped/unknown calls.
            return {"status": "EXPIRED", "error": f"function call not found: {exc}"}

        try:
            output = call.get(timeout=0)
        except TimeoutError:
            # Job is still running.
            return {"status": "IN_PROGRESS"}
        except Exception as exc:
            return {"status": "FAILED", "error": f"{type(exc).__name__}: {exc}"}

        return {"status": "COMPLETED", "output": output}
