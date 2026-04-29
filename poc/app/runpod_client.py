from __future__ import annotations

from typing import Any

import requests


class RunpodClient:
    def __init__(self, api_key: str, endpoint_id: str, session: Any | None = None):
        self.api_key = api_key
        self.endpoint_id = endpoint_id
        self.session = session or requests.Session()

    @property
    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self.api_key}",
            "content-type": "application/json",
        }

    def submit_job(self, input_payload: dict[str, Any], execution_timeout_ms: int = 3600000) -> dict[str, Any]:
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/run"
        payload = {
            "input": input_payload,
            "policy": {
                "executionTimeout": execution_timeout_ms,
                "ttl": 7200000,
            },
        }
        response = self.session.post(url, headers=self._headers, json=payload, timeout=30)
        response.raise_for_status()
        return response.json()

    def get_status(self, job_id: str) -> dict[str, Any]:
        url = f"https://api.runpod.ai/v2/{self.endpoint_id}/status/{job_id}"
        response = self.session.get(url, headers=self._headers, timeout=30)
        response.raise_for_status()
        return response.json()
