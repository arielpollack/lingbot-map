from poc.app.runpod_client import RunpodClient


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self.payload = payload
        self.status_code = status_code
        self.text = str(payload)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.text)

    def json(self):
        return self.payload


class FakeSession:
    def __init__(self):
        self.calls = []

    def post(self, url, headers=None, json=None, timeout=None):
        self.calls.append(("POST", url, headers, json, timeout))
        return FakeResponse({"id": "job-123", "status": "IN_QUEUE"})

    def get(self, url, headers=None, timeout=None):
        self.calls.append(("GET", url, headers, None, timeout))
        return FakeResponse({"id": "job-123", "status": "COMPLETED", "output": {"ok": True}})


def test_submit_posts_async_run_payload():
    session = FakeSession()
    client = RunpodClient(api_key="rp", endpoint_id="ep", session=session)

    response = client.submit_job({"run_id": "abc"}, execution_timeout_ms=900000)

    assert response["id"] == "job-123"
    method, url, headers, payload, timeout = session.calls[0]
    assert method == "POST"
    assert url == "https://api.runpod.ai/v2/ep/run"
    assert headers["authorization"] == "Bearer rp"
    assert headers["content-type"] == "application/json"
    assert payload["input"] == {"run_id": "abc"}
    assert payload["policy"]["executionTimeout"] == 900000
    assert payload["policy"]["ttl"] == 7200000
    assert timeout == 30


def test_status_gets_job_status():
    session = FakeSession()
    client = RunpodClient(api_key="rp", endpoint_id="ep", session=session)

    response = client.get_status("job-123")

    assert response["status"] == "COMPLETED"
    assert session.calls[0][1] == "https://api.runpod.ai/v2/ep/status/job-123"
