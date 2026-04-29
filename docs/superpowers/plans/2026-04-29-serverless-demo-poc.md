# Serverless Demo POC Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a functional local upload page that submits LingBot-Map reconstruction jobs to Runpod Serverless, stores inputs/outputs in Cloudflare R2, and opens completed GLB artifacts in a simple FPS-style viewer.

**Architecture:** Add a self-contained `poc/` app for local control-plane concerns and a `poc/worker/` Runpod worker for GPU execution. The worker reuses the existing `demo.py` inference functions and a new package-level GLB export helper so it can run headlessly without Viser. The browser UI stays plain: upload form, run table, polling, and a Three.js free-fly viewer.

**Tech Stack:** Python 3.10+, FastAPI, uvicorn, boto3, requests, pytest, Runpod Serverless, Cloudflare R2 S3 API, Hugging Face Hub, Three.js.

---

## File Structure

- Create `poc/__init__.py`: marks the POC package.
- Create `poc/app/__init__.py`: marks local app package.
- Create `poc/app/config.py`: environment parsing and validation.
- Create `poc/app/r2.py`: small Cloudflare R2 client wrapper.
- Create `poc/app/runpod_client.py`: Runpod async submit/status client.
- Create `poc/app/runs.py`: JSON-file run state store.
- Create `poc/app/main.py`: FastAPI routes and static file serving.
- Create `poc/app/static/index.html`: plain upload/run list page.
- Create `poc/app/static/app.js`: upload, polling, and run table logic.
- Create `poc/app/static/viewer.html`: GLB viewer shell.
- Create `poc/app/static/viewer.js`: Three.js GLB loading and FPS controls.
- Create `poc/app/static/styles.css`: minimal functional styling.
- Create `poc/worker/requirements.txt`: worker dependencies.
- Create `poc/worker/Dockerfile`: Runpod worker image.
- Create `poc/worker/handler.py`: Runpod Serverless handler.
- Create `poc/worker/run_reconstruction.py`: headless reconstruction orchestration.
- Create `poc/README.md`: setup, local run, worker build, demo steps.
- Create `poc/.env.example`: all required environment variables.
- Create `tests/poc/test_config.py`: config validation tests.
- Create `tests/poc/test_runs.py`: JSON run store tests.
- Create `tests/poc/test_runpod_client.py`: Runpod request/status tests.
- Create `tests/poc/test_worker_handler.py`: handler success/failure unit tests with monkeypatched IO.
- Modify `lingbot_map/vis/glb_export.py`: add a focused `export_predictions_glb_file()` helper.

## Task 1: POC Configuration

**Files:**
- Create: `poc/__init__.py`
- Create: `poc/app/__init__.py`
- Create: `poc/app/config.py`
- Create: `tests/poc/test_config.py`

- [x] **Step 1: Write the failing config tests**

Create `tests/poc/test_config.py`:

```python
import pytest

from poc.app.config import AppConfig, require_env


def test_require_env_reads_required_values(monkeypatch):
    monkeypatch.setenv("RUNPOD_API_KEY", "rp-key")
    monkeypatch.setenv("RUNPOD_ENDPOINT_ID", "endpoint-id")
    monkeypatch.setenv("R2_ACCOUNT_ID", "account")
    monkeypatch.setenv("R2_ACCESS_KEY_ID", "access")
    monkeypatch.setenv("R2_SECRET_ACCESS_KEY", "secret")
    monkeypatch.setenv("R2_BUCKET", "bucket")
    monkeypatch.setenv("R2_PUBLIC_BASE_URL", "https://cdn.example.com/base/")

    config = require_env()

    assert config.runpod_api_key == "rp-key"
    assert config.runpod_endpoint_id == "endpoint-id"
    assert config.r2_endpoint_url == "https://account.r2.cloudflarestorage.com"
    assert config.r2_public_base_url == "https://cdn.example.com/base"


def test_require_env_reports_missing_values(monkeypatch):
    for name in AppConfig.required_env_names():
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError) as exc:
        require_env()

    message = str(exc.value)
    assert "Missing required environment variables" in message
    assert "RUNPOD_API_KEY" in message
    assert "R2_BUCKET" in message
```

- [x] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/poc/test_config.py -q`

Expected: FAIL with `ModuleNotFoundError: No module named 'poc'`.

- [x] **Step 3: Implement config parsing**

Create `poc/__init__.py` as an empty file.

Create `poc/app/__init__.py` as an empty file.

Create `poc/app/config.py`:

```python
from dataclasses import dataclass
import os


@dataclass(frozen=True)
class AppConfig:
    runpod_api_key: str
    runpod_endpoint_id: str
    r2_account_id: str
    r2_access_key_id: str
    r2_secret_access_key: str
    r2_bucket: str
    r2_public_base_url: str | None = None
    data_dir: str = "poc/.data"

    @classmethod
    def required_env_names(cls) -> tuple[str, ...]:
        return (
            "RUNPOD_API_KEY",
            "RUNPOD_ENDPOINT_ID",
            "R2_ACCOUNT_ID",
            "R2_ACCESS_KEY_ID",
            "R2_SECRET_ACCESS_KEY",
            "R2_BUCKET",
        )

    @property
    def r2_endpoint_url(self) -> str:
        return f"https://{self.r2_account_id}.r2.cloudflarestorage.com"


def require_env() -> AppConfig:
    missing = [name for name in AppConfig.required_env_names() if not os.getenv(name)]
    if missing:
        joined = ", ".join(missing)
        raise RuntimeError(f"Missing required environment variables: {joined}")

    public_base = os.getenv("R2_PUBLIC_BASE_URL")
    return AppConfig(
        runpod_api_key=os.environ["RUNPOD_API_KEY"],
        runpod_endpoint_id=os.environ["RUNPOD_ENDPOINT_ID"],
        r2_account_id=os.environ["R2_ACCOUNT_ID"],
        r2_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
        r2_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
        r2_bucket=os.environ["R2_BUCKET"],
        r2_public_base_url=public_base.rstrip("/") if public_base else None,
        data_dir=os.getenv("POC_DATA_DIR", "poc/.data"),
    )
```

- [x] **Step 4: Run config tests**

Run: `python -m pytest tests/poc/test_config.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

Run:

```bash
git add poc/__init__.py poc/app/__init__.py poc/app/config.py tests/poc/test_config.py
git commit -m "Add POC configuration"
```

## Task 2: JSON Run Store

**Files:**
- Create: `poc/app/runs.py`
- Create: `tests/poc/test_runs.py`

- [x] **Step 1: Write failing run store tests**

Create `tests/poc/test_runs.py`:

```python
from poc.app.runs import RunStore


def test_run_store_creates_updates_and_lists_runs(tmp_path):
    store = RunStore(tmp_path / "runs.json")

    run = store.create_run(filename="portrait.mp4", input_key="inputs/abc/portrait.mp4")
    assert run["id"]
    assert run["status"] == "uploaded"
    assert run["filename"] == "portrait.mp4"
    assert run["input_key"] == "inputs/abc/portrait.mp4"

    updated = store.update_run(run["id"], status="completed", job_id="job-1", output_key="runs/abc/scene.glb")
    assert updated["status"] == "completed"
    assert updated["job_id"] == "job-1"
    assert updated["output_key"] == "runs/abc/scene.glb"

    runs = store.list_runs()
    assert [item["id"] for item in runs] == [run["id"]]


def test_run_store_persists_between_instances(tmp_path):
    path = tmp_path / "runs.json"
    first = RunStore(path)
    run = first.create_run(filename="clip.mp4", input_key="inputs/run/clip.mp4")

    second = RunStore(path)
    assert second.get_run(run["id"])["filename"] == "clip.mp4"
```

- [x] **Step 2: Run the tests and verify they fail**

Run: `python -m pytest tests/poc/test_runs.py -q`

Expected: FAIL with `ModuleNotFoundError` or `ImportError` for `poc.app.runs`.

- [x] **Step 3: Implement the run store**

Create `poc/app/runs.py`:

```python
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
```

- [x] **Step 4: Run run store tests**

Run: `python -m pytest tests/poc/test_runs.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

Run:

```bash
git add poc/app/runs.py tests/poc/test_runs.py
git commit -m "Add POC run store"
```

## Task 3: R2 And Runpod Clients

**Files:**
- Create: `poc/app/r2.py`
- Create: `poc/app/runpod_client.py`
- Create: `tests/poc/test_runpod_client.py`

- [x] **Step 1: Write failing Runpod client tests**

Create `tests/poc/test_runpod_client.py`:

```python
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
    assert payload["input"] == {"run_id": "abc"}
    assert payload["policy"]["executionTimeout"] == 900000
    assert timeout == 30


def test_status_gets_job_status():
    session = FakeSession()
    client = RunpodClient(api_key="rp", endpoint_id="ep", session=session)

    response = client.get_status("job-123")

    assert response["status"] == "COMPLETED"
    assert session.calls[0][1] == "https://api.runpod.ai/v2/ep/status/job-123"
```

- [x] **Step 2: Run tests and verify failure**

Run: `python -m pytest tests/poc/test_runpod_client.py -q`

Expected: FAIL with `ModuleNotFoundError` or missing `RunpodClient`.

- [x] **Step 3: Implement Runpod client**

Create `poc/app/runpod_client.py`:

```python
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
```

- [x] **Step 4: Implement R2 client wrapper**

Create `poc/app/r2.py`:

```python
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
from urllib.parse import quote

import boto3

from poc.app.config import AppConfig


class R2Client:
    def __init__(self, config: AppConfig):
        self.bucket = config.r2_bucket
        self.public_base_url = config.r2_public_base_url
        self.client = boto3.client(
            "s3",
            endpoint_url=config.r2_endpoint_url,
            aws_access_key_id=config.r2_access_key_id,
            aws_secret_access_key=config.r2_secret_access_key,
            region_name="auto",
        )

    def upload_fileobj(self, fileobj: BinaryIO, key: str, content_type: str | None = None) -> None:
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            self.client.upload_fileobj(fileobj, self.bucket, key, ExtraArgs=extra_args)
        else:
            self.client.upload_fileobj(fileobj, self.bucket, key)

    def upload_file(self, path: str | Path, key: str, content_type: str | None = None) -> None:
        extra_args = {"ContentType": content_type} if content_type else None
        if extra_args:
            self.client.upload_file(str(path), self.bucket, key, ExtraArgs=extra_args)
        else:
            self.client.upload_file(str(path), self.bucket, key)

    def download_file(self, key: str, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.client.download_file(self.bucket, key, str(path))

    def get_object_body(self, key: str):
        return self.client.get_object(Bucket=self.bucket, Key=key)["Body"]

    def public_url(self, key: str) -> str | None:
        if not self.public_base_url:
            return None
        return f"{self.public_base_url}/{quote(key)}"
```

- [x] **Step 5: Run client tests**

Run: `python -m pytest tests/poc/test_runpod_client.py -q`

Expected: PASS.

- [x] **Step 6: Commit**

Run:

```bash
git add poc/app/r2.py poc/app/runpod_client.py tests/poc/test_runpod_client.py
git commit -m "Add POC storage and Runpod clients"
```

## Task 4: Local FastAPI App And Upload Flow

**Files:**
- Create: `poc/app/main.py`
- Create: `poc/app/static/index.html`
- Create: `poc/app/static/app.js`
- Create: `poc/app/static/styles.css`

- [x] **Step 1: Create FastAPI app**

Create `poc/app/main.py`:

```python
from __future__ import annotations

from pathlib import Path
import mimetypes

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from poc.app.config import require_env
from poc.app.r2 import R2Client
from poc.app.runpod_client import RunpodClient
from poc.app.runs import RunStore


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

config = require_env()
r2 = R2Client(config)
runpod = RunpodClient(config.runpod_api_key, config.runpod_endpoint_id)
store = RunStore(Path(config.data_dir) / "runs.json")

app = FastAPI(title="LingBot-Map POC")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/viewer")
def viewer():
    return FileResponse(STATIC_DIR / "viewer.html")


@app.get("/api/runs")
def list_runs():
    return {"runs": store.list_runs()}


@app.post("/api/runs")
def create_run(
    video: UploadFile = File(...),
    fps: int = Form(10),
    first_k: int | None = Form(None),
    stride: int = Form(1),
    mode: str = Form("streaming"),
    mask_sky: bool = Form(False),
):
    if not video.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename")
    if mode not in {"streaming", "windowed"}:
        raise HTTPException(status_code=400, detail="mode must be streaming or windowed")

    content_type = video.content_type or mimetypes.guess_type(video.filename)[0] or "application/octet-stream"
    provisional_run = store.create_run(filename=video.filename, input_key="")
    run_id = provisional_run["id"]
    input_key = f"inputs/{run_id}/{video.filename}"
    store.update_run(run_id, input_key=input_key)

    try:
        r2.upload_fileobj(video.file, input_key, content_type=content_type)
        payload = {
            "run_id": run_id,
            "input_video_key": input_key,
            "output_prefix": f"runs/{run_id}/",
            "options": {
                "fps": fps,
                "first_k": first_k,
                "stride": stride,
                "mode": mode,
                "mask_sky": mask_sky,
                "conf_threshold": 1.5,
                "downsample_factor": 10,
                "point_size": 0.00001,
            },
        }
        response = runpod.submit_job(payload)
        job_id = response.get("id")
        run = store.update_run(run_id, status="submitted", job_id=job_id, result=response)
        return run
    except Exception as exc:
        run = store.update_run(run_id, status="failed", error=str(exc))
        raise HTTPException(status_code=500, detail=run["error"]) from exc


@app.get("/api/runs/{run_id}")
def get_run(run_id: str):
    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc

    job_id = run.get("job_id")
    if job_id and run["status"] not in {"completed", "failed"}:
        status = runpod.get_status(job_id)
        runpod_status = status.get("status")
        if runpod_status == "COMPLETED":
            output = status.get("output") or {}
            run = store.update_run(run_id, status="completed", result=status, output=output)
        elif runpod_status in {"FAILED", "CANCELLED", "TIMED_OUT"}:
            run = store.update_run(run_id, status="failed", result=status, error=status.get("error") or runpod_status)
        else:
            run = store.update_run(run_id, status=str(runpod_status or "running").lower(), result=status)
    return run


@app.get("/api/runs/{run_id}/artifact/{name}")
def get_artifact(run_id: str, name: str):
    try:
        run = store.get_run(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Run not found") from exc

    key = f"runs/{run_id}/{name}"
    public_url = r2.public_url(key)
    if public_url:
        return {"url": public_url}

    body = r2.get_object_body(key)
    media_type = "model/gltf-binary" if name.endswith(".glb") else "application/json"
    return StreamingResponse(body.iter_chunks(), media_type=media_type)
```

- [x] **Step 2: Create the upload page**

Create `poc/app/static/index.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LingBot-Map POC</title>
    <link rel="stylesheet" href="/static/styles.css">
  </head>
  <body>
    <main>
      <h1>LingBot-Map POC</h1>
      <form id="upload-form">
        <label>Video <input id="video" name="video" type="file" accept="video/*" required></label>
        <label>FPS <input id="fps" name="fps" type="number" value="10" min="1" max="60"></label>
        <label>First K <input id="first_k" name="first_k" type="number" min="1"></label>
        <label>Stride <input id="stride" name="stride" type="number" value="1" min="1"></label>
        <label>Mode
          <select id="mode" name="mode">
            <option value="streaming">streaming</option>
            <option value="windowed">windowed</option>
          </select>
        </label>
        <label><input id="mask_sky" name="mask_sky" type="checkbox"> mask sky</label>
        <button type="submit">Upload and run</button>
      </form>
      <p id="status"></p>
      <table>
        <thead>
          <tr><th>Created</th><th>File</th><th>Status</th><th>Job</th><th>Action</th></tr>
        </thead>
        <tbody id="runs"></tbody>
      </table>
    </main>
    <script src="/static/app.js"></script>
  </body>
</html>
```

- [x] **Step 3: Add upload and polling JavaScript**

Create `poc/app/static/app.js`:

```javascript
const form = document.querySelector("#upload-form");
const statusNode = document.querySelector("#status");
const runsNode = document.querySelector("#runs");

async function apiJson(url, options = {}) {
  const response = await fetch(url, options);
  const text = await response.text();
  const data = text ? JSON.parse(text) : {};
  if (!response.ok) {
    throw new Error(data.detail || response.statusText);
  }
  return data;
}

function actionCell(run) {
  if (run.status === "completed") {
    return `<a href="/viewer?run=${encodeURIComponent(run.id)}" target="_blank">view</a>`;
  }
  return "";
}

function renderRuns(runs) {
  runsNode.innerHTML = runs.map((run) => `
    <tr>
      <td>${run.created_at || ""}</td>
      <td>${run.filename || ""}</td>
      <td>${run.status || ""}</td>
      <td>${run.job_id || ""}</td>
      <td>${actionCell(run)}</td>
    </tr>
  `).join("");
}

async function refreshRuns() {
  const data = await apiJson("/api/runs");
  const refreshed = [];
  for (const run of data.runs) {
    if (!["completed", "failed"].includes(run.status) && run.job_id) {
      refreshed.push(await apiJson(`/api/runs/${run.id}`));
    } else {
      refreshed.push(run);
    }
  }
  renderRuns(refreshed);
}

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  statusNode.textContent = "Uploading...";
  const formData = new FormData(form);
  if (!formData.get("first_k")) {
    formData.delete("first_k");
  }
  try {
    const run = await apiJson("/api/runs", { method: "POST", body: formData });
    statusNode.textContent = `Submitted ${run.id}`;
    form.reset();
    await refreshRuns();
  } catch (error) {
    statusNode.textContent = error.message;
  }
});

refreshRuns().catch((error) => {
  statusNode.textContent = error.message;
});
setInterval(() => refreshRuns().catch(() => {}), 5000);
```

- [x] **Step 4: Add plain styling**

Create `poc/app/static/styles.css`:

```css
body {
  font-family: system-ui, sans-serif;
  margin: 0;
  background: #f7f7f7;
  color: #111;
}

main {
  max-width: 980px;
  margin: 32px auto;
  padding: 0 16px;
}

form {
  display: grid;
  gap: 12px;
  padding: 16px;
  background: #fff;
  border: 1px solid #ddd;
}

label {
  display: grid;
  gap: 4px;
}

input,
select,
button {
  font: inherit;
  padding: 8px;
}

button {
  cursor: pointer;
}

table {
  width: 100%;
  margin-top: 24px;
  border-collapse: collapse;
  background: #fff;
}

th,
td {
  text-align: left;
  padding: 8px;
  border: 1px solid #ddd;
  vertical-align: top;
}
```

- [x] **Step 5: Install local app dependencies if needed**

Run: `python -m pip install fastapi uvicorn python-multipart boto3 requests pytest`

Expected: packages install successfully.

- [x] **Step 6: Start the local server**

Run: `uvicorn poc.app.main:app --reload --port 7860`

Expected: app starts at `http://127.0.0.1:7860`. If env vars are missing, it exits with the config error from Task 1.

- [x] **Step 7: Commit**

Run:

```bash
git add poc/app/main.py poc/app/static/index.html poc/app/static/app.js poc/app/static/styles.css
git commit -m "Add local POC upload app"
```

## Task 5: Headless GLB Export Helper

**Files:**
- Modify: `lingbot_map/vis/glb_export.py`

- [x] **Step 1: Add a package-level export helper**

Append this function to `lingbot_map/vis/glb_export.py`:

```python
def export_predictions_glb_file(
    predictions: dict,
    output_path: str,
    conf_threshold: float = 1.5,
    downsample_factor: int = 10,
    show_cam: bool = True,
) -> dict:
    """Export prepared demo predictions to a GLB file without starting Viser."""
    if trimesh is None:
        raise ImportError("trimesh is required for GLB export. Install with: pip install trimesh")

    world_points = predictions.get("world_points")
    conf = predictions.get("world_points_conf", predictions.get("depth_conf"))
    images = predictions["images"]
    extrinsics = predictions["extrinsic"]

    if world_points is None:
        from lingbot_map.utils.geometry import unproject_depth_map_to_point_map

        world_points = unproject_depth_map_to_point_map(
            predictions["depth"],
            predictions["extrinsic"],
            predictions["intrinsic"],
        )
        conf = predictions.get("depth_conf")

    if images.ndim == 4 and images.shape[1] == 3:
        colors = np.transpose(images, (0, 2, 3, 1))
    else:
        colors = images

    points = world_points[:, ::downsample_factor, ::downsample_factor, :].reshape(-1, 3)
    colors = colors[:, ::downsample_factor, ::downsample_factor, :].reshape(-1, 3)

    if conf is not None:
        conf_values = conf[:, ::downsample_factor, ::downsample_factor].reshape(-1)
        mask = conf_values >= conf_threshold
        points = points[mask]
        colors = colors[mask]

    colors_u8 = (np.clip(colors, 0.0, 1.0) * 255).astype(np.uint8)
    scene = trimesh.Scene()
    scene.add_geometry(trimesh.PointCloud(vertices=points, colors=colors_u8))

    if show_cam:
        camera_matrices = np.zeros((len(extrinsics), 4, 4))
        camera_matrices[:, :3, :4] = extrinsics
        camera_matrices[:, 3, 3] = 1.0
        if len(points) > 0:
            lo = np.percentile(points, 5, axis=0)
            hi = np.percentile(points, 95, axis=0)
            scene_scale = max(np.linalg.norm(hi - lo), 0.1)
        else:
            scene_scale = 1.0
        colormap = matplotlib.colormaps.get_cmap("gist_rainbow")
        for index, world_to_camera in enumerate(camera_matrices):
            camera_to_world = np.linalg.inv(world_to_camera)
            rgba = colormap(index / max(len(camera_matrices) - 1, 1))
            color = tuple(int(255 * value) for value in rgba[:3])
            integrate_camera_into_scene(scene, camera_to_world, color, scene_scale)

    scene.export(output_path)
    return {
        "output_path": output_path,
        "point_count": int(len(points)),
        "camera_count": int(len(extrinsics)),
    }
```

- [x] **Step 2: Smoke-check import**

Run: `python - <<'PY'\nfrom lingbot_map.vis.glb_export import export_predictions_glb_file\nprint(export_predictions_glb_file.__name__)\nPY`

Expected: prints `export_predictions_glb_file`.

- [x] **Step 3: Commit**

Run:

```bash
git add lingbot_map/vis/glb_export.py
git commit -m "Add headless GLB export helper"
```

## Task 6: Worker Reconstruction Wrapper

**Files:**
- Create: `poc/worker/run_reconstruction.py`

- [x] **Step 1: Create headless reconstruction wrapper**

Create `poc/worker/run_reconstruction.py`:

```python
from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json
import time

import cv2
from huggingface_hub import hf_hub_download
import torch

from demo import load_images, load_model, postprocess, prepare_for_visualization
from lingbot_map.vis.glb_export import export_predictions_glb_file


def video_metadata(video_path: str | Path) -> dict:
    cap = cv2.VideoCapture(str(video_path))
    try:
        return {
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0),
            "fps": float(cap.get(cv2.CAP_PROP_FPS) or 0),
            "frame_count": int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0),
        }
    finally:
        cap.release()


def resolve_model_path() -> str:
    return hf_hub_download(repo_id="robbyant/lingbot-map", filename="lingbot-map-long.pt")


def run_reconstruction(video_path: str | Path, output_dir: str | Path, options: dict) -> dict:
    start = time.time()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = Path(video_path)
    model_path = resolve_model_path()
    source = video_metadata(video_path)

    args = Namespace(
        image_folder=None,
        video_path=str(video_path),
        fps=int(options.get("fps", 10)),
        first_k=options.get("first_k"),
        stride=int(options.get("stride", 1)),
        model_path=model_path,
        image_size=518,
        patch_size=14,
        mode=options.get("mode", "streaming"),
        enable_3d_rope=True,
        max_frame_num=1024,
        num_scale_frames=8,
        keyframe_interval=None,
        kv_cache_sliding_window=64,
        camera_num_iterations=4,
        use_sdpa=bool(options.get("use_sdpa", False)),
        compile=False,
        offload_to_cpu=True,
        window_size=64,
        overlap_size=16,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    images, paths, _ = load_images(
        video_path=str(video_path),
        fps=args.fps,
        first_k=args.first_k,
        stride=args.stride,
        image_size=args.image_size,
        patch_size=args.patch_size,
    )
    model = load_model(args, device)
    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    if not torch.cuda.is_available():
        dtype = torch.float32

    images = images.to(device)
    if args.keyframe_interval is None:
        args.keyframe_interval = (images.shape[0] + 319) // 320 if args.mode == "streaming" and images.shape[0] > 320 else 1

    with torch.no_grad(), torch.amp.autocast("cuda", dtype=dtype, enabled=torch.cuda.is_available()):
        if args.mode == "streaming":
            predictions = model.inference_streaming(
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"),
            )
        else:
            predictions = model.inference_windowed(
                images,
                window_size=args.window_size,
                overlap_size=args.overlap_size,
                num_scale_frames=args.num_scale_frames,
                output_device=torch.device("cpu"),
            )

    predictions, images_cpu = postprocess(predictions, predictions["images"])
    prepared = prepare_for_visualization(predictions, images_cpu)
    glb_path = output_dir / "scene.glb"
    export = export_predictions_glb_file(
        prepared,
        str(glb_path),
        conf_threshold=float(options.get("conf_threshold", 1.5)),
        downsample_factor=int(options.get("downsample_factor", 10)),
    )

    metadata = {
        "source": source,
        "options": options,
        "frame_count": len(paths),
        "model_path": model_path,
        "duration_seconds": round(time.time() - start, 3),
        "export": export,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {"scene_path": str(glb_path), "metadata_path": str(metadata_path), "metadata": metadata}
```

- [x] **Step 2: Smoke-check import**

Run: `python - <<'PY'\nfrom poc.worker.run_reconstruction import video_metadata, run_reconstruction\nprint(video_metadata.__name__, run_reconstruction.__name__)\nPY`

Expected: imports without starting inference.

- [x] **Step 3: Commit**

Run:

```bash
git add poc/worker/run_reconstruction.py
git commit -m "Add headless reconstruction worker wrapper"
```

## Task 7: Runpod Worker Handler

**Files:**
- Create: `poc/worker/handler.py`
- Create: `tests/poc/test_worker_handler.py`

- [x] **Step 1: Write handler tests with monkeypatched dependencies**

Create `tests/poc/test_worker_handler.py`:

```python
from pathlib import Path

import poc.worker.handler as handler


class FakeR2:
    def __init__(self):
        self.downloads = []
        self.uploads = []

    def download_file(self, key, path):
        self.downloads.append((key, str(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"video")

    def upload_file(self, path, key, content_type=None):
        self.uploads.append((str(path), key, content_type))


def test_handler_downloads_runs_and_uploads(monkeypatch, tmp_path):
    fake_r2 = FakeR2()
    monkeypatch.setattr(handler, "build_r2_client", lambda: fake_r2)
    monkeypatch.setattr(handler.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "work"))

    def fake_run_reconstruction(video_path, output_dir, options):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = output_dir / "scene.glb"
        metadata = output_dir / "metadata.json"
        scene.write_bytes(b"glb")
        metadata.write_text("{}", encoding="utf-8")
        return {"scene_path": str(scene), "metadata_path": str(metadata), "metadata": {"ok": True}}

    monkeypatch.setattr(handler, "run_reconstruction", fake_run_reconstruction)

    result = handler.handle_job({
        "input": {
            "run_id": "abc",
            "input_video_key": "inputs/abc/video.mp4",
            "output_prefix": "runs/abc/",
            "options": {"fps": 10},
        }
    })

    assert result["run_id"] == "abc"
    assert result["scene_key"] == "runs/abc/scene.glb"
    assert fake_r2.downloads[0][0] == "inputs/abc/video.mp4"
    assert ("scene.glb", "metadata.json") == (
        Path(fake_r2.uploads[0][0]).name,
        Path(fake_r2.uploads[1][0]).name,
    )
```

- [x] **Step 2: Run test and verify failure**

Run: `python -m pytest tests/poc/test_worker_handler.py -q`

Expected: FAIL with missing `poc.worker.handler`.

- [x] **Step 3: Implement handler**

Create `poc/worker/handler.py`:

```python
from __future__ import annotations

from pathlib import Path
import os
import shutil
import tempfile

from poc.app.config import require_env
from poc.app.r2 import R2Client
from poc.worker.run_reconstruction import run_reconstruction


def build_r2_client() -> R2Client:
    return R2Client(require_env())


def handle_job(job: dict) -> dict:
    payload = job.get("input") or {}
    run_id = payload["run_id"]
    input_video_key = payload["input_video_key"]
    output_prefix = payload.get("output_prefix", f"runs/{run_id}/")
    options = payload.get("options") or {}

    work_dir = Path(tempfile.mkdtemp(prefix=f"lingbot-map-{run_id}-"))
    r2 = build_r2_client()
    try:
        input_path = work_dir / "input" / Path(input_video_key).name
        output_dir = work_dir / "output"
        r2.download_file(input_video_key, input_path)
        result = run_reconstruction(input_path, output_dir, options)

        scene_key = f"{output_prefix.rstrip('/')}/scene.glb"
        metadata_key = f"{output_prefix.rstrip('/')}/metadata.json"
        r2.upload_file(result["scene_path"], scene_key, content_type="model/gltf-binary")
        r2.upload_file(result["metadata_path"], metadata_key, content_type="application/json")
        return {
            "run_id": run_id,
            "scene_key": scene_key,
            "metadata_key": metadata_key,
            "metadata": result["metadata"],
        }
    finally:
        if os.getenv("KEEP_RUNPOD_WORKDIR") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handle_job})
```

- [x] **Step 4: Run handler test**

Run: `python -m pytest tests/poc/test_worker_handler.py -q`

Expected: PASS.

- [x] **Step 5: Commit**

Run:

```bash
git add poc/worker/handler.py tests/poc/test_worker_handler.py
git commit -m "Add Runpod worker handler"
```

## Task 8: Worker Packaging

**Files:**
- Create: `poc/worker/requirements.txt`
- Create: `poc/worker/Dockerfile`
- Create: `poc/.env.example`
- Create: `poc/README.md`

- [x] **Step 1: Create worker requirements**

Create `poc/worker/requirements.txt`:

```text
runpod~=1.7.6
boto3
requests
huggingface_hub
fastapi
python-multipart
trimesh
matplotlib
onnxruntime
viser>=0.2.23
flashinfer-python
```

- [x] **Step 2: Create Dockerfile**

Create `poc/worker/Dockerfile`:

```dockerfile
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace/lingbot-map

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE.txt demo.py ./
COPY lingbot_map ./lingbot_map
COPY poc ./poc

RUN pip install --no-cache-dir -e ".[vis]" \
    && pip install --no-cache-dir -r poc/worker/requirements.txt

CMD ["python", "-u", "poc/worker/handler.py"]
```

- [x] **Step 3: Create environment example**

Create `poc/.env.example`:

```dotenv
RUNPOD_API_KEY=
RUNPOD_ENDPOINT_ID=
R2_ACCOUNT_ID=
R2_ACCESS_KEY_ID=
R2_SECRET_ACCESS_KEY=
R2_BUCKET=
R2_PUBLIC_BASE_URL=
POC_DATA_DIR=poc/.data
```

- [x] **Step 4: Create POC README**

Create `poc/README.md`:

```markdown
# LingBot-Map Serverless POC

## Local app

```bash
python -m pip install fastapi uvicorn python-multipart boto3 requests pytest
cp poc/.env.example poc/.env
set -a
source poc/.env
set +a
uvicorn poc.app.main:app --reload --port 7860
```

Open `http://127.0.0.1:7860`.

## Worker image

```bash
docker build -f poc/worker/Dockerfile -t lingbot-map-runpod-poc .
docker tag lingbot-map-runpod-poc YOUR_DOCKERHUB_USER/lingbot-map-runpod-poc:latest
docker push YOUR_DOCKERHUB_USER/lingbot-map-runpod-poc:latest
```

Create a Runpod Serverless endpoint with that image. Configure the same R2 environment variables on the endpoint.

## Run flow

1. Upload a video in the local page.
2. The local app uploads it to R2 and submits a Runpod async job.
3. The worker downloads the video, downloads the LingBot-Map checkpoint from Hugging Face, exports `scene.glb`, and uploads results to R2.
4. The local page polls status. When complete, click `view`.
```

- [x] **Step 5: Commit**

Run:

```bash
git add poc/worker/requirements.txt poc/worker/Dockerfile poc/.env.example poc/README.md
git commit -m "Add Runpod worker packaging docs"
```

## Task 9: Static FPS Viewer

**Files:**
- Create: `poc/app/static/viewer.html`
- Create: `poc/app/static/viewer.js`

- [x] **Step 1: Create viewer HTML**

Create `poc/app/static/viewer.html`:

```html
<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>LingBot-Map Viewer</title>
    <link rel="stylesheet" href="/static/styles.css">
    <style>
      html, body, #viewer { width: 100%; height: 100%; margin: 0; overflow: hidden; }
      #hud { position: fixed; top: 12px; left: 12px; background: rgba(255,255,255,.9); padding: 8px; z-index: 2; }
      canvas { display: block; }
    </style>
  </head>
  <body>
    <div id="hud">Click to lock pointer. WASD move, mouse look, Space up, Shift down.</div>
    <div id="viewer"></div>
    <script type="module" src="/static/viewer.js"></script>
  </body>
</html>
```

- [x] **Step 2: Create viewer JavaScript**

Create `poc/app/static/viewer.js`:

```javascript
import * as THREE from "https://unpkg.com/three@0.164.1/build/three.module.js";
import { GLTFLoader } from "https://unpkg.com/three@0.164.1/examples/jsm/loaders/GLTFLoader.js";

const params = new URLSearchParams(location.search);
const runId = params.get("run");
const root = document.querySelector("#viewer");
const hud = document.querySelector("#hud");

const scene = new THREE.Scene();
scene.background = new THREE.Color(0xf0f0f0);

const camera = new THREE.PerspectiveCamera(70, innerWidth / innerHeight, 0.01, 10000);
camera.position.set(0, 0, 2);

const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(innerWidth, innerHeight);
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
root.appendChild(renderer.domElement);

scene.add(new THREE.AmbientLight(0xffffff, 1.0));
const light = new THREE.DirectionalLight(0xffffff, 1.0);
light.position.set(2, 5, 3);
scene.add(light);

const keys = new Set();
let locked = false;
let yaw = 0;
let pitch = 0;
let last = performance.now();
const speed = 1.5;

renderer.domElement.addEventListener("click", () => renderer.domElement.requestPointerLock());
document.addEventListener("pointerlockchange", () => {
  locked = document.pointerLockElement === renderer.domElement;
});
document.addEventListener("mousemove", (event) => {
  if (!locked) return;
  yaw -= event.movementX * 0.002;
  pitch -= event.movementY * 0.002;
  pitch = Math.max(-Math.PI / 2 + 0.01, Math.min(Math.PI / 2 - 0.01, pitch));
});
document.addEventListener("keydown", (event) => keys.add(event.code));
document.addEventListener("keyup", (event) => keys.delete(event.code));

function artifactUrl(name) {
  if (!runId) {
    throw new Error("Missing ?run=...");
  }
  return `/api/runs/${encodeURIComponent(runId)}/artifact/${encodeURIComponent(name)}`;
}

function setCameraFromObject(object) {
  const box = new THREE.Box3().setFromObject(object);
  const center = new THREE.Vector3();
  const size = new THREE.Vector3();
  box.getCenter(center);
  box.getSize(size);
  const radius = Math.max(size.x, size.y, size.z, 1);
  camera.position.copy(center).add(new THREE.Vector3(0, radius * 0.2, radius * 0.8));
  camera.lookAt(center);
}

async function loadScene() {
  hud.textContent = "Loading scene...";
  const loader = new GLTFLoader();
  loader.load(
    artifactUrl("scene.glb"),
    (gltf) => {
      scene.add(gltf.scene);
      setCameraFromObject(gltf.scene);
      hud.textContent = "Click to lock pointer. WASD move, mouse look, Space up, Shift down.";
    },
    undefined,
    (error) => {
      hud.textContent = `Failed to load GLB: ${error.message}`;
    }
  );
}

function updateCamera(delta) {
  camera.rotation.order = "YXZ";
  camera.rotation.y = yaw;
  camera.rotation.x = pitch;

  const forward = new THREE.Vector3();
  camera.getWorldDirection(forward);
  forward.y = 0;
  forward.normalize();
  const right = new THREE.Vector3().crossVectors(forward, camera.up).normalize();
  const movement = new THREE.Vector3();

  if (keys.has("KeyW")) movement.add(forward);
  if (keys.has("KeyS")) movement.sub(forward);
  if (keys.has("KeyD")) movement.sub(right);
  if (keys.has("KeyA")) movement.add(right);
  if (keys.has("Space")) movement.y += 1;
  if (keys.has("ShiftLeft") || keys.has("ShiftRight")) movement.y -= 1;
  if (movement.lengthSq() > 0) {
    movement.normalize().multiplyScalar(speed * delta);
    camera.position.add(movement);
  }
}

function animate(now) {
  const delta = Math.min((now - last) / 1000, 0.05);
  last = now;
  updateCamera(delta);
  renderer.render(scene, camera);
  requestAnimationFrame(animate);
}

addEventListener("resize", () => {
  camera.aspect = innerWidth / innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(innerWidth, innerHeight);
});

loadScene();
requestAnimationFrame(animate);
```

- [x] **Step 3: Smoke-check local server serves viewer**

Run: `uvicorn poc.app.main:app --port 7860`

Expected: `GET http://127.0.0.1:7860/viewer?run=test` returns the viewer HTML. It may show a GLB load error until a real run exists.

- [x] **Step 4: Commit**

Run:

```bash
git add poc/app/static/viewer.html poc/app/static/viewer.js
git commit -m "Add static FPS GLB viewer"
```

## Task 10: Final Verification

**Files:**
- Modify only if verification reveals a bug.

- [x] **Step 1: Run unit tests**

Run: `python -m pytest tests/poc -q`

Expected: all POC unit tests pass.

- [x] **Step 2: Run import checks**

Run:

```bash
python - <<'PY'
from poc.app.config import AppConfig
from poc.app.runs import RunStore
from poc.app.runpod_client import RunpodClient
from poc.worker.handler import handle_job
from lingbot_map.vis.glb_export import export_predictions_glb_file
print("imports ok")
PY
```

Expected: prints `imports ok`.

- [x] **Step 3: Run local app with dummy env**

Run:

```bash
RUNPOD_API_KEY=dummy \
RUNPOD_ENDPOINT_ID=dummy \
R2_ACCOUNT_ID=dummy \
R2_ACCESS_KEY_ID=dummy \
R2_SECRET_ACCESS_KEY=dummy \
R2_BUCKET=dummy \
uvicorn poc.app.main:app --port 7860
```

Expected: server starts. Uploading will fail against dummy R2, but `/` and `/viewer?run=test` render.

- [x] **Step 4: Build worker image metadata**

Run: `docker build -f poc/worker/Dockerfile -t lingbot-map-runpod-poc .`

Expected: Docker build completes. If network or Docker daemon is unavailable, record that in the final handoff.

- [x] **Step 5: Commit verification fixes**

If any fixes were needed:

```bash
git add <changed-files>
git commit -m "Fix POC verification issues"
```

## Self-Review

Spec coverage:

- Video upload including portrait: Task 4 accepts `video/*`; Task 6 records source dimensions and keeps OpenCV extraction.
- Runpod Serverless: Tasks 3, 7, and 8 cover async `/run`, worker handler, and Docker packaging.
- R2 storage: Tasks 3, 4, and 7 cover upload, download, proxy/public artifact access, and worker artifact uploads.
- Local upload/view runs page: Task 4 covers the app and polling table.
- FPS viewer: Task 9 covers static Three.js pointer-lock controls.
- Headless worker: Tasks 5 and 6 avoid Viser and export GLB directly.

Placeholder scan:

- No incomplete-marker terms or intentionally vague implementation steps remain.

Type consistency:

- `run_id`, `input_video_key`, `output_prefix`, `options`, `scene_key`, and `metadata_key` are consistent across local app, worker handler, and tests.
