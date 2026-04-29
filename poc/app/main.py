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
