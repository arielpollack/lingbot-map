from __future__ import annotations

from pathlib import Path
import mimetypes

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

from poc.app.config import require_env
from poc.app.modal_client import ModalClient
from poc.app.r2 import R2Client
from poc.app.runs import RunStore


APP_DIR = Path(__file__).resolve().parent
STATIC_DIR = APP_DIR / "static"

config = require_env()
r2 = R2Client(config)
runner = ModalClient(config.modal_app_name, config.modal_function_name)
store = RunStore(Path(config.data_dir) / "runs.json")

app = FastAPI(title="LingBot-Map POC")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/viewer")
def viewer():
    return FileResponse(STATIC_DIR / "viewer.html")


@app.get("/splat")
def splat():
    return FileResponse(STATIC_DIR / "splat.html")


@app.get("/mesh")
def mesh():
    return FileResponse(STATIC_DIR / "mesh.html")


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
    if fps < 1 or fps > 60:
        raise HTTPException(status_code=400, detail="fps must be between 1 and 60")
    if first_k is not None and first_k < 1:
        raise HTTPException(status_code=400, detail="first_k must be greater than or equal to 1")
    if stride < 1:
        raise HTTPException(status_code=400, detail="stride must be greater than or equal to 1")

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
                "mesh_enabled": True,
                "conf_threshold": 1.5,
                "downsample_factor": 10,
                "point_size": 0.00001,
            },
        }
        response = runner.submit_job(payload)
        job_id = response.get("id")
        run = store.update_run(run_id, status="submitted", job_id=job_id, result=response)
        return run
    except Exception as exc:
        run = store.update_run(run_id, status="failed", error=str(exc))
        raise HTTPException(status_code=500, detail=run["error"]) from exc


@app.post("/api/runs/bundle")
def create_bundle_run(bundle: UploadFile = File(...)):
    """Phone capture bundle upload (Tier 1 LiDAR / Tier 2 VIO).

    Accepts a single zip blob shaped as ios_app produces:
        manifest.json, frames/, poses/, depth/

    Routes to the LiDAR-aware pipeline on the worker side. The web-side flow
    is identical to `POST /api/runs` (provisional run row → R2 upload →
    Modal job spawn → return run record).
    """
    if not bundle.filename:
        raise HTTPException(status_code=400, detail="Upload must include a filename")

    content_type = bundle.content_type or mimetypes.guess_type(bundle.filename)[0] or "application/zip"
    provisional_run = store.create_run(filename=bundle.filename, input_key="")
    run_id = provisional_run["id"]
    input_key = f"inputs/{run_id}/{bundle.filename}"
    store.update_run(run_id, input_key=input_key, source="bundle")

    try:
        r2.upload_fileobj(bundle.file, input_key, content_type=content_type)
        payload = {
            "run_id": run_id,
            "input_bundle_key": input_key,
            "output_prefix": f"runs/{run_id}/",
            "options": {},
        }
        response = runner.submit_job(payload)
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
        try:
            status = runner.get_status(job_id)
        except Exception as exc:
            run = store.update_run(run_id, status="failed", error=f"status check failed: {exc}")
            return run
        modal_status = status.get("status")
        if modal_status == "COMPLETED":
            output = status.get("output") or {}
            run = store.update_run(run_id, status="completed", result=status, output=output)
        elif modal_status in {"FAILED", "CANCELLED", "TIMED_OUT", "EXPIRED"}:
            run = store.update_run(run_id, status="failed", result=status, error=status.get("error") or modal_status)
        else:
            run = store.update_run(run_id, status=str(modal_status or "running").lower(), result=status)
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

    head = r2.head_object(key)
    if name.endswith(".glb"):
        media_type = "model/gltf-binary"
    elif name.endswith(".ply") or name.endswith(".splat"):
        # `.splat` is the antimatter15 binary format (no gzip — content is
        # already a packed binary float/int payload). PLY may carry
        # Content-Encoding: gzip from older runs; the head_object lookup
        # below picks that up.
        media_type = "application/octet-stream"
    else:
        media_type = "application/json"

    headers = {}
    encoding = head.get("ContentEncoding")
    if encoding:
        headers["Content-Encoding"] = encoding
    length = head.get("ContentLength")
    if length is not None:
        headers["Content-Length"] = str(length)

    body = r2.get_object_body(key)
    return StreamingResponse(body.iter_chunks(), media_type=media_type, headers=headers)
