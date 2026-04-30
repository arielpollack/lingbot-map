"""Modal Serverless deployment for the lingbot-map worker.

Replaces the Runpod handler. Reuses the existing Docker image (which already
has lingbot-map + 3DGS CUDA wheels + train.py baked in) via from_registry.

Deploy:
    modal token new           # one-time, opens browser auth
    modal deploy poc/worker/modal_app.py

Submit a job from Python (handled by poc/app/modal_client.py):
    f = modal.Function.from_name("lingbot-map-poc", "process_video")
    call = f.spawn(payload)
    # later
    result = modal.FunctionCall.from_id(call.object_id).get(timeout=0)
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "lingbot-map-poc"
FUNCTION_NAME = "process_video"

# Reuse the worker image we already build for Runpod (Docker Hub, public).
# Bumping this tag is the way to ship a new worker version on Modal — `modal
# deploy` pins to whatever digest this string resolves to at deploy time, and
# every running invocation of the deployed app uses that pinned image.
WORKER_IMAGE = os.getenv(
    "LINGBOT_WORKER_IMAGE",
    "arielpollack/lingbot-map-runpod-poc:gsplat-app-20260430-2240",
)

# Pre-existing Modal secret in this workspace. Holds CLOUDFLARE_ACCOUNT_ID,
# HF_TOKEN, R2_ACCESS_KEY_ID, R2_BUCKET_NAME, R2_ENDPOINT, R2_SECRET_ACCESS_KEY.
# Modal injects these as env vars at runtime; we re-map them to the names our
# config.py expects in the entrypoint below.
worker_secret = modal.Secret.from_name("huggingface-secret")

# Base image. Mounting the live `poc/` tree on top is only relevant at deploy
# time (the local CLI computes paths relative to this file). Inside the
# container at runtime, `__file__` points at `/root/modal_app.py` which has no
# `parents[2]`, so we guard with `modal.is_local()`.
image = (
    modal.Image.from_registry(WORKER_IMAGE, add_python="3.11")
    .pip_install("modal", "open3d==0.19.0", "xatlas==0.0.11")
    .env({"PYTHONPATH": "/workspace/lingbot-map"})
    .workdir("/workspace/lingbot-map")
)
if modal.is_local():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    image = image.add_local_dir(
        str(_REPO_ROOT / "poc"),
        remote_path="/workspace/lingbot-map/poc",
        ignore=["__pycache__", "*.pyc", ".data", ".env"],
    )

app = modal.App(APP_NAME, image=image)


def _apply_secret_aliases() -> None:
    """Map the huggingface-secret env var names to the names our config expects."""
    aliases = {
        "R2_BUCKET": "R2_BUCKET_NAME",
        "R2_ACCOUNT_ID": "CLOUDFLARE_ACCOUNT_ID",
    }
    for ours, theirs in aliases.items():
        if not os.getenv(ours) and os.getenv(theirs):
            os.environ[ours] = os.environ[theirs]


@app.function(
    gpu="A10G",          # arch 8.6 — matches our 3DGS wheels (TORCH_CUDA_ARCH_LIST=8.6;8.9)
    timeout=1800,        # 30 min — covers cold start + reconstruction + 7k splat iters
    secrets=[worker_secret],
    name=FUNCTION_NAME,
)
def process_video(payload: dict) -> dict:
    """Modal entry point. Delegates to the platform-agnostic pipeline."""
    _apply_secret_aliases()
    from poc.worker.pipeline import process_video as run_pipeline
    return run_pipeline(payload)


@app.function(
    timeout=120,
    secrets=[worker_secret],
)
def probe_worker() -> dict:
    """Cheap CPU-only sanity check. Confirms the deployed container can:
      - import the pipeline + downstream modules
      - read R2 with the credentials we passed in
      - see the bundled 3DGS train.py
    Run before paying for a GPU job:
        modal run poc/worker/modal_app.py::probe_worker
    """
    _apply_secret_aliases()
    checks: dict = {}

    try:
        from poc.worker import pipeline, gsplat, mesh
        from poc.worker.pipeline import process_video  # noqa
        checks["import_pipeline"] = True
        checks["pipeline_path"] = pipeline.__file__
        checks["gsplat_path"] = gsplat.__file__
        checks["mesh_path"] = mesh.__file__
    except Exception as exc:
        checks["import_pipeline"] = f"FAIL: {type(exc).__name__}: {exc}"
        return checks

    try:
        import open3d as o3d
        checks["open3d"] = o3d.__version__
    except Exception as exc:
        checks["open3d"] = f"FAIL: {type(exc).__name__}: {exc}"

    try:
        # Verify the gsplat-library trainer (new for the lidar_mesh tier)
        # can resolve its CUDA kernels at import time. The runtime call
        # path needs `rasterization` + `DefaultStrategy`.
        import gsplat as _gsplat_lib
        from gsplat.rendering import rasterization  # noqa
        from gsplat.strategy import DefaultStrategy  # noqa
        checks["gsplat_lib"] = _gsplat_lib.__version__
    except Exception as exc:
        checks["gsplat_lib"] = f"FAIL: {type(exc).__name__}: {exc}"

    try:
        from poc.app.config import require_env
        from poc.app.r2 import R2Client
        cfg = require_env()
        r2 = R2Client(cfg)
        # Just touch the bucket to ensure creds + endpoint URL work.
        r2.client.list_objects_v2(Bucket=cfg.r2_bucket, MaxKeys=1)
        checks["r2_connect"] = True
        checks["r2_bucket"] = cfg.r2_bucket
    except Exception as exc:
        checks["r2_connect"] = f"FAIL: {type(exc).__name__}: {exc}"

    import os.path
    train_script = "/opt/gaussian-splatting/train.py"
    checks["gaussian_splatting_train_py"] = (
        train_script if os.path.exists(train_script) else f"MISSING: {train_script}"
    )

    return checks
