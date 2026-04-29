from __future__ import annotations

from pathlib import Path
from typing import Any
import os
import shutil
import tempfile

from poc.app.config import require_env
from poc.app.r2 import R2Client


def build_r2_client() -> R2Client:
    return R2Client(require_env())


def run_reconstruction(video_path: str | Path, output_dir: str | Path, options: dict) -> dict:
    from poc.worker.run_reconstruction import run_reconstruction as reconstruct

    return reconstruct(video_path, output_dir, options)


def handle_job(job: dict[str, Any]) -> dict[str, Any]:
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
