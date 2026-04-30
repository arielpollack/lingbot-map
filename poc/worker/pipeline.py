"""Pure reconstruction pipeline: takes a payload dict, downloads input from R2,
runs the reconstruction + 3DGS phases, uploads artifacts to R2, returns
output keys. No serverless-platform dependencies — wrapped by `modal_app.py`."""

from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any

from poc.app.config import require_env
from poc.app.r2 import R2Client


def process_video(payload: dict[str, Any]) -> dict[str, Any]:
    """Run the full reconstruction + 3DGS pipeline for a single capture.

    Two payload shapes are supported:

    Raw video (Tier 3, web upload + Flutter app fallback):
        {
            "run_id": str,
            "input_video_key": str,        # R2 key for the .mov / .mp4
            "output_prefix": str | None,
            "options": dict | None,
        }

    Phone capture bundle (Tier 1 LiDAR via ios_app):
        {
            "run_id": str,
            "input_bundle_key": str,       # R2 key for the .zip
            "output_prefix": str | None,
            "options": dict | None,
        }

    Bundle inputs are routed to `lidar_pipeline.process_bundle`, which skips
    lingbot-map and uses ARKit poses + LiDAR depth directly.
    """
    if payload.get("input_bundle_key"):
        from poc.worker.lidar_pipeline import process_bundle
        return process_bundle(payload)

    run_id = payload["run_id"]
    input_video_key = payload["input_video_key"]
    output_prefix = (payload.get("output_prefix") or f"runs/{run_id}/").rstrip("/")
    options = payload.get("options") or {}

    work_dir = Path(tempfile.mkdtemp(prefix=f"lingbot-map-{run_id}-"))
    r2 = R2Client(require_env())

    try:
        input_path = work_dir / "input" / Path(input_video_key).name
        output_dir = work_dir / "output"
        r2.download_file(input_video_key, input_path)

        from poc.worker.run_reconstruction import run_reconstruction as reconstruct

        result = reconstruct(input_path, output_dir, options)

        scene_key = f"{output_prefix}/scene.glb"
        metadata_key = f"{output_prefix}/metadata.json"
        r2.upload_file(result["scene_path"], scene_key, content_type="model/gltf-binary")
        r2.upload_file(result["metadata_path"], metadata_key, content_type="application/json")

        splat_key = None
        mesh_key = None
        mesh_path = result.get("mesh_path")
        if mesh_path and Path(mesh_path).exists():
            mesh_key = f"{output_prefix}/mesh.glb"
            r2.upload_file(mesh_path, mesh_key, content_type="model/gltf-binary")

        splat_gz_path = result.get("splat_path")
        if splat_gz_path and Path(splat_gz_path).exists():
            splat_key = f"{output_prefix}/splat.ply"
            r2.upload_file(
                splat_gz_path,
                splat_key,
                content_type="application/octet-stream",
                content_encoding="gzip",
            )

        diagnostic_keys = []
        for artifact in result.get("diagnostic_artifacts", []):
            artifact_path = artifact.get("path")
            artifact_name = Path(artifact.get("name") or Path(artifact_path).name).name
            if not artifact_path or not Path(artifact_path).exists():
                continue
            artifact_key = f"{output_prefix}/diagnostics/{artifact_name}"
            r2.upload_file(
                artifact_path,
                artifact_key,
                content_type=artifact.get("content_type") or "application/octet-stream",
                content_encoding=artifact.get("content_encoding"),
            )
            diagnostic_keys.append(artifact_key)

        return {
            "run_id": run_id,
            "scene_key": scene_key,
            "mesh_key": mesh_key,
            "splat_key": splat_key,
            "diagnostic_keys": diagnostic_keys,
            "metadata_key": metadata_key,
            "metadata": result["metadata"],
        }
    finally:
        if os.getenv("KEEP_WORKDIR") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)
