from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json
import os
import tarfile
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import cv2
from huggingface_hub import hf_hub_download
import torch

from demo import load_images, load_model, postprocess, prepare_for_visualization
from lingbot_map.vis.glb_export import export_predictions_glb_file
from lingbot_map.utils.geometry import (
    closed_form_inverse_se3_general,
    unproject_depth_map_to_point_map,
)

from poc.worker.gsplat import (
    compute_render_metrics,
    export_colmap_workspace,
    gzip_file,
    render_gaussian_splatting_train_views,
    reprojection_sanity,
    train_gaussian_splatting,
    viewer_camera_from_colmap,
)
from poc.worker.mesh import create_clay_mesh_from_prepared


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


def run_reconstruction(
    video_path: str | Path,
    output_dir: str | Path,
    options: dict,
) -> dict:
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
        # Default to PyTorch SDPA so we don't depend on flashinfer's runtime JIT
        # (which would force the worker base image onto the much larger CUDA
        # devel variant just to provide nvcc).
        use_sdpa=bool(options.get("use_sdpa", True)),
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
    dtype = (
        torch.bfloat16
        if torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        else torch.float16
    )
    if not torch.cuda.is_available():
        dtype = torch.float32

    if dtype != torch.float32 and getattr(model, "aggregator", None) is not None:
        print(f"Casting aggregator to {dtype} (heads kept in fp32)")
        model.aggregator = model.aggregator.to(dtype=dtype)

    images = images.to(device)
    if args.keyframe_interval is None:
        args.keyframe_interval = (
            (images.shape[0] + 319) // 320
            if args.mode == "streaming" and images.shape[0] > 320
            else 1
        )

    with torch.no_grad(), torch.amp.autocast(
        "cuda",
        dtype=dtype,
        enabled=torch.cuda.is_available(),
    ):
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

    # Match the canonical viser viewer (use_point_map=False): the model's
    # world_points head is unstable on some inputs and collapses to a tiny
    # cluster. Unproject from depth + camera intrinsics/extrinsics instead.
    prepared.pop("world_points", None)
    prepared.pop("world_points_conf", None)

    # The viser viewer filters by conf percentile (default top 50%), not an
    # absolute threshold. Compute the absolute value from the percentile so
    # downstream filtering matches.
    conf_percentile = float(options.get("conf_percentile", 50.0))
    depth_conf = prepared.get("depth_conf")
    if depth_conf is not None:
        import numpy as _np
        conf_flat = _np.asarray(depth_conf).reshape(-1)
        if conf_flat.size:
            conf_threshold = float(_np.percentile(conf_flat, conf_percentile))
        else:
            conf_threshold = 0.1
    else:
        conf_threshold = 0.1

    glb_path = output_dir / "scene.glb"
    export = export_predictions_glb_file(
        prepared,
        str(glb_path),
        conf_threshold=conf_threshold,
        downsample_factor=int(options.get("downsample_factor", 4)),
    )
    export["conf_threshold_used"] = conf_threshold
    export["conf_percentile_used"] = conf_percentile

    mesh_info = create_clay_mesh_from_prepared(
        prepared, output_dir, options, scene_glb_path=glb_path
    )
    splat_info = _run_splat_phase(prepared, output_dir, options)

    metadata = {
        "source": source,
        "options": dict(options),
        "frame_count": len(paths),
        "model_path": model_path,
        "duration_seconds": round(time.time() - start, 3),
        "export": export,
        "mesh": mesh_info,
        "splat": splat_info,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "scene_path": str(glb_path),
        "mesh_path": mesh_info.get("path"),
        "splat_path": splat_info.get("gz_path"),
        "metadata_path": str(metadata_path),
        "diagnostic_artifacts": splat_info.get("diagnostic_artifacts", []),
        "metadata": metadata,
    }


def _run_splat_phase(prepared: dict, output_dir: Path, options: dict) -> dict:
    """Train a 3D Gaussian Splat from the reconstruction. Returns metadata dict
    with `gz_path` set if successful, `error` otherwise. Failures are logged
    but never raised — the job still succeeds with point-cloud-only."""
    import numpy as np
    import tempfile

    info: dict = {"enabled": True}
    splat_start = time.time()
    try:
        diagnostics_enabled = bool(options.get("splat_diagnostics"))
        diagnostic_artifacts: list[dict] = []
        diagnostics_dir = output_dir / "diagnostics"
        if diagnostics_enabled:
            diagnostics_dir.mkdir(parents=True, exist_ok=True)

        depth = prepared.get("depth")
        extrinsic_c2w = prepared.get("extrinsic")     # (S, 3, 4) c2w from postprocess
        intrinsic = prepared.get("intrinsic")         # (S, 3, 3)
        depth_conf = prepared.get("depth_conf")       # (S, H, W)
        images_chw = prepared.get("images")           # (S, 3, H, W) float [0, 1]

        if depth is None or extrinsic_c2w is None or intrinsic is None or images_chw is None:
            info["error"] = "missing prediction tensors required for COLMAP export"
            print(f"3DGS skipped: {info['error']}", flush=True)
            return info

        # COLMAP wants w2c. Re-invert the c2w extrinsics (postprocess inverted
        # the original w2c output of the model).
        S = extrinsic_c2w.shape[0]
        ext_4x4_c2w = np.zeros((S, 4, 4), dtype=np.float32)
        ext_4x4_c2w[:, :3, :4] = np.asarray(extrinsic_c2w)
        ext_4x4_c2w[:, 3, 3] = 1.0
        ext_4x4_w2c = np.linalg.inv(ext_4x4_c2w)
        ext_3x4_w2c = ext_4x4_w2c[:, :3, :4]

        # Unproject from depth to TRUE world points. The unproject function
        # docstring says it expects w2c (it inverts internally to get c2w).
        # `glb_export.py` happens to pass c2w which yields a self-consistent
        # but absolute-frame-wrong scene that still LOOKS right; for COLMAP we
        # need true world coords matched with true w2c, so feed w2c here.
        world_points = unproject_depth_map_to_point_map(
            depth, ext_3x4_w2c, intrinsic
        )  # (S, H, W, 3)
        info["viewer_camera"] = viewer_camera_from_colmap(
            ext_4x4_w2c,
            np.asarray(intrinsic),
            image_size=world_points.shape[1:3],
            scene_points=world_points,
        )

        # (S, 3, H, W) -> (S, H, W, 3) for COLMAP image export.
        gt_images = np.transpose(np.asarray(images_chw), (0, 2, 3, 1))

        # Sanity check: do world_points reproject onto their source pixels via
        # the colmap_extrinsics + intrinsics? Median error >> 1 px means our
        # convention is wrong before we even spend GPU time on training.
        sanity = reprojection_sanity(
            world_points,
            np.asarray(intrinsic),
            ext_4x4_w2c,
            confidence=np.asarray(depth_conf) if depth_conf is not None else None,
        )
        info["reprojection"] = sanity
        print(f"3DGS reprojection sanity: {sanity}", flush=True)

        with tempfile.TemporaryDirectory(prefix="colmap-") as colmap_dir:
            colmap_stats = export_colmap_workspace(
                np.asarray(intrinsic),
                ext_4x4_w2c,
                np.asarray(world_points),
                gt_images,
                np.asarray(depth_conf) if depth_conf is not None
                    else np.ones(world_points.shape[:3], dtype=np.float32),
                colmap_dir,
                conf_threshold=float(options.get("splat_init_conf_threshold", 1.5)),
                target_points=int(options.get("splat_init_points", 50_000)),
                random_seed=options.get("splat_init_seed"),
            )
            info["colmap_export"] = colmap_stats

            if diagnostics_enabled:
                colmap_archive = diagnostics_dir / "colmap_workspace.tar.gz"
                _archive_directory(Path(colmap_dir), colmap_archive)
                diagnostic_artifacts.append(
                    {
                        "path": str(colmap_archive),
                        "name": "colmap_workspace.tar.gz",
                        "content_type": "application/gzip",
                    }
                )

            if options.get("splat_skip_training"):
                info["skipped_training"] = True
                info["duration_seconds"] = round(time.time() - splat_start, 3)
                info["diagnostic_artifacts"] = diagnostic_artifacts
                return info

            iterations = int(options.get("splat_iterations", 7000))
            gs_output_dir = output_dir / "gs_output"
            gs_output_dir.mkdir(parents=True, exist_ok=True)
            train_log_path = diagnostics_dir / "3dgs_train.log" if diagnostics_enabled else None

            ply_path = train_gaussian_splatting(
                colmap_dir,
                str(gs_output_dir),
                iterations=iterations,
                log_path=train_log_path,
                quiet=not diagnostics_enabled,
            )

            if train_log_path is not None and train_log_path.exists():
                diagnostic_artifacts.append(
                    {
                        "path": str(train_log_path),
                        "name": "3dgs_train.log",
                        "content_type": "text/plain",
                    }
                )

            if ply_path and diagnostics_enabled and options.get("splat_eval_train_views", True):
                render_log_path = diagnostics_dir / "3dgs_render_train.log"
                render_dirs = render_gaussian_splatting_train_views(
                    gs_output_dir,
                    iteration=iterations,
                    log_path=render_log_path,
                )
                if render_log_path.exists():
                    diagnostic_artifacts.append(
                        {
                            "path": str(render_log_path),
                            "name": "3dgs_render_train.log",
                            "content_type": "text/plain",
                        }
                    )
                if render_dirs:
                    metrics = compute_render_metrics(
                        render_dirs["renders_dir"],
                        render_dirs["gt_dir"],
                    )
                    info["train_view_metrics"] = metrics
                    metrics_path = diagnostics_dir / "train_view_metrics.json"
                    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    diagnostic_artifacts.append(
                        {
                            "path": str(metrics_path),
                            "name": "train_view_metrics.json",
                            "content_type": "application/json",
                        }
                    )
                    renders_archive = diagnostics_dir / "train_view_renders.tar.gz"
                    _archive_directory(Path(gs_output_dir) / "train" / f"ours_{iterations}", renders_archive)
                    diagnostic_artifacts.append(
                        {
                            "path": str(renders_archive),
                            "name": "train_view_renders.tar.gz",
                            "content_type": "application/gzip",
                        }
                    )

        if not ply_path:
            info["error"] = "train_gaussian_splatting returned no PLY"
            info["diagnostic_artifacts"] = diagnostic_artifacts
            return info

        gz_path = gzip_file(ply_path)
        info.update(
            {
                "ply_path": ply_path,
                "gz_path": gz_path,
                "iterations": iterations,
                "duration_seconds": round(time.time() - splat_start, 3),
                "size_mb_raw": round(os.path.getsize(ply_path) / 1e6, 2),
                "size_mb_gz": round(os.path.getsize(gz_path) / 1e6, 2),
                "diagnostic_artifacts": diagnostic_artifacts,
            }
        )
        return info
    except Exception as e:
        info["error"] = f"{type(e).__name__}: {e}"
        print(f"3DGS phase failed: {info['error']}", flush=True)
        import traceback
        traceback.print_exc()
        return info


def _archive_directory(src_dir: Path, dst_path: Path) -> None:
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst_path, "w:gz") as archive:
        archive.add(src_dir, arcname=src_dir.name)
