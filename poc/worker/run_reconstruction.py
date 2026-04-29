from __future__ import annotations

from argparse import Namespace
from pathlib import Path
import json
import os
import time

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

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
    glb_path = output_dir / "scene.glb"
    export = export_predictions_glb_file(
        prepared,
        str(glb_path),
        conf_threshold=float(options.get("conf_threshold", 1.5)),
        downsample_factor=int(options.get("downsample_factor", 10)),
    )

    metadata = {
        "source": source,
        "options": dict(options),
        "frame_count": len(paths),
        "model_path": model_path,
        "duration_seconds": round(time.time() - start, 3),
        "export": export,
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "scene_path": str(glb_path),
        "metadata_path": str(metadata_path),
        "metadata": metadata,
    }
