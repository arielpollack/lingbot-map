"""3D Gaussian Splatting helpers — COLMAP export + train.py invocation.

Ported from ~/develop/vitour/packages/gpu-worker/handler.py (rotation_matrix_to_quaternion,
export_colmap_workspace, train_gaussian_splatting). Same algorithm; only the
default iteration count is changed (7000 instead of 30000) and the module is
isolated from the rest of the worker so it can be tested without GPU deps.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any


CONF_THRESHOLD = 1.5  # absolute conf cutoff for COLMAP init points


def rotation_matrix_to_quaternion(R):
    """Convert a 3x3 rotation matrix to quaternion (qw, qx, qy, qz).

    Uses the Shepperd method with trace-based branching for numerical stability.
    """
    import numpy as np

    trace = R[0, 0] + R[1, 1] + R[2, 2]

    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        qw = 0.25 / s
        qx = (R[2, 1] - R[1, 2]) * s
        qy = (R[0, 2] - R[2, 0]) * s
        qz = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        qw = (R[2, 1] - R[1, 2]) / s
        qx = 0.25 * s
        qy = (R[0, 1] + R[1, 0]) / s
        qz = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        qw = (R[0, 2] - R[2, 0]) / s
        qx = (R[0, 1] + R[1, 0]) / s
        qy = 0.25 * s
        qz = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        qw = (R[1, 0] - R[0, 1]) / s
        qx = (R[0, 2] + R[2, 0]) / s
        qy = (R[1, 2] + R[2, 1]) / s
        qz = 0.25 * s

    return qw, qx, qy, qz


def _to_numpy(x):
    """Accept either torch.Tensor or numpy.ndarray, return ndarray."""
    import numpy as np

    if hasattr(x, "cpu"):
        return x.cpu().numpy()
    return np.asarray(x)


def export_colmap_workspace(
    intrinsics,
    colmap_extrinsics,
    world_points,
    gt_images,
    confidence,
    colmap_dir,
    *,
    conf_threshold: float = CONF_THRESHOLD,
    target_points: int = 50_000,
    random_seed: int | None = None,
):
    """Convert reconstruction outputs to a COLMAP text-format workspace.

    Args:
        intrinsics: [S, 3, 3] camera intrinsics (for preprocessed image size).
        colmap_extrinsics: [S, 4, 4] camera-from-world extrinsics (w2c).
        world_points: [S, H, W, 3] per-pixel world points (already in world coords).
        gt_images: [S, H, W, 3] RGB images (0-1 range, preprocessed size).
        confidence: [S, H, W] confidence map.
        colmap_dir: output directory.
    """
    import cv2
    import numpy as np

    gt_images_np = _to_numpy(gt_images)
    S, H, W, _ = gt_images_np.shape

    images_dir = os.path.join(colmap_dir, "images")
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    print(f"Saving {S} images to {images_dir}...")
    for i in range(S):
        img_rgb = (gt_images_np[i].clip(0, 1) * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(images_dir, f"{i:06d}.png"), img_bgr)

    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {S}\n")
        for i in range(S):
            K = _to_numpy(intrinsics[i])
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            camera_id = i + 1
            f.write(f"{camera_id} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")
    print(f"Wrote {cameras_path}")

    images_path = os.path.join(sparse_dir, "images.txt")
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {S}\n")
        for i in range(S):
            w2c = _to_numpy(colmap_extrinsics[i])
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
            image_id = i + 1
            camera_id = i + 1
            name = f"{i:06d}.png"
            f.write(f"{image_id} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} {camera_id} {name}\n")
            f.write("\n")  # empty POINTS2D line
    print(f"Wrote {images_path}")

    # Subsample high-confidence points for COLMAP init.
    points_path = os.path.join(sparse_dir, "points3D.txt")

    wp_np = _to_numpy(world_points)
    conf_np = _to_numpy(confidence)
    cols_np = gt_images_np

    all_pts = []
    all_cols = []
    for i in range(S):
        pts = wp_np[i]
        cols = cols_np[i]
        conf = conf_np[i]
        valid = (
            np.isfinite(pts).all(axis=-1)
            & (np.abs(pts).max(axis=-1) < 100)
            & (conf > conf_threshold)
        )
        all_pts.append(pts[valid].astype(np.float64))
        all_cols.append((cols[valid].clip(0, 1) * 255).astype(np.uint8))

    pts_all = np.concatenate(all_pts, axis=0)
    cols_all = np.concatenate(all_cols, axis=0)
    print(f"  COLMAP points3D: {pts_all.shape[0]} valid points before subsampling")
    valid_points_before_subsampling = int(pts_all.shape[0])

    if pts_all.shape[0] > target_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(pts_all.shape[0], target_points, replace=False)
        pts_all = pts_all[idx]
        cols_all = cols_all[idx]

    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {pts_all.shape[0]}\n")
        for j in range(pts_all.shape[0]):
            point_id = j + 1
            x, y, z = pts_all[j]
            r, g, b = cols_all[j]
            f.write(f"{point_id} {x} {y} {z} {r} {g} {b} 0.0\n")
    print(f"Wrote {points_path} ({pts_all.shape[0]} points)")
    print(f"COLMAP workspace exported to {colmap_dir}")
    return {
        "conf_threshold": float(conf_threshold),
        "target_points": int(target_points),
        "valid_points_before_subsampling": valid_points_before_subsampling,
        "points_written": int(pts_all.shape[0]),
        "random_seed": random_seed,
    }


def export_colmap_workspace_from_points(
    intrinsics,
    colmap_extrinsics,
    gt_images,
    init_points,
    init_colors,
    colmap_dir,
    *,
    target_points: int = 50_000,
    random_seed: int | None = None,
):
    """COLMAP exporter for the lidar_mesh tier.

    Same image / cameras / images outputs as `export_colmap_workspace`, but
    `points3D.txt` is written from a supplied `(init_points, init_colors)`
    pair instead of from per-pixel depth-unprojected world points. The
    lidar_mesh tier samples its init points from the ARKit fused mesh
    surface — denser and cleaner than what a depth-unproject pass would
    produce.

    Args:
        intrinsics: [S, 3, 3] OpenCV-style camera intrinsics.
        colmap_extrinsics: [S, 4, 4] world → camera (w2c).
        gt_images: [S, H, W, 3] RGB. uint8 OR float in [0, 1].
        init_points: [N, 3] 3D world points to seed the splat.
        init_colors: [N, 3] uint8 RGB seed colors.
        colmap_dir: output directory.
    """
    import cv2
    import numpy as np

    gt_images_np = _to_numpy(gt_images)
    if gt_images_np.dtype != np.uint8:
        gt_images_np = (np.clip(gt_images_np, 0.0, 1.0) * 255.0).astype(np.uint8)
    S, H, W, _ = gt_images_np.shape

    images_dir = os.path.join(colmap_dir, "images")
    sparse_dir = os.path.join(colmap_dir, "sparse", "0")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(sparse_dir, exist_ok=True)

    print(f"[gsplat-mesh] saving {S} images → {images_dir}", flush=True)
    for i in range(S):
        img_bgr = cv2.cvtColor(gt_images_np[i], cv2.COLOR_RGB2BGR)
        cv2.imwrite(os.path.join(images_dir, f"{i:06d}.png"), img_bgr)

    cameras_path = os.path.join(sparse_dir, "cameras.txt")
    with open(cameras_path, "w") as f:
        f.write("# Camera list with one line of data per camera:\n")
        f.write("# CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n")
        f.write(f"# Number of cameras: {S}\n")
        for i in range(S):
            K = _to_numpy(intrinsics[i])
            fx, fy = K[0, 0], K[1, 1]
            cx, cy = K[0, 2], K[1, 2]
            f.write(f"{i + 1} PINHOLE {W} {H} {fx} {fy} {cx} {cy}\n")

    images_path = os.path.join(sparse_dir, "images.txt")
    with open(images_path, "w") as f:
        f.write("# Image list with two lines of data per image:\n")
        f.write("# IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
        f.write("# POINTS2D[] as (X, Y, POINT3D_ID)\n")
        f.write(f"# Number of images: {S}\n")
        for i in range(S):
            w2c = _to_numpy(colmap_extrinsics[i])
            R = w2c[:3, :3]
            t = w2c[:3, 3]
            qw, qx, qy, qz = rotation_matrix_to_quaternion(R)
            f.write(
                f"{i + 1} {qw} {qx} {qy} {qz} {t[0]} {t[1]} {t[2]} "
                f"{i + 1} {i:06d}.png\n"
            )
            f.write("\n")

    pts = np.asarray(init_points, dtype=np.float64)
    cols = np.asarray(init_colors, dtype=np.uint8)
    if cols.shape[0] != pts.shape[0]:
        raise ValueError(
            f"init_points/init_colors length mismatch: {pts.shape[0]} vs {cols.shape[0]}"
        )
    n_total = int(pts.shape[0])
    if pts.shape[0] > target_points:
        rng = np.random.default_rng(random_seed)
        idx = rng.choice(pts.shape[0], target_points, replace=False)
        pts = pts[idx]
        cols = cols[idx]

    points_path = os.path.join(sparse_dir, "points3D.txt")
    with open(points_path, "w") as f:
        f.write("# 3D point list with one line of data per point:\n")
        f.write("# POINT3D_ID, X, Y, Z, R, G, B, ERROR, TRACK[]\n")
        f.write(f"# Number of points: {pts.shape[0]}\n")
        for j in range(pts.shape[0]):
            x, y, z = pts[j]
            r, g, b = cols[j]
            f.write(f"{j + 1} {x} {y} {z} {r} {g} {b} 0.0\n")

    print(
        f"[gsplat-mesh] points3D.txt: {pts.shape[0]} (sampled from {n_total})",
        flush=True,
    )
    return {
        "frames": int(S),
        "init_points_total": n_total,
        "init_points_written": int(pts.shape[0]),
        "random_seed": random_seed,
    }


def train_gaussian_splatting(
    colmap_dir: str | Path,
    output_dir: str | Path,
    iterations: int = 7000,
    sh_degree: int = 3,
    timeout_s: int = 1800,
    train_script: str = "/opt/gaussian-splatting/train.py",
    log_path: str | Path | None = None,
    quiet: bool = True,
) -> str | None:
    """Run 3D Gaussian Splatting training on a COLMAP workspace.

    Args:
        colmap_dir: path to COLMAP workspace (with images/ and sparse/0/).
        output_dir: directory to receive the trained model.
        iterations: number of training iterations (POC default 7000).
        sh_degree: spherical-harmonics degree (3 = full color fidelity).
        timeout_s: subprocess timeout.
        train_script: path to graphdeco-inria train.py inside the image.

    Returns:
        Absolute path to the resulting `point_cloud.ply`, or None on failure.
    """
    output_dir = str(output_dir)
    output_ply = os.path.join(
        output_dir, "point_cloud", f"iteration_{iterations}", "point_cloud.ply"
    )

    cmd = [
        sys.executable, train_script,
        "-s", str(colmap_dir),
        "--model_path", output_dir,
        "--iterations", str(iterations),
        "--sh_degree", str(sh_degree),
    ]
    if quiet:
        cmd.append("--quiet")
    print(f"Starting 3DGS training: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"3DGS training timed out after {timeout_s}s", flush=True)
        return None
    except Exception as e:
        print(f"3DGS training error: {e}", flush=True)
        return None

    if log_path is not None:
        _write_subprocess_log(log_path, cmd, result.stdout, result.stderr)

    if result.returncode != 0:
        print(f"3DGS training failed (exit code {result.returncode})", flush=True)
        if result.stdout:
            print(f"  stdout (last 2000 chars): {result.stdout[-2000:]}", flush=True)
        if result.stderr:
            print(f"  stderr (last 2000 chars): {result.stderr[-2000:]}", flush=True)
        return None

    if os.path.exists(output_ply):
        print(f"3DGS training complete: {output_ply}", flush=True)
        return output_ply
    print(f"3DGS training finished but output PLY not found: {output_ply}", flush=True)
    return None


def render_gaussian_splatting_train_views(
    model_dir: str | Path,
    iteration: int,
    timeout_s: int = 1800,
    render_script: str = "/opt/gaussian-splatting/render.py",
    log_path: str | Path | None = None,
) -> dict[str, str] | None:
    """Render Graphdeco train views for the trained model and return output dirs."""
    model_dir = str(model_dir)
    cmd = [
        sys.executable, render_script,
        "-m", model_dir,
        "--iteration", str(iteration),
        "--skip_test",
    ]
    print(f"Rendering 3DGS train views: {' '.join(cmd)}", flush=True)

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        print(f"3DGS train-view render timed out after {timeout_s}s", flush=True)
        return None
    except Exception as e:
        print(f"3DGS train-view render error: {e}", flush=True)
        return None

    if log_path is not None:
        _write_subprocess_log(log_path, cmd, result.stdout, result.stderr)

    if result.returncode != 0:
        print(f"3DGS train-view render failed (exit code {result.returncode})", flush=True)
        if result.stdout:
            print(f"  stdout (last 2000 chars): {result.stdout[-2000:]}", flush=True)
        if result.stderr:
            print(f"  stderr (last 2000 chars): {result.stderr[-2000:]}", flush=True)
        return None

    train_dir = Path(model_dir) / "train" / f"ours_{iteration}"
    renders_dir = train_dir / "renders"
    gt_dir = train_dir / "gt"
    if renders_dir.exists() and gt_dir.exists():
        return {"renders_dir": str(renders_dir), "gt_dir": str(gt_dir)}
    print(f"3DGS train-view render finished but output dirs missing: {train_dir}", flush=True)
    return None


def compute_render_metrics(renders_dir: str | Path, gt_dir: str | Path) -> dict[str, float | int]:
    """Compute simple RGB train-view metrics from Graphdeco render outputs."""
    import math
    import numpy as np
    from PIL import Image

    renders_dir = Path(renders_dir)
    gt_dir = Path(gt_dir)
    render_names = {path.name for path in renders_dir.glob("*.png")}
    gt_names = {path.name for path in gt_dir.glob("*.png")}
    names = sorted(render_names & gt_names)
    if not names:
        return {"frame_count": 0, "mae": None, "mse": None, "psnr": None}

    maes = []
    mses = []
    for name in names:
        render = np.asarray(Image.open(renders_dir / name).convert("RGB"), dtype=np.float64) / 255.0
        gt = np.asarray(Image.open(gt_dir / name).convert("RGB"), dtype=np.float64) / 255.0
        diff = render - gt
        maes.append(float(np.mean(np.abs(diff))))
        mses.append(float(np.mean(diff * diff)))

    mae = float(np.mean(maes))
    mse = float(np.mean(mses))
    psnr = math.inf if mse == 0 else 20.0 * math.log10(1.0 / math.sqrt(mse))
    return {
        "frame_count": len(names),
        "mae": mae,
        "mse": mse,
        "psnr": psnr,
    }


def viewer_camera_from_colmap(
    colmap_extrinsics,
    intrinsics,
    image_size: tuple[int, int],
    scene_points,
    frame_index: int = 0,
) -> dict[str, Any]:
    """Create a Three.js-friendly starting camera from a COLMAP OpenCV w2c pose."""
    import numpy as np

    w2c_all = _to_numpy(colmap_extrinsics).astype(np.float64)
    K_all = _to_numpy(intrinsics).astype(np.float64)
    points = _to_numpy(scene_points).reshape(-1, 3).astype(np.float64)

    frame_index = max(0, min(int(frame_index), len(w2c_all) - 1))
    c2w_cv = np.linalg.inv(w2c_all[frame_index])
    cv_to_gl = np.eye(4, dtype=np.float64)
    cv_to_gl[1, 1] = -1.0
    cv_to_gl[2, 2] = -1.0
    c2w_gl = c2w_cv @ cv_to_gl

    position = c2w_gl[:3, 3]
    forward = -c2w_gl[:3, 2]
    forward = forward / max(np.linalg.norm(forward), 1e-12)
    up = c2w_gl[:3, 1]
    up = up / max(np.linalg.norm(up), 1e-12)
    target = position + forward

    finite = points[np.isfinite(points).all(axis=1)]
    if finite.size:
        lo = np.percentile(finite, 5, axis=0)
        hi = np.percentile(finite, 95, axis=0)
        scene_extent = float(max(np.linalg.norm(hi - lo), 1.0))
    else:
        scene_extent = 10.0

    height, _width = image_size
    fy = float(K_all[frame_index, 1, 1])
    fov_degrees = float(2.0 * np.degrees(np.arctan(float(height) / (2.0 * fy))))

    return {
        "frame_index": frame_index,
        "position": [float(v) for v in position],
        "target": [float(v) for v in target],
        "up": [float(v) for v in up],
        "fov_degrees": fov_degrees,
        "near": 0.01,
        "far": float(max(1000.0, scene_extent * 100.0)),
        "move_speed": float(max(0.25, scene_extent * 0.15)),
    }


def _write_subprocess_log(log_path: str | Path, cmd: list[str], stdout: str | None, stderr: str | None) -> None:
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(
        "$ "
        + " ".join(cmd)
        + "\n\n[stdout]\n"
        + (stdout or "")
        + "\n\n[stderr]\n"
        + (stderr or "")
        + "\n",
        encoding="utf-8",
    )


def reprojection_sanity(
    world_points,
    intrinsics,
    colmap_extrinsics,
    confidence=None,
    max_samples_per_frame: int = 2000,
) -> dict:
    """Check whether `world_points` reproject onto their source pixels via the
    provided w2c extrinsics + intrinsics. Median pixel error should be small
    (~0-2 px) if the conventions all line up. Large errors mean the world
    points aren't in the same coord system as the cameras, OR the cameras
    aren't actually w2c.

    Returns a dict {positive_depth_pct, median_px, p95_px, samples}.
    Ported from ~/develop/vitour/packages/gpu-worker/handler.py:348.
    """
    import numpy as np

    wp = _to_numpy(world_points)
    K_all = _to_numpy(intrinsics)
    w2c_all = _to_numpy(colmap_extrinsics)
    conf = _to_numpy(confidence) if confidence is not None else None

    S, H, W, _ = wp.shape
    errors = []
    positive_depth = 0
    total = 0

    for i in range(S):
        pts = wp[i]
        valid = np.isfinite(pts).all(axis=-1)
        if conf is not None:
            valid = valid & (conf[i] > CONF_THRESHOLD)
        if not valid.any():
            continue

        xy = np.argwhere(valid)
        if xy.shape[0] > max_samples_per_frame:
            step = max(1, xy.shape[0] // max_samples_per_frame)
            xy = xy[::step][:max_samples_per_frame]

        sample_pts = pts[xy[:, 0], xy[:, 1]].astype(np.float64)
        w2c = w2c_all[i]
        R = w2c[:3, :3]
        t = w2c[:3, 3]
        cam = sample_pts @ R.T + t
        z = cam[:, 2]
        depth_valid = z > 1e-6

        K = K_all[i].astype(np.float64)
        z_safe = np.maximum(z, 1e-6)
        u = K[0, 0] * (cam[:, 0] / z_safe) + K[0, 2]
        v = K[1, 1] * (cam[:, 1] / z_safe) + K[1, 2]
        source_x = xy[:, 1].astype(np.float64)
        source_y = xy[:, 0].astype(np.float64)
        err = np.sqrt((u - source_x) ** 2 + (v - source_y) ** 2)

        if depth_valid.any():
            errors.append(err[depth_valid])
        positive_depth += int(depth_valid.sum())
        total += int(depth_valid.size)

    if not errors:
        return {"samples": 0, "median_px": None, "p95_px": None, "positive_depth_pct": 0.0}

    all_errors = np.concatenate(errors)
    return {
        "samples": int(all_errors.size),
        "median_px": float(np.median(all_errors)),
        "p95_px": float(np.quantile(all_errors, 0.95)),
        "positive_depth_pct": 100.0 * positive_depth / max(1, total),
    }


def gzip_file(src_path: str | Path) -> str:
    """gzip a file in place (creates `<src>.gz`); return the gz path."""
    import gzip
    import shutil

    src = str(src_path)
    dst = src + ".gz"
    with open(src, "rb") as f_in, gzip.open(dst, "wb") as f_out:
        shutil.copyfileobj(f_in, f_out)
    return dst
