"""gsplat-based 3D Gaussian Splatting trainer with depth supervision.

Replaces the graphdeco-inria subprocess `train.py` for the `lidar_mesh`
tier. Trains in-process using the `gsplat` library
(nerfstudio-project/gsplat) so we can:

  • supervise with the on-device LiDAR depth maps (graphdeco's train.py
    has no depth loss path);
  • train at sh_degree=0 (no spherical harmonics) for indoor diffuse
    surfaces — ~80% smaller PLY/`.splat` and matches the `.splat` format
    output anyway;
  • avoid subprocess + COLMAP-workspace I/O — significant fixed-cost
    overhead that doesn't scale with iteration count.

Output format: numpy arrays in REAL space (scales = meters, opacities ∈
[0, 1], quats unit-norm wxyz). The serialization step
(`splat_format.encode_splat_file`) is separate.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

import numpy as np


# Confidence values are 0 (low), 1 (medium), 2 (high). We mask depth loss
# to medium+high samples; LiDAR returns 0 on glossy surfaces, far range,
# and edges — sites where the depth is unreliable enough that supervising
# from it actively pulls the splat off the true surface.
DEPTH_CONF_MIN = 1


@dataclass
class Camera:
    """Single training camera. All arrays at the SAME resolution.

    For the lidar_mesh tier, the calling code downsamples the RGB to the
    LiDAR depth resolution (192 × 256 by default) and scales the intrinsics
    accordingly. Training at depth resolution is ~50× cheaper than at
    1280 × 720 and matches the depth supervision natively.
    """
    image: np.ndarray              # (H, W, 3) uint8 RGB
    intrinsic: np.ndarray          # (3, 3) f32 OpenCV K at this resolution
    extrinsic_w2c: np.ndarray      # (4, 4) f32 OpenCV w2c (world → camera)
    depth: np.ndarray | None       # (H, W) f32 meters, OR None
    depth_conf: np.ndarray | None  # (H, W) u8 ARConfidence 0..2, OR None


def train_lidar_mesh_splat(
    *,
    init_xyz: np.ndarray,
    init_rgb: np.ndarray,
    cameras: list[Camera],
    iterations: int = 7000,
    depth_lambda: float = 0.2,
    densify_grad_threshold: float = 0.0004,
    init_opacity: float = 0.1,
    init_scale_meters: float | None = None,
    device: str = "cuda",
    log_every: int = 200,
) -> dict[str, Any]:
    """Train 3DGS via gsplat library, optionally with LiDAR depth supervision.

    Args:
        init_xyz:  (N, 3) f32 world positions to seed the Gaussians.
            Typically sampled uniformly from the ARKit mesh surface.
        init_rgb:  (N, 3) uint8 seed colors. Mid-gray is fine; the trainer
            overrides them within the first few hundred iterations from the
            ground-truth images.
        cameras:   list of `Camera` objects, all at the same H × W. RGB +
            intrinsic + extrinsic must be provided; depth + conf are
            optional (per camera). Cameras without depth simply skip the
            depth-loss term that step.
        iterations: training step count.
        depth_lambda: scale factor on the depth-L1 term in the per-iter
            loss. 0.0 disables depth supervision entirely. Tuning: too high
            freezes the splat to the LiDAR estimate (poor on
            thin/transparent surfaces); too low yields no benefit. Start at
            0.2.
        densify_grad_threshold: gsplat `DefaultStrategy.grow_grad2d`. The
            graphdeco default is 0.0002, which densifies aggressively and
            is the reason their PLYs balloon to 30–100 MB. We use 0.0004
            (half as aggressive) for tighter Gaussian counts.
        init_opacity: initial REAL opacity for every Gaussian (default 0.1).
        init_scale_meters: initial REAL scale (sigma in meters) per axis.
            If None, derived from mean nearest-neighbour distance among the
            init points (a heuristic that scales with point density).
        device: torch device.
        log_every: print loss every N iterations (set to 0 to silence).

    Returns:
        dict with REAL-space arrays:
            "means":     (N, 3) f32
            "quats_wxyz": (N, 4) f32 unit-norm
            "scales":    (N, 3) f32 meters
            "opacities": (N,)   f32 in [0, 1]
            "rgb":       (N, 3) uint8
            "stats":     dict (iterations, final_gaussians, duration_s,
                              loss_history, depth_supervision)
    """
    # Lazy imports — these pull in CUDA-bound libraries that don't exist
    # outside the worker container, so importing eagerly would break tests
    # on the dev machine.
    import torch
    import torch.nn.functional as F
    from gsplat.rendering import rasterization
    from gsplat.strategy import DefaultStrategy

    if init_xyz.shape[0] == 0:
        raise ValueError("init_xyz is empty — need at least one seed point")
    if init_xyz.shape[0] != init_rgb.shape[0]:
        raise ValueError(
            f"init_xyz/init_rgb length mismatch: {init_xyz.shape[0]} vs {init_rgb.shape[0]}"
        )
    if not cameras:
        raise ValueError("Need at least one camera")

    n_cams = len(cameras)
    H, W = cameras[0].image.shape[:2]
    for cam in cameras:
        if cam.image.shape[:2] != (H, W):
            raise ValueError(
                f"All cameras must share H × W; got mix of "
                f"{cam.image.shape[:2]} vs {(H, W)}"
            )

    has_depth = any(cam.depth is not None for cam in cameras)
    use_depth = has_depth and depth_lambda > 0
    if not use_depth:
        print(
            "[gsplat-train] depth supervision DISABLED "
            f"(has_depth={has_depth}, depth_lambda={depth_lambda})",
            flush=True,
        )

    # ── Initial Gaussian parameters ────────────────────────────────────────
    n_init = int(init_xyz.shape[0])
    if init_scale_meters is None:
        init_scale_meters = _estimate_init_scale(init_xyz)
    print(
        f"[gsplat-train] init: {n_init} gaussians, "
        f"scale={init_scale_meters:.4f} m, opacity={init_opacity}",
        flush=True,
    )

    means = torch.tensor(init_xyz, dtype=torch.float32, device=device)
    # Identity quaternions (w=1, x=y=z=0) — let the optimizer rotate them.
    quats = torch.zeros((n_init, 4), dtype=torch.float32, device=device)
    quats[:, 0] = 1.0
    # Log-scale storage (gsplat convention: stored = log(real); applied =
    # exp(stored) before rasterization).
    log_scale = math.log(init_scale_meters)
    scales = torch.full((n_init, 3), log_scale, dtype=torch.float32, device=device)
    # Logit storage for opacity (stored = logit(real); applied = sigmoid).
    logit_init = math.log(init_opacity / (1.0 - init_opacity))
    opacities = torch.full((n_init,), logit_init, dtype=torch.float32, device=device)
    # DC-only colors in [0, 1].
    colors = torch.tensor(
        init_rgb.astype(np.float32) / 255.0,
        dtype=torch.float32,
        device=device,
    )

    params = torch.nn.ParameterDict(
        {
            "means": torch.nn.Parameter(means),
            "scales": torch.nn.Parameter(scales),
            "quats": torch.nn.Parameter(quats),
            "opacities": torch.nn.Parameter(opacities),
            "colors": torch.nn.Parameter(colors),
        }
    )

    # Per-param Adam optimizers (gsplat's strategy mutates params in-place
    # during densify/prune and expects to find the matching optimizer).
    # Learning rates from gsplat's simple_trainer.py reference.
    optimizers = {
        "means": torch.optim.Adam([params["means"]], lr=1.6e-4, eps=1e-15),
        "scales": torch.optim.Adam([params["scales"]], lr=5e-3),
        "quats": torch.optim.Adam([params["quats"]], lr=1e-3),
        "opacities": torch.optim.Adam([params["opacities"]], lr=5e-2),
        "colors": torch.optim.Adam([params["colors"]], lr=2.5e-3),
    }

    # ── Densify + prune strategy ───────────────────────────────────────────
    # Rough scene scale = max coord extent of init points.
    init_xyz_t = torch.tensor(init_xyz, dtype=torch.float32)
    extent = float((init_xyz_t.max(dim=0).values - init_xyz_t.min(dim=0).values).max().item())
    scene_scale = max(extent, 1.0)

    # Cap refine_stop_iter at most ~75% of total — past that we want pure
    # gradient updates without topology changes.
    refine_stop_iter = int(min(15_000, max(int(iterations * 0.75), 100)))

    strategy = DefaultStrategy(
        grow_grad2d=densify_grad_threshold,
        grow_scale3d=0.01,
        prune_opa=0.005,
        prune_scale3d=0.1,
        refine_start_iter=500,
        refine_stop_iter=refine_stop_iter,
        refine_every=100,
        reset_every=3000,
        absgrad=False,
        revised_opacity=False,
        verbose=False,
    )
    try:
        strategy.check_sanity(params, optimizers)
    except Exception as exc:
        print(f"[gsplat-train] strategy.check_sanity warning: {exc}", flush=True)
    state = strategy.initialize_state(scene_scale=scene_scale)

    # ── Pre-upload camera tensors (small at depth-res, fits comfortably) ──
    cam_tensors = []
    for cam in cameras:
        gt_rgb = torch.tensor(
            cam.image.astype(np.float32) / 255.0,
            dtype=torch.float32,
            device=device,
        )  # (H, W, 3)
        K = torch.tensor(cam.intrinsic, dtype=torch.float32, device=device)  # (3, 3)
        viewmat = torch.tensor(
            cam.extrinsic_w2c, dtype=torch.float32, device=device
        )  # (4, 4)
        if cam.depth is not None:
            gt_depth = torch.tensor(cam.depth, dtype=torch.float32, device=device)  # (H, W)
            if cam.depth_conf is not None:
                conf = torch.tensor(
                    cam.depth_conf, dtype=torch.float32, device=device
                )
                conf_mask = (conf >= float(DEPTH_CONF_MIN)).float()
            else:
                conf_mask = torch.ones_like(gt_depth)
            # Only consider valid (positive) depth values; sometimes ARKit
            # writes 0.0 in unmeasured regions.
            conf_mask = conf_mask * (gt_depth > 0).float()
        else:
            gt_depth = None
            conf_mask = None
        cam_tensors.append(
            {
                "gt_rgb": gt_rgb,
                "K": K,
                "viewmat": viewmat,
                "gt_depth": gt_depth,
                "conf_mask": conf_mask,
            }
        )

    # ── Training loop ──────────────────────────────────────────────────────
    loss_history: list[tuple[int, float]] = []
    rng = np.random.default_rng(0)
    start = time.time()

    rgb_l1_w = 0.8
    rgb_ssim_w = 0.2

    for step in range(iterations):
        cam_idx = int(rng.integers(0, n_cams))
        cam = cam_tensors[cam_idx]

        # Apply param transforms.
        means_t = params["means"]
        quats_t = torch.nn.functional.normalize(params["quats"], dim=-1)
        scales_t = torch.exp(params["scales"])
        opacities_t = torch.sigmoid(params["opacities"])
        colors_t = params["colors"]  # already in [0, 1] (will be clamped by loss)

        renders, _alphas, info = rasterization(
            means=means_t,
            quats=quats_t,
            scales=scales_t,
            opacities=opacities_t,
            colors=colors_t,
            viewmats=cam["viewmat"][None],
            Ks=cam["K"][None],
            width=W,
            height=H,
            render_mode="RGB+ED" if use_depth else "RGB",
            sh_degree=None,
            packed=False,
            near_plane=0.01,
            far_plane=100.0,
        )
        # renders shape: (1, H, W, 3) for "RGB" or (1, H, W, 4) for "RGB+ED".
        rendered_rgb = renders[0, ..., :3]  # (H, W, 3)
        if use_depth and cam["gt_depth"] is not None:
            rendered_depth = renders[0, ..., 3]  # (H, W)
        else:
            rendered_depth = None

        # Pre-backward — strategy hooks need raw render `info` for grad tracking.
        strategy.step_pre_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
        )

        # RGB loss = L1 + (1 - SSIM). SSIM is the cheap structural one — we
        # avoid pulling in fused-ssim here to keep the trainer self-contained
        # at depth resolution; standard L1 dominates anyway at small res.
        rgb_l1 = (rendered_rgb - cam["gt_rgb"]).abs().mean()
        ssim_val = _ssim_simple(rendered_rgb, cam["gt_rgb"])
        rgb_loss = rgb_l1_w * rgb_l1 + rgb_ssim_w * (1.0 - ssim_val)

        if rendered_depth is not None:
            mask = cam["conf_mask"]
            mask_sum = mask.sum().clamp(min=1.0)
            depth_diff = (rendered_depth - cam["gt_depth"]).abs() * mask
            depth_loss = depth_diff.sum() / mask_sum
            loss = rgb_loss + depth_lambda * depth_loss
        else:
            depth_loss = torch.zeros((), device=device)
            loss = rgb_loss

        loss.backward()

        # Optimizer step (per param so the strategy's later state surgery
        # doesn't trip a single-optimizer's expectation about its param set).
        for opt in optimizers.values():
            opt.step()
            opt.zero_grad(set_to_none=True)

        strategy.step_post_backward(
            params=params,
            optimizers=optimizers,
            state=state,
            step=step,
            info=info,
            packed=False,
        )

        if log_every and (step % log_every == 0 or step == iterations - 1):
            n_now = int(params["means"].shape[0])
            print(
                f"[gsplat-train] step={step:5d}  loss={float(loss):.4f}  "
                f"rgb={float(rgb_loss):.4f}  depth={float(depth_loss):.4f}  "
                f"gauss={n_now}",
                flush=True,
            )
            loss_history.append((int(step), float(loss)))

    duration_s = round(time.time() - start, 3)

    # ── Convert params to REAL-space numpy and return ──────────────────────
    with torch.no_grad():
        out_means = params["means"].detach().cpu().numpy().astype(np.float32)
        out_quats = (
            torch.nn.functional.normalize(params["quats"], dim=-1)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        out_scales = (
            torch.exp(params["scales"]).detach().cpu().numpy().astype(np.float32)
        )
        out_opacities = (
            torch.sigmoid(params["opacities"]).detach().cpu().numpy().astype(np.float32)
        )
        out_rgb = (
            (params["colors"].clamp(0.0, 1.0) * 255.0)
            .detach()
            .cpu()
            .numpy()
            .round()
            .astype(np.uint8)
        )

    n_final = int(out_means.shape[0])
    print(
        f"[gsplat-train] DONE iters={iterations} "
        f"final_gaussians={n_final} duration_s={duration_s} "
        f"depth_supervision={'on' if use_depth else 'off'}",
        flush=True,
    )

    return {
        "means": out_means,
        "quats_wxyz": out_quats,
        "scales": out_scales,
        "opacities": out_opacities,
        "rgb": out_rgb,
        "stats": {
            "iterations": int(iterations),
            "init_gaussians": int(n_init),
            "final_gaussians": n_final,
            "duration_s": duration_s,
            "depth_supervision": bool(use_depth),
            "depth_lambda": float(depth_lambda) if use_depth else 0.0,
            "init_scale_meters": float(init_scale_meters),
            "scene_scale": float(scene_scale),
            "loss_history": loss_history,
        },
    }


def _estimate_init_scale(xyz: np.ndarray, k: int = 3, max_samples: int = 5_000) -> float:
    """Rough scale heuristic: mean distance to the kth-nearest neighbour.

    Used as the initial Gaussian sigma. With k=3 we get a 3DGS-style
    "matches local point density" sigma. Subsampled to keep the KDTree
    query bounded.
    """
    from scipy.spatial import cKDTree

    n = xyz.shape[0]
    if n <= 1:
        return 0.05
    if n > max_samples:
        idx = np.random.default_rng(0).choice(n, max_samples, replace=False)
        sample = xyz[idx]
    else:
        sample = xyz

    tree = cKDTree(sample.astype(np.float32))
    # k+1 because the first NN of a point is itself.
    distances, _ = tree.query(sample, k=min(k + 1, len(sample)))
    if distances.ndim == 1:
        # Only one neighbour returned (very small sample); fall back.
        return 0.05
    nn_dist = distances[:, -1]  # k-th NN distance
    return float(max(np.mean(nn_dist), 1e-3))


def _ssim_simple(x: "torch.Tensor", y: "torch.Tensor", *, win: int = 11) -> "torch.Tensor":
    """Simple Gaussian-window SSIM over an HWC float image.

    Implementation cribbed from common torch SSIM blog posts; standard
    SSIM constants. Returns a scalar in roughly [-1, 1] (1 = identical).
    Self-contained so the trainer doesn't need an extra package.
    """
    import torch
    import torch.nn.functional as F

    if x.dim() != 3 or y.dim() != 3:
        raise ValueError(f"_ssim_simple expects (H, W, C); got {x.shape} / {y.shape}")

    # to (1, C, H, W)
    x_b = x.permute(2, 0, 1).unsqueeze(0)
    y_b = y.permute(2, 0, 1).unsqueeze(0)
    C = x_b.shape[1]

    # 1D Gaussian kernel.
    sigma = 1.5
    coords = torch.arange(win, dtype=x.dtype, device=x.device) - (win // 2)
    g = torch.exp(-(coords ** 2) / (2.0 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[:, None] * g[None, :]
    kernel = kernel_2d.expand(C, 1, win, win).contiguous()

    pad = win // 2
    mu_x = F.conv2d(x_b, kernel, padding=pad, groups=C)
    mu_y = F.conv2d(y_b, kernel, padding=pad, groups=C)
    mu_x2 = mu_x * mu_x
    mu_y2 = mu_y * mu_y
    mu_xy = mu_x * mu_y
    sigma_x2 = F.conv2d(x_b * x_b, kernel, padding=pad, groups=C) - mu_x2
    sigma_y2 = F.conv2d(y_b * y_b, kernel, padding=pad, groups=C) - mu_y2
    sigma_xy = F.conv2d(x_b * y_b, kernel, padding=pad, groups=C) - mu_xy

    C1 = (0.01 ** 2)
    C2 = (0.03 ** 2)
    ssim_map = ((2 * mu_xy + C1) * (2 * sigma_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    )
    return ssim_map.mean()
