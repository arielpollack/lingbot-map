"""Encode trained 3D Gaussians to the antimatter15-style `.splat` format.

32 bytes per Gaussian, no spherical harmonics (DC color only). Replaces the
gzipped PLY output of the graphdeco trainer for the `lidar_mesh` tier.

Layout per Gaussian (little-endian):

    offset  size  field
      0      12   position   (3 × f32)
     12      12   scale      (3 × f32, real meters — NOT log)
     24       4   color RGBA (4 × u8)
     28       4   rotation   (4 × u8 quaternion (w,x,y,z), encoded as
                              `(q * 128 + 128).clip(0, 255)`)

Reference: https://github.com/antimatter15/splat/blob/main/convert.py
mkkellogg/gaussian-splats-3d auto-detects this format from the `.splat`
file extension and renders it directly (no SH evaluation).
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np


SPLAT_BYTES_PER_GAUSSIAN = 32


def encode_splat_file(
    *,
    means: np.ndarray,
    quats_wxyz: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    rgb: np.ndarray,
    output_path: str,
    sort_by_distance_to: tuple[float, float, float] | None = None,
) -> dict[str, Any]:
    """Pack `(N, ...)` Gaussian parameters into the `.splat` binary file.

    Args:
        means:        (N, 3) f32 world position.
        quats_wxyz:   (N, 4) f32 rotation as quaternion (w, x, y, z).
                      Caller must pre-normalize; we re-normalize defensively.
        scales:       (N, 3) f32 REAL scale in meters (not log scale).
                      Caller is responsible for `exp(log_scales)` if their
                      params are stored in log space.
        opacities:    (N,) f32 REAL opacity in [0, 1] (not logit).
                      Caller is responsible for `sigmoid(logit_opacities)`.
        rgb:          (N, 3) uint8 in [0, 255].
        output_path:  destination file path.
        sort_by_distance_to: optional (x, y, z) reference point. When set,
                      Gaussians are written in increasing distance from this
                      point — gives a nicer initial appearance before the
                      viewer's GPU sort kicks in. Pass scene-center for the
                      typical "back-to-front from inside" effect.

    Returns:
        dict with `gaussians` (count), `bytes`, `format`, and `sorted` flag.
    """
    means = np.asarray(means, dtype=np.float32)
    quats_wxyz = np.asarray(quats_wxyz, dtype=np.float32)
    scales = np.asarray(scales, dtype=np.float32)
    opacities = np.asarray(opacities, dtype=np.float32).reshape(-1)
    rgb = np.asarray(rgb, dtype=np.uint8)

    n = means.shape[0]
    if (
        means.shape != (n, 3)
        or quats_wxyz.shape != (n, 4)
        or scales.shape != (n, 3)
        or opacities.shape != (n,)
        or rgb.shape != (n, 3)
    ):
        raise ValueError(
            f"Inconsistent shapes for N={n}: means={means.shape}, "
            f"quats={quats_wxyz.shape}, scales={scales.shape}, "
            f"opacities={opacities.shape}, rgb={rgb.shape}"
        )

    # Optional depth-from-reference sort.
    if sort_by_distance_to is not None:
        ref = np.asarray(sort_by_distance_to, dtype=np.float32).reshape(3)
        d2 = np.sum((means - ref) ** 2, axis=1)
        order = np.argsort(d2, kind="stable")
        means = means[order]
        quats_wxyz = quats_wxyz[order]
        scales = scales[order]
        opacities = opacities[order]
        rgb = rgb[order]

    # Re-normalize quaternions defensively. Any zero-length quat becomes
    # identity (1, 0, 0, 0) to avoid producing NaN encoded bytes.
    norms = np.linalg.norm(quats_wxyz, axis=1, keepdims=True)
    safe_norms = np.where(norms > 1e-8, norms, 1.0)
    quats_normed = quats_wxyz / safe_norms
    bad = (norms <= 1e-8).reshape(-1)
    if bad.any():
        quats_normed[bad] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    # Encode rotation: float [-1, 1] → uint8 via (q * 128 + 128).clip(0, 255).
    # This is the antimatter15 convention used by mkkellogg's viewer.
    rot_u8 = np.clip(np.rint(quats_normed * 128.0 + 128.0), 0, 255).astype(np.uint8)

    # Encode color RGBA: alpha = opacity * 255.
    alpha_u8 = np.clip(np.rint(opacities * 255.0), 0, 255).astype(np.uint8)
    rgba_u8 = np.concatenate([rgb, alpha_u8[:, None]], axis=1)

    # Pack into a single (N, 32) buffer using a structured dtype, then
    # write raw bytes. Structured dtype with explicit offsets matches the
    # spec exactly and avoids any per-Gaussian Python loop.
    record_dtype = np.dtype(
        {
            "names": ["pos", "scale", "rgba", "rot"],
            "formats": ["3<f4", "3<f4", "4u1", "4u1"],
            "offsets": [0, 12, 24, 28],
            "itemsize": SPLAT_BYTES_PER_GAUSSIAN,
        }
    )
    buf = np.empty(n, dtype=record_dtype)
    buf["pos"] = means
    buf["scale"] = scales
    buf["rgba"] = rgba_u8
    buf["rot"] = rot_u8

    with open(output_path, "wb") as f:
        f.write(buf.tobytes())

    return {
        "format": "splat",
        "gaussians": int(n),
        "bytes": int(os.path.getsize(output_path)),
        "sorted": sort_by_distance_to is not None,
    }


def decode_splat_file(path: str) -> dict[str, np.ndarray]:
    """Decode a `.splat` file back into per-Gaussian arrays.

    Inverse of `encode_splat_file`. Used by tests to round-trip the binary
    representation; not used at runtime in the worker. The opacity returned
    is REAL (not logit); rotation is the (w, x, y, z) float quaternion.
    """
    record_dtype = np.dtype(
        {
            "names": ["pos", "scale", "rgba", "rot"],
            "formats": ["3<f4", "3<f4", "4u1", "4u1"],
            "offsets": [0, 12, 24, 28],
            "itemsize": SPLAT_BYTES_PER_GAUSSIAN,
        }
    )
    raw = np.fromfile(path, dtype=record_dtype)
    n = int(raw.shape[0])

    rot_f = (raw["rot"].astype(np.float32) - 128.0) / 128.0
    return {
        "means": raw["pos"].copy(),
        "scales": raw["scale"].copy(),
        "rgb": raw["rgba"][:, :3].copy(),
        "opacities": raw["rgba"][:, 3].astype(np.float32) / 255.0,
        "quats_wxyz": rot_f,
        "n": n,
    }
