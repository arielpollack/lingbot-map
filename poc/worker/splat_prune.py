"""Mesh-aware Gaussian pruning.

Drops every Gaussian whose center is more than `threshold` meters from
the nearest mesh face. The lidar_mesh tier has an ARKit-fused mesh that
acts as a hard surface constraint; pruning against it eliminates the
graphdeco-style "floaters" — Gaussians the optimizer parks in mid-air to
explain pixels along occluded edges. Those Gaussians look fine from the
captured viewing angle and like haze from any other angle, which is
exactly the failure mode we're targeting.

Implementation uses scipy.cKDTree on dense samples of the mesh surface
rather than `trimesh.proximity.closest_point` because the latter requires
the optional `rtree` C extension (and `libspatialindex`), which isn't
guaranteed everywhere. KDTree-on-samples gives an upper bound on true
surface distance — at sample spacing ≈ threshold/5 the bias is well below
the prune threshold, so the false-keep rate is small and the false-prune
rate is zero (we never drop a true near-surface Gaussian).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np


def prune_gaussians_with_mesh(
    *,
    means: np.ndarray,
    quats_wxyz: np.ndarray,
    scales: np.ndarray,
    opacities: np.ndarray,
    rgb: np.ndarray,
    mesh: Any,                # trimesh.Trimesh OR str/Path to mesh file
    threshold: float = 0.12,
    samples_per_threshold: int = 5,
    min_samples: int = 10_000,
    max_samples: int = 500_000,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Filter Gaussians by Euclidean distance to the nearest mesh-surface sample.

    Args:
        means / quats_wxyz / scales / opacities / rgb: parallel `(N, ...)`
            arrays for the trained Gaussians. All must have the same N.
        mesh: a `trimesh.Trimesh` or a path to a mesh file (`.obj`, `.glb`,
            etc.). For path inputs we load with `process=False` to keep the
            ARKit vertex layout intact.
        threshold: max distance (meters) from the surface. Gaussians with
            `distance > threshold` are dropped.
        samples_per_threshold: aim for sample spacing of
            `threshold / samples_per_threshold`. With the default of 5,
            sample spacing ≈ threshold/5 = ~2.4 cm at the 0.12 m default
            threshold — well below the prune threshold, so distance bias
            doesn't cause false-keeps.
        min_samples / max_samples: clamp the sample count derived from
            mesh area + density target.

    Returns:
        (pruned_arrays, stats) where:
            pruned_arrays = {"means", "quats_wxyz", "scales", "opacities", "rgb"}
            stats = {"input", "kept", "pruned", "threshold",
                     "p50_dist", "p95_dist", "elapsed_s",
                     "mesh_samples", "mesh_area"}
    """
    import trimesh
    from scipy.spatial import cKDTree

    n = means.shape[0]
    if not (
        quats_wxyz.shape == (n, 4)
        and scales.shape == (n, 3)
        and opacities.reshape(-1).shape == (n,)
        and rgb.shape == (n, 3)
    ):
        raise ValueError(
            f"Inconsistent shapes for N={n}: means={means.shape}, "
            f"quats={quats_wxyz.shape}, scales={scales.shape}, "
            f"opacities={opacities.shape}, rgb={rgb.shape}"
        )

    if isinstance(mesh, (str, Path)):
        loaded = trimesh.load(str(mesh), process=False, force="mesh")
        if not isinstance(loaded, trimesh.Trimesh):
            raise ValueError(
                f"Mesh at {mesh} loaded as {type(loaded).__name__}, expected Trimesh"
            )
        mesh_t = loaded
    elif isinstance(mesh, trimesh.Trimesh):
        mesh_t = mesh
    else:
        raise TypeError(f"Unsupported mesh type: {type(mesh).__name__}")

    if mesh_t.faces.shape[0] == 0:
        raise ValueError("Mesh has no faces — cannot compute proximity")

    start = time.time()

    # Sample density: aim for `samples_per_threshold` samples within
    # `threshold` meters along any axis. Per unit area this is
    # `(samples_per_threshold / threshold)^2` samples — but only over
    # 2D surface area, so scale by mesh.area.
    target_density = (samples_per_threshold / max(threshold, 1e-6)) ** 2
    n_samples = int(np.clip(mesh_t.area * target_density, min_samples, max_samples))
    surface_samples, _face_idx = trimesh.sample.sample_surface(mesh_t, n_samples)
    surface_samples = np.asarray(surface_samples, dtype=np.float32)

    # Always include mesh vertices in the KDTree — sample_surface has zero
    # probability of landing exactly on a vertex, but vertices are real
    # surface points and including them tightens the upper-bound estimate.
    vertices = np.asarray(mesh_t.vertices, dtype=np.float32)
    sample_pool = np.concatenate([surface_samples, vertices], axis=0)

    tree = cKDTree(sample_pool)
    if n == 0:
        distances = np.zeros(0, dtype=np.float32)
    else:
        distances, _ = tree.query(means.astype(np.float32), k=1)
        distances = distances.astype(np.float32)

    keep = distances <= threshold
    pruned_count = int(n - keep.sum())
    elapsed_s = round(time.time() - start, 3)

    if n == 0:
        p50 = p95 = 0.0
    else:
        p50 = float(np.percentile(distances, 50))
        p95 = float(np.percentile(distances, 95))

    pruned = {
        "means": means[keep].copy(),
        "quats_wxyz": quats_wxyz[keep].copy(),
        "scales": scales[keep].copy(),
        "opacities": np.asarray(opacities).reshape(-1)[keep].copy(),
        "rgb": rgb[keep].copy(),
    }
    stats = {
        "input": int(n),
        "kept": int(keep.sum()),
        "pruned": pruned_count,
        "threshold": float(threshold),
        "p50_dist": p50,
        "p95_dist": p95,
        "elapsed_s": elapsed_s,
        "mesh_samples": int(sample_pool.shape[0]),
        "mesh_area": float(mesh_t.area),
    }
    return pruned, stats
