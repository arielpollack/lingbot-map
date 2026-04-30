"""Tests for mesh-aware Gaussian pruning.

Uses a single triangle in the XY-plane as the "mesh" and checks that
Gaussians at known offsets above the plane are kept or pruned correctly.
"""

import numpy as np
import pytest

trimesh = pytest.importorskip("trimesh")

from poc.worker.splat_prune import prune_gaussians_with_mesh


def _make_unit_triangle() -> "trimesh.Trimesh":
    """Single triangle in the XY-plane (z=0) covering [0,1]×[0,1]."""
    verts = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    return trimesh.Trimesh(vertices=verts, faces=faces, process=False)


def _make_gaussians(positions: np.ndarray) -> dict[str, np.ndarray]:
    n = positions.shape[0]
    return {
        "means": positions.astype(np.float32),
        "quats_wxyz": np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (n, 1)),
        "scales": np.full((n, 3), 0.05, dtype=np.float32),
        "opacities": np.full(n, 0.5, dtype=np.float32),
        "rgb": np.full((n, 3), 128, dtype=np.uint8),
    }


def test_prunes_gaussians_outside_threshold():
    mesh = _make_unit_triangle()
    # Three points: on the surface, just under threshold, far above.
    positions = np.array(
        [
            [0.25, 0.25, 0.0],   # exactly on triangle  → kept
            [0.25, 0.25, 0.05],  # 5 cm above           → kept (threshold=0.12)
            [0.25, 0.25, 0.50],  # 50 cm above          → pruned
        ],
        dtype=np.float32,
    )
    g = _make_gaussians(positions)
    pruned, stats = prune_gaussians_with_mesh(**g, mesh=mesh, threshold=0.12)

    assert stats["input"] == 3
    assert stats["kept"] == 2
    assert stats["pruned"] == 1
    assert pruned["means"].shape == (2, 3)
    np.testing.assert_array_equal(
        pruned["means"][:, 2],
        np.array([0.0, 0.05], dtype=np.float32),
    )


def test_keeps_all_when_threshold_huge():
    mesh = _make_unit_triangle()
    positions = np.array([[0.5, 0.5, 10.0]], dtype=np.float32)
    g = _make_gaussians(positions)
    pruned, stats = prune_gaussians_with_mesh(**g, mesh=mesh, threshold=100.0)
    assert stats["kept"] == 1
    assert stats["pruned"] == 0


def test_prunes_all_when_threshold_tiny():
    mesh = _make_unit_triangle()
    positions = np.array(
        [[0.25, 0.25, 0.05], [0.5, 0.5, 0.10]], dtype=np.float32
    )
    g = _make_gaussians(positions)
    pruned, stats = prune_gaussians_with_mesh(
        **g, mesh=mesh, threshold=0.001
    )
    assert stats["kept"] == 0
    assert stats["pruned"] == 2
    assert pruned["means"].shape == (0, 3)


def test_handles_off_triangle_distance_uses_3d_metric():
    """Verify the distance metric is full 3D (not just XY or z-axis)."""
    mesh = _make_unit_triangle()
    # Point well off both axes: nearest mesh vertex is (1, 0, 0) or (0, 1, 0)
    # at distance sqrt((2-1)^2 + 2^2 + 0^2) = sqrt(5) ≈ 2.236. Surface
    # samples between vertices should be slightly closer but still > 2.
    positions = np.array([[2.0, 2.0, 0.0]], dtype=np.float32)
    g = _make_gaussians(positions)
    _, stats = prune_gaussians_with_mesh(**g, mesh=mesh, threshold=3.0)
    assert stats["kept"] == 1
    assert 1.5 < stats["p50_dist"] < 2.4

    _, stats_strict = prune_gaussians_with_mesh(**g, mesh=mesh, threshold=1.0)
    assert stats_strict["kept"] == 0


def test_loads_mesh_from_path(tmp_path):
    mesh = _make_unit_triangle()
    path = tmp_path / "tri.obj"
    mesh.export(str(path))

    positions = np.array([[0.25, 0.25, 0.05]], dtype=np.float32)
    g = _make_gaussians(positions)
    pruned, stats = prune_gaussians_with_mesh(
        **g, mesh=str(path), threshold=0.12
    )
    assert stats["kept"] == 1
    assert pruned["means"].shape == (1, 3)


def test_returns_distance_percentiles():
    mesh = _make_unit_triangle()
    rng = np.random.default_rng(123)
    # Mix of points clearly inside the threshold and clearly outside.
    near = rng.uniform(-0.05, 0.05, size=(100, 3)).astype(np.float32)
    near[:, :2] = rng.uniform(0.0, 1.0, size=(100, 2)).astype(np.float32)  # inside triangle XY
    far = rng.uniform(2.0, 3.0, size=(100, 3)).astype(np.float32)
    g = _make_gaussians(np.vstack([near, far]))

    _, stats = prune_gaussians_with_mesh(**g, mesh=mesh, threshold=0.5)
    assert stats["input"] == 200
    assert 0.0 < stats["p50_dist"] < stats["p95_dist"]
    assert stats["mesh_samples"] > 0


def test_empty_input_returns_empty():
    mesh = _make_unit_triangle()
    g = {
        "means": np.zeros((0, 3), dtype=np.float32),
        "quats_wxyz": np.zeros((0, 4), dtype=np.float32),
        "scales": np.zeros((0, 3), dtype=np.float32),
        "opacities": np.zeros((0,), dtype=np.float32),
        "rgb": np.zeros((0, 3), dtype=np.uint8),
    }
    pruned, stats = prune_gaussians_with_mesh(**g, mesh=mesh, threshold=0.1)
    assert stats["input"] == 0
    assert stats["kept"] == 0
    assert stats["pruned"] == 0
    assert pruned["means"].shape == (0, 3)


def test_inconsistent_shapes_raises():
    mesh = _make_unit_triangle()
    g = _make_gaussians(np.zeros((5, 3), dtype=np.float32))
    g["rgb"] = g["rgb"][:3]  # wrong length

    with pytest.raises(ValueError, match="Inconsistent shapes"):
        prune_gaussians_with_mesh(**g, mesh=mesh, threshold=0.1)


def test_empty_mesh_raises():
    mesh = trimesh.Trimesh(
        vertices=np.zeros((0, 3)), faces=np.zeros((0, 3), dtype=np.int64)
    )
    g = _make_gaussians(np.zeros((1, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="no faces"):
        prune_gaussians_with_mesh(**g, mesh=mesh, threshold=0.1)
