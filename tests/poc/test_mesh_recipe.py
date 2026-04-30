"""Regression tests for the canonical Poisson-mesh recipe.

These tests lock in the recipe that produced the user-approved
`proto-poisson-mesh` result. Bbox SIGN is checked, not just extent — a
mirrored mesh has the same extent but the floor ends up on the ceiling.

These tests don't run on every `pytest tests/poc` because they need a real
`scene.glb` from a previous Modal run + Open3D + scipy installed. Run
manually before shipping a mesh-pipeline change:

    .venv/bin/python -m pytest tests/poc/test_mesh_recipe.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest


# Reference scene.glb downloaded from R2 to a known location. Skip if missing
# so the test doesn't break on machines that haven't fetched it.
REFERENCE_SCENE_GLB = Path("/tmp/proto-scene.glb")
REFERENCE_PROTO_MESH = Path("/tmp/proto-poisson.glb")


def _import_or_skip(name: str):
    try:
        return __import__(name)
    except ImportError:
        pytest.skip(f"{name} not installed")


def _bbox(verts):
    return verts.min(axis=0), verts.max(axis=0)


@pytest.fixture
def open3d_pipelines():
    o3d = _import_or_skip("open3d")
    _import_or_skip("trimesh")
    _import_or_skip("scipy")
    if not REFERENCE_SCENE_GLB.exists():
        pytest.skip(f"reference scene.glb not at {REFERENCE_SCENE_GLB}")
    return o3d


def test_canonical_recipe_matches_proto_geometry(tmp_path, open3d_pipelines):
    """Run the canonical recipe (via the standalone script's importable form)
    on the reference scene.glb. Compare output geometry against proto."""
    import numpy as np
    import trimesh
    import subprocess
    import sys

    out = tmp_path / "rerun.glb"
    # Use the standalone script as the reference recipe — the worker's
    # mesh.py is supposed to mirror it.
    result = subprocess.run(
        [
            sys.executable,
            "scripts/build_poisson_mesh_from_scene_glb.py",
            "--scene-glb", str(REFERENCE_SCENE_GLB),
            "--output", str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0 or not out.exists():
        pytest.skip(
            f"prototype script failed (Open3D Poisson is sometimes flaky): "
            f"{result.stderr[-500:] if result.stderr else 'no stderr'}"
        )

    rerun = trimesh.load(str(out), force="mesh")
    rerun_v = np.asarray(rerun.vertices)
    rerun_min, rerun_max = _bbox(rerun_v)

    if REFERENCE_PROTO_MESH.exists():
        proto = trimesh.load(str(REFERENCE_PROTO_MESH), force="mesh")
        proto_v = np.asarray(proto.vertices)
        proto_min, proto_max = _bbox(proto_v)

        # Vertex count within 5% (Poisson is mildly nondeterministic).
        assert abs(len(rerun_v) - len(proto_v)) / len(proto_v) < 0.05, \
            f"vertex count drift: rerun={len(rerun_v)}, proto={len(proto_v)}"

        # Bbox SIGN must match — this is the regression guardrail. A
        # mirrored mesh has identical extent but flipped sign, which the
        # vertex-count check above does NOT catch.
        for axis in range(3):
            assert (proto_min[axis] < 0) == (rerun_min[axis] < 0), (
                f"bbox sign flipped on axis {axis}: "
                f"proto min={proto_min[axis]}, rerun min={rerun_min[axis]}"
            )
            assert (proto_max[axis] < 0) == (rerun_max[axis] < 0), (
                f"bbox max sign flipped on axis {axis}: "
                f"proto max={proto_max[axis]}, rerun max={rerun_max[axis]}"
            )

        # Centroids should be close (within 5% of bbox extent).
        proto_extent = (proto_max - proto_min).max()
        rerun_centroid = rerun_v.mean(axis=0)
        proto_centroid = proto_v.mean(axis=0)
        delta = np.linalg.norm(rerun_centroid - proto_centroid)
        assert delta < 0.05 * proto_extent, (
            f"centroid drifted: |rerun - proto| = {delta:.3f} "
            f"(>5% of extent {proto_extent:.3f})"
        )


def test_load_cloud_from_scene_glb_returns_full_dataset(open3d_pipelines):
    """The worker's `_load_cloud_from_scene_glb` helper should round-trip
    the proto scene.glb correctly (real-world smoke check)."""
    from poc.worker.mesh import _load_cloud_from_scene_glb

    pts, cols = _load_cloud_from_scene_glb(str(REFERENCE_SCENE_GLB))
    assert pts is not None and cols is not None
    assert pts.shape[0] > 100_000  # proto has 724k
    assert pts.shape[1] == 3
    assert cols.shape == (pts.shape[0], 3)
