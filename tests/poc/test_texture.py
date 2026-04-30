"""Synthetic test for the texture bake. Uses a simple 2-triangle quad mesh
viewed from a single camera, verifies the atlas pixel-fill semantics."""

from __future__ import annotations

from pathlib import Path

import pytest


def _import_or_skip(name: str):
    try:
        return __import__(name)
    except ImportError:
        pytest.skip(f"{name} not installed")


def test_bake_texture_atlas_produces_glb(tmp_path):
    np = _import_or_skip("numpy")
    _import_or_skip("xatlas")
    _import_or_skip("open3d")
    _import_or_skip("trimesh")

    from poc.worker.texture import bake_texture_atlas

    # 2-triangle quad in z=0 plane, span [-1,1] x [-1,1].
    vertices = np.array(
        [
            [-1.0, -1.0, 0.0],
            [1.0, -1.0, 0.0],
            [1.0, 1.0, 0.0],
            [-1.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    # Counter-clockwise winding when viewed from +Z so the face normal
    # points in +Z direction (toward the camera at z=+3).
    faces = np.array([[0, 2, 1], [0, 3, 2]], dtype=np.int32)

    # Single 32x32 frame, all red.
    H = W = 32
    image = np.zeros((1, H, W, 3), dtype=np.uint8)
    image[0, :, :] = (240, 50, 50)  # bright red
    intrinsics = np.array(
        [[[20.0, 0.0, W / 2], [0.0, 20.0, H / 2], [0.0, 0.0, 1.0]]],
        dtype=np.float32,
    )
    # Camera at (0,0,3) looking at -Z (so the quad at z=0 is in front).
    extrinsics_w2c = np.array(
        [
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 3.0],
                [0.0, 0.0, 0.0, 1.0],
            ]
        ],
        dtype=np.float32,
    )

    output_path = tmp_path / "out.glb"
    info = bake_texture_atlas(
        vertices=vertices,
        faces=faces,
        images_hwc_uint8=image,
        intrinsics=intrinsics,
        extrinsics_w2c=extrinsics_w2c,
        output_path=output_path,
        atlas_resolution=128,
        top_k_frames=1,
        visibility_check=False,
    )

    assert info["status"] == "ok"
    assert output_path.exists()
    assert info["pixels_filled"] > 0
    assert info["fill_percent"] > 0

    # Load output and verify it has a baseColorTexture
    import trimesh

    mesh = trimesh.load(str(output_path), force="mesh")
    assert mesh.vertices.shape[0] >= 4  # at least the original 4 (xatlas may split)
    assert mesh.faces.shape[0] >= 2
    assert hasattr(mesh.visual, "material")
    img = mesh.visual.material.baseColorTexture
    assert img is not None
    assert img.size == (128, 128)

    # Most pixels in filled regions should be reddish (we fed all-red).
    import numpy as np

    arr = np.array(img)
    # Top corners may be the fallback gray; check center mass via stats.
    # Take only pixels close to red — at least some should be present.
    is_reddish = (arr[..., 0] > 150) & (arr[..., 1] < 120) & (arr[..., 2] < 120)
    assert is_reddish.sum() > 100, (
        f"expected at least 100 reddish pixels in atlas, got {is_reddish.sum()}"
    )
