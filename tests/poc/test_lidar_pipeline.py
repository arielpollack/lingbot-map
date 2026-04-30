"""Unit tests for the Tier 1 (LiDAR) bundle reader.

Synthesizes a tiny bundle on disk that mimics what the iOS app produces,
then asserts `_bundle_to_prepared` returns the right shapes and applies
the GL → OpenCV camera frame conversion."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from poc.worker.lidar_pipeline import _bundle_to_prepared


def _write_bundle(
    tmp_path: Path,
    *,
    frame_count: int = 3,
    image_resolution: tuple[int, int] = (8, 6),
    depth_resolution: tuple[int, int] = (4, 3),
) -> tuple[Path, dict]:
    bundle_dir = tmp_path / "bundle"
    (bundle_dir / "frames").mkdir(parents=True)
    (bundle_dir / "poses").mkdir(parents=True)
    (bundle_dir / "depth").mkdir(parents=True)

    image_w, image_h = image_resolution
    depth_w, depth_h = depth_resolution

    K_image = np.array(
        [[100.0, 0.0, image_w / 2.0], [0.0, 100.0, image_h / 2.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )
    # Identity ARKit pose (camera at origin, looking down -Z in GL convention).
    T_c2w = np.eye(4, dtype=np.float32)

    for i in range(frame_count):
        idx = f"{i:06d}"
        rgb = np.full((image_h, image_w, 3), fill_value=i * 10, dtype=np.uint8)
        cv2.imwrite(str(bundle_dir / f"frames/{idx}.jpg"), rgb)

        pose = {
            "intrinsics": K_image.tolist(),
            "transform_c2w": T_c2w.tolist(),
            "image_resolution": [image_w, image_h],
            "depth_resolution": [depth_w, depth_h],
        }
        (bundle_dir / f"poses/{idx}.json").write_text(json.dumps(pose))

        depth = np.full((depth_h, depth_w), fill_value=1.0 + i, dtype=np.float32)
        depth.tofile(bundle_dir / f"depth/{idx}.bin")

        conf = np.full((depth_h, depth_w), fill_value=2, dtype=np.uint8)
        conf.tofile(bundle_dir / f"depth/{idx}.conf")

    manifest = {
        "version": 1,
        "tier": "lidar",
        "frame_count": frame_count,
        "fps": 10.0,
        "image_resolution": list(image_resolution),
        "depth_resolution": list(depth_resolution),
    }
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))
    return bundle_dir, manifest


def test_bundle_to_prepared_returns_expected_shapes(tmp_path):
    bundle_dir, manifest = _write_bundle(tmp_path, frame_count=4)
    prepared = _bundle_to_prepared(bundle_dir, manifest)

    depth_w, depth_h = manifest["depth_resolution"]
    # depth + depth_conf carry a trailing singleton dim because
    # `unproject_depth_map_to_point_map` calls .squeeze(-1) per frame.
    assert prepared["depth"].shape == (4, depth_h, depth_w, 1)
    assert prepared["depth_conf"].shape == (4, depth_h, depth_w, 1)
    assert prepared["images"].shape == (4, 3, depth_h, depth_w)
    assert prepared["intrinsic"].shape == (4, 3, 3)
    assert prepared["extrinsic"].shape == (4, 3, 4)


def test_bundle_to_prepared_rescales_intrinsics_to_depth_resolution(tmp_path):
    bundle_dir, manifest = _write_bundle(
        tmp_path,
        frame_count=1,
        image_resolution=(800, 600),
        depth_resolution=(200, 150),
    )
    prepared = _bundle_to_prepared(bundle_dir, manifest)

    K = prepared["intrinsic"][0]
    # sx = 200/800 = 0.25; sy = 150/600 = 0.25
    assert K[0, 0] == pytest.approx(100.0 * 0.25)  # fx
    assert K[1, 1] == pytest.approx(100.0 * 0.25)  # fy
    assert K[0, 2] == pytest.approx((800 / 2.0) * 0.25)  # cx
    assert K[1, 2] == pytest.approx((600 / 2.0) * 0.25)  # cy


def test_bundle_to_prepared_applies_gl_to_opencv_flip_to_extrinsics(tmp_path):
    bundle_dir, manifest = _write_bundle(tmp_path, frame_count=1)
    prepared = _bundle_to_prepared(bundle_dir, manifest)

    # Identity ARKit c2w should produce a c2w whose rotation is diag(1,-1,-1).
    R = prepared["extrinsic"][0, :3, :3]
    expected = np.diag([1.0, -1.0, -1.0]).astype(np.float32)
    np.testing.assert_allclose(R, expected, atol=1e-6)
    # Translation unchanged.
    np.testing.assert_allclose(prepared["extrinsic"][0, :3, 3], np.zeros(3))


def test_bundle_to_prepared_reads_depth_and_confidence(tmp_path):
    bundle_dir, manifest = _write_bundle(tmp_path, frame_count=2)
    prepared = _bundle_to_prepared(bundle_dir, manifest)

    # frame 0 was filled with depth=1.0, frame 1 with depth=2.0
    np.testing.assert_allclose(prepared["depth"][0, ..., 0], np.full(prepared["depth"].shape[1:3], 1.0))
    np.testing.assert_allclose(prepared["depth"][1, ..., 0], np.full(prepared["depth"].shape[1:3], 2.0))
    # All confidence pixels were set to 2 (high).
    assert (prepared["depth_conf"] == 2.0).all()


def test_bundle_to_prepared_falls_back_when_confidence_missing(tmp_path):
    bundle_dir, manifest = _write_bundle(tmp_path, frame_count=1)
    (bundle_dir / "depth/000000.conf").unlink()

    prepared = _bundle_to_prepared(bundle_dir, manifest)
    # Missing conf → defaults to 2.0 (high).
    assert (prepared["depth_conf"][0] == 2.0).all()


def test_bundle_to_prepared_rejects_zero_depth_resolution(tmp_path):
    bundle_dir, manifest = _write_bundle(tmp_path, frame_count=1)
    manifest["depth_resolution"] = [0, 0]
    (bundle_dir / "manifest.json").write_text(json.dumps(manifest))

    with pytest.raises(ValueError, match="zero depth"):
        _bundle_to_prepared(bundle_dir, manifest)
