from pathlib import Path

import pytest

from poc.worker import mesh


def test_create_clay_mesh_can_be_disabled(tmp_path):
    info = mesh.create_clay_mesh_from_prepared(
        prepared={},
        output_dir=tmp_path,
        options={"mesh_enabled": False},
    )

    assert info == {"enabled": False, "reason": "disabled"}


def test_create_clay_mesh_reports_missing_required_tensors(tmp_path):
    info = mesh.create_clay_mesh_from_prepared(
        prepared={},
        output_dir=tmp_path,
        options={"mesh_enabled": True},
    )

    assert info["enabled"] is True
    assert info["status"] == "failed"
    assert "missing prediction tensors" in info["error"]


def test_create_clay_mesh_surfaces_missing_open3d(monkeypatch, tmp_path):
    prepared = {
        "depth": pytest.importorskip("numpy").ones((1, 2, 2), dtype="float32"),
        "extrinsic": pytest.importorskip("numpy").eye(4, dtype="float32")[:3, :4][None],
        "intrinsic": pytest.importorskip("numpy").eye(3, dtype="float32")[None],
        "images": pytest.importorskip("numpy").ones((1, 3, 2, 2), dtype="float32"),
    }
    monkeypatch.setattr(
        mesh,
        "_import_open3d",
        lambda: (_ for _ in ()).throw(ImportError("open3d unavailable")),
    )

    info = mesh.create_clay_mesh_from_prepared(
        prepared=prepared,
        output_dir=tmp_path,
        options={"mesh_enabled": True},
    )

    assert info["enabled"] is True
    assert info["status"] == "failed"
    assert info["error"] == "ImportError: open3d unavailable"
    assert not Path(tmp_path / "mesh.glb").exists()


def test_create_clay_mesh_accepts_depth_with_trailing_channel(monkeypatch, tmp_path):
    np = pytest.importorskip("numpy")
    prepared = {
        "depth": np.ones((1, 2, 2, 1), dtype="float32"),
        "extrinsic": np.eye(4, dtype="float32")[:3, :4][None],
        "intrinsic": np.eye(3, dtype="float32")[None],
        "images": np.ones((1, 3, 2, 2), dtype="float32"),
    }
    monkeypatch.setattr(
        mesh,
        "_import_open3d",
        lambda: (_ for _ in ()).throw(ImportError("open3d unavailable")),
    )

    info = mesh.create_clay_mesh_from_prepared(
        prepared=prepared,
        output_dir=tmp_path,
        options={"mesh_enabled": True},
    )

    assert info["error"] == "ImportError: open3d unavailable"


def test_depth_to_shw_squeezes_singleton_channel():
    np = pytest.importorskip("numpy")

    depth = mesh._depth_to_shw(np.ones((2, 3, 4, 1), dtype="float32"))

    assert depth.shape == (2, 3, 4)


def test_images_to_hwc_uint8_returns_c_contiguous_buffer():
    np = pytest.importorskip("numpy")

    images = np.ones((2, 3, 4, 5), dtype="float32")
    converted = mesh._images_to_hwc_uint8(images)

    assert converted.shape == (2, 4, 5, 3)
    assert converted.flags["C_CONTIGUOUS"]


def test_color_vertices_from_cloud_returns_nn_colors():
    np = pytest.importorskip("numpy")

    # 200 red points clustered near origin + 200 green points clustered near (10,10,10).
    rng = np.random.default_rng(0)
    red_pts = rng.normal(loc=[0, 0, 0], scale=0.01, size=(200, 3))
    green_pts = rng.normal(loc=[10, 10, 10], scale=0.01, size=(200, 3))
    cloud_points = np.vstack([red_pts, green_pts]).astype("float32")
    cloud_colors = np.vstack(
        [np.tile([255, 0, 0], (200, 1)), np.tile([0, 255, 0], (200, 1))]
    ).astype("uint8")
    mesh_vertices = np.array([[0.1, 0, 0], [9, 9, 9]], dtype="float32")

    result = mesh._color_vertices_from_cloud(
        mesh_vertices=mesh_vertices,
        cloud_points=cloud_points,
        cloud_colors=cloud_colors,
    )

    assert result["vertex_colors"].shape == (2, 4)
    assert result["vertex_colors"].dtype == np.uint8
    assert tuple(result["vertex_colors"][0, :3]) == (255, 0, 0)
    assert tuple(result["vertex_colors"][1, :3]) == (0, 255, 0)
    assert result["stats"]["cloud_points"] == 400


def test_color_vertices_from_cloud_falls_back_on_empty_cloud():
    np = pytest.importorskip("numpy")

    result = mesh._color_vertices_from_cloud(
        mesh_vertices=np.zeros((10, 3), dtype="float32"),
        cloud_points=np.zeros((0, 3), dtype="float32"),
        cloud_colors=np.zeros((0, 3), dtype="uint8"),
    )

    assert result["vertex_colors"].shape == (10, 4)
    assert "fallback_reason" in result["stats"]
