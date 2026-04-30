from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image

from poc.worker import gsplat


def test_export_colmap_workspace_uses_configurable_confidence_threshold_and_seed(tmp_path):
    intrinsics = np.repeat(np.eye(3, dtype=np.float32)[None], 1, axis=0)
    intrinsics[:, 0, 0] = 10
    intrinsics[:, 1, 1] = 10
    extrinsics = np.repeat(np.eye(4, dtype=np.float32)[None], 1, axis=0)
    world_points = np.array(
        [[
            [[0.0, 0.0, 1.0], [1.0, 0.0, 1.0]],
            [[0.0, 1.0, 1.0], [1.0, 1.0, 1.0]],
        ]],
        dtype=np.float32,
    )
    images = np.ones((1, 2, 2, 3), dtype=np.float32)
    confidence = np.array([[[0.1, 2.0], [3.0, 4.0]]], dtype=np.float32)

    stats = gsplat.export_colmap_workspace(
        intrinsics,
        extrinsics,
        world_points,
        images,
        confidence,
        tmp_path,
        conf_threshold=2.5,
        target_points=2,
        random_seed=123,
    )

    points_path = tmp_path / "sparse" / "0" / "points3D.txt"
    point_lines = [
        line
        for line in points_path.read_text(encoding="utf-8").splitlines()
        if line and not line.startswith("#")
    ]
    assert len(point_lines) == 2
    assert stats["valid_points_before_subsampling"] == 2
    assert stats["points_written"] == 2
    assert stats["conf_threshold"] == 2.5


def test_train_gaussian_splatting_writes_stdout_and_stderr_log(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, timeout):
        output_ply = tmp_path / "out" / "point_cloud" / "iteration_9" / "point_cloud.ply"
        output_ply.parent.mkdir(parents=True)
        output_ply.write_bytes(b"ply")
        return SimpleNamespace(returncode=0, stdout="training stdout", stderr="training stderr")

    monkeypatch.setattr(gsplat.subprocess, "run", fake_run)
    log_path = tmp_path / "train.log"

    ply_path = gsplat.train_gaussian_splatting(
        tmp_path / "colmap",
        tmp_path / "out",
        iterations=9,
        log_path=log_path,
        quiet=False,
    )

    assert ply_path == str(tmp_path / "out" / "point_cloud" / "iteration_9" / "point_cloud.ply")
    assert log_path.read_text(encoding="utf-8") == (
        "$ "
        + " ".join(
            [
                gsplat.sys.executable,
                "/opt/gaussian-splatting/train.py",
                "-s",
                str(tmp_path / "colmap"),
                "--model_path",
                str(tmp_path / "out"),
                "--iterations",
                "9",
                "--sh_degree",
                "3",
            ]
        )
        + "\n\n[stdout]\ntraining stdout\n\n[stderr]\ntraining stderr\n"
    )


def test_render_gaussian_splatting_train_views_writes_log(monkeypatch, tmp_path):
    def fake_run(cmd, capture_output, text, timeout):
        renders = tmp_path / "out" / "train" / "ours_9" / "renders"
        gt = tmp_path / "out" / "train" / "ours_9" / "gt"
        renders.mkdir(parents=True)
        gt.mkdir(parents=True)
        return SimpleNamespace(returncode=0, stdout="render stdout", stderr="render stderr")

    monkeypatch.setattr(gsplat.subprocess, "run", fake_run)
    log_path = tmp_path / "render.log"

    result = gsplat.render_gaussian_splatting_train_views(
        tmp_path / "out",
        iteration=9,
        log_path=log_path,
    )

    assert result == {
        "renders_dir": str(tmp_path / "out" / "train" / "ours_9" / "renders"),
        "gt_dir": str(tmp_path / "out" / "train" / "ours_9" / "gt"),
    }
    log_text = log_path.read_text(encoding="utf-8")
    assert "[stdout]\nrender stdout" in log_text
    assert "--skip_test" in log_text


def test_compute_render_metrics_reports_mae_mse_psnr(tmp_path):
    renders = tmp_path / "renders"
    gt = tmp_path / "gt"
    renders.mkdir()
    gt.mkdir()
    Image.fromarray(np.full((2, 2, 3), 128, dtype=np.uint8)).save(renders / "00000.png")
    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(gt / "00000.png")

    metrics = gsplat.compute_render_metrics(renders, gt)

    expected = 128 / 255
    assert metrics["frame_count"] == 1
    assert metrics["mae"] == expected
    assert metrics["mse"] == expected * expected
    assert metrics["psnr"] == 20.0 * np.log10(1.0 / expected)


def test_viewer_camera_from_colmap_uses_training_camera_convention():
    intrinsics = np.array([[[100.0, 0.0, 50.0], [0.0, 200.0, 40.0], [0.0, 0.0, 1.0]]])
    w2c = np.repeat(np.eye(4, dtype=np.float32)[None], 1, axis=0)
    points = np.array([[[[-1.0, -2.0, 3.0], [4.0, 5.0, 6.0]]]], dtype=np.float32)

    camera = gsplat.viewer_camera_from_colmap(
        w2c,
        intrinsics,
        image_size=(80, 100),
        scene_points=points,
    )

    assert camera["position"] == [0.0, 0.0, 0.0]
    assert camera["target"] == [0.0, 0.0, 1.0]
    assert camera["up"] == [0.0, -1.0, 0.0]
    assert camera["near"] == 0.01
    assert camera["far"] > 10
    assert camera["move_speed"] > 0
    assert camera["fov_degrees"] == 2.0 * np.degrees(np.arctan(80.0 / (2.0 * 200.0)))
