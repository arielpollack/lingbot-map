"""Test the platform-agnostic process_video pipeline (used by Modal entry)."""

import sys
import types
from pathlib import Path

import poc.worker.pipeline as pipeline


class FakeR2:
    def __init__(self):
        self.downloads = []
        self.uploads = []

    def download_file(self, key, path):
        self.downloads.append((key, str(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"video")

    def upload_file(self, path, key, content_type=None, content_encoding=None):
        self.uploads.append((str(path), key, content_type, content_encoding))


def test_process_video_downloads_runs_and_uploads(monkeypatch, tmp_path):
    fake_r2 = FakeR2()
    monkeypatch.setattr(pipeline, "R2Client", lambda cfg: fake_r2)
    monkeypatch.setattr(pipeline, "require_env", lambda: object())
    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "work"))

    def fake_run_reconstruction(video_path, output_dir, options):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = output_dir / "scene.glb"
        metadata = output_dir / "metadata.json"
        scene.write_bytes(b"glb")
        metadata.write_text("{}", encoding="utf-8")
        return {"scene_path": str(scene), "metadata_path": str(metadata), "metadata": {"ok": True}}

    # Stub the heavy torch-importing module before pipeline.process_video imports it.
    fake_module = types.ModuleType("poc.worker.run_reconstruction")
    fake_module.run_reconstruction = fake_run_reconstruction
    monkeypatch.setitem(sys.modules, "poc.worker.run_reconstruction", fake_module)

    result = pipeline.process_video(
        {
            "run_id": "abc",
            "input_video_key": "inputs/abc/video.mp4",
            "output_prefix": "runs/abc/",
            "options": {"fps": 10},
        }
    )

    assert result["run_id"] == "abc"
    assert result["scene_key"] == "runs/abc/scene.glb"
    assert result["splat_key"] is None  # fake reconstruction didn't produce a splat
    assert fake_r2.downloads[0][0] == "inputs/abc/video.mp4"
    upload_names = {Path(u[0]).name for u in fake_r2.uploads}
    assert upload_names == {"scene.glb", "metadata.json"}


def test_process_video_uploads_reconstruction_diagnostic_artifacts(monkeypatch, tmp_path):
    fake_r2 = FakeR2()
    monkeypatch.setattr(pipeline, "R2Client", lambda cfg: fake_r2)
    monkeypatch.setattr(pipeline, "require_env", lambda: object())
    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "work"))

    def fake_run_reconstruction(video_path, output_dir, options):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = output_dir / "scene.glb"
        metadata = output_dir / "metadata.json"
        log = output_dir / "diagnostics" / "3dgs_train.log"
        archive = output_dir / "diagnostics" / "colmap_workspace.tar.gz"
        log.parent.mkdir(parents=True)
        scene.write_bytes(b"glb")
        metadata.write_text("{}", encoding="utf-8")
        log.write_text("train log", encoding="utf-8")
        archive.write_bytes(b"tgz")
        return {
            "scene_path": str(scene),
            "metadata_path": str(metadata),
            "metadata": {"ok": True},
            "diagnostic_artifacts": [
                {
                    "path": str(log),
                    "name": "3dgs_train.log",
                    "content_type": "text/plain",
                },
                {
                    "path": str(archive),
                    "name": "colmap_workspace.tar.gz",
                    "content_type": "application/gzip",
                },
            ],
        }

    fake_module = types.ModuleType("poc.worker.run_reconstruction")
    fake_module.run_reconstruction = fake_run_reconstruction
    monkeypatch.setitem(sys.modules, "poc.worker.run_reconstruction", fake_module)

    result = pipeline.process_video(
        {
            "run_id": "abc",
            "input_video_key": "inputs/abc/video.mp4",
            "output_prefix": "runs/abc/",
            "options": {"fps": 10, "splat_diagnostics": True},
        }
    )

    assert result["diagnostic_keys"] == [
        "runs/abc/diagnostics/3dgs_train.log",
        "runs/abc/diagnostics/colmap_workspace.tar.gz",
    ]
    uploads = {(Path(path).name, key, content_type) for path, key, content_type, _ in fake_r2.uploads}
    assert ("3dgs_train.log", "runs/abc/diagnostics/3dgs_train.log", "text/plain") in uploads
    assert (
        "colmap_workspace.tar.gz",
        "runs/abc/diagnostics/colmap_workspace.tar.gz",
        "application/gzip",
    ) in uploads


def test_process_video_uploads_mesh_artifact(monkeypatch, tmp_path):
    fake_r2 = FakeR2()
    monkeypatch.setattr(pipeline, "R2Client", lambda cfg: fake_r2)
    monkeypatch.setattr(pipeline, "require_env", lambda: object())
    monkeypatch.setattr(pipeline.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "work"))

    def fake_run_reconstruction(video_path, output_dir, options):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = output_dir / "scene.glb"
        mesh = output_dir / "mesh.glb"
        metadata = output_dir / "metadata.json"
        scene.write_bytes(b"glb")
        mesh.write_bytes(b"mesh-glb")
        metadata.write_text("{}", encoding="utf-8")
        return {
            "scene_path": str(scene),
            "mesh_path": str(mesh),
            "metadata_path": str(metadata),
            "metadata": {"mesh": {"enabled": True, "status": "ok"}},
        }

    fake_module = types.ModuleType("poc.worker.run_reconstruction")
    fake_module.run_reconstruction = fake_run_reconstruction
    monkeypatch.setitem(sys.modules, "poc.worker.run_reconstruction", fake_module)

    result = pipeline.process_video(
        {
            "run_id": "abc",
            "input_video_key": "inputs/abc/video.mp4",
            "output_prefix": "runs/abc/",
            "options": {"fps": 10, "mesh_enabled": True},
        }
    )

    assert result["mesh_key"] == "runs/abc/mesh.glb"
    uploads = {(Path(path).name, key, content_type) for path, key, content_type, _ in fake_r2.uploads}
    assert ("mesh.glb", "runs/abc/mesh.glb", "model/gltf-binary") in uploads
