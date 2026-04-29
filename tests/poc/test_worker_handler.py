from pathlib import Path

import poc.worker.handler as handler


class FakeR2:
    def __init__(self):
        self.downloads = []
        self.uploads = []

    def download_file(self, key, path):
        self.downloads.append((key, str(path)))
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_bytes(b"video")

    def upload_file(self, path, key, content_type=None):
        self.uploads.append((str(path), key, content_type))


def test_handler_downloads_runs_and_uploads(monkeypatch, tmp_path):
    fake_r2 = FakeR2()
    monkeypatch.setattr(handler, "build_r2_client", lambda: fake_r2)
    monkeypatch.setattr(handler.tempfile, "mkdtemp", lambda prefix: str(tmp_path / "work"))

    def fake_run_reconstruction(video_path, output_dir, options):
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        scene = output_dir / "scene.glb"
        metadata = output_dir / "metadata.json"
        scene.write_bytes(b"glb")
        metadata.write_text("{}", encoding="utf-8")
        return {"scene_path": str(scene), "metadata_path": str(metadata), "metadata": {"ok": True}}

    monkeypatch.setattr(handler, "run_reconstruction", fake_run_reconstruction)

    result = handler.handle_job(
        {
            "input": {
                "run_id": "abc",
                "input_video_key": "inputs/abc/video.mp4",
                "output_prefix": "runs/abc/",
                "options": {"fps": 10},
            }
        }
    )

    assert result["run_id"] == "abc"
    assert result["scene_key"] == "runs/abc/scene.glb"
    assert fake_r2.downloads[0][0] == "inputs/abc/video.mp4"
    assert ("scene.glb", "metadata.json") == (
        Path(fake_r2.uploads[0][0]).name,
        Path(fake_r2.uploads[1][0]).name,
    )
