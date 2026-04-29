from concurrent.futures import ThreadPoolExecutor

from poc.app.runs import RunStore


def test_run_store_creates_updates_and_lists_runs(tmp_path):
    store = RunStore(tmp_path / "runs.json")

    run = store.create_run(filename="portrait.mp4", input_key="inputs/abc/portrait.mp4")
    assert run["id"]
    assert run["status"] == "uploaded"
    assert run["filename"] == "portrait.mp4"
    assert run["input_key"] == "inputs/abc/portrait.mp4"

    updated = store.update_run(run["id"], status="completed", job_id="job-1", output_key="runs/abc/scene.glb")
    assert updated["status"] == "completed"
    assert updated["job_id"] == "job-1"
    assert updated["output_key"] == "runs/abc/scene.glb"

    runs = store.list_runs()
    assert [item["id"] for item in runs] == [run["id"]]


def test_run_store_persists_between_instances(tmp_path):
    path = tmp_path / "runs.json"
    first = RunStore(path)
    run = first.create_run(filename="clip.mp4", input_key="inputs/run/clip.mp4")

    second = RunStore(path)
    assert second.get_run(run["id"])["filename"] == "clip.mp4"


def test_run_store_serializes_concurrent_creates(tmp_path):
    store = RunStore(tmp_path / "runs.json")

    def create_run(index):
        return store.create_run(filename=f"clip-{index}.mp4", input_key=f"inputs/{index}/clip.mp4")

    with ThreadPoolExecutor(max_workers=32) as executor:
        created = list(executor.map(create_run, range(32)))

    created_ids = {run["id"] for run in created}
    listed_ids = {run["id"] for run in store.list_runs()}
    assert created_ids <= listed_ids
