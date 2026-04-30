"""Deploy the FastAPI app as a Modal asgi_app — gives us a public HTTPS
URL that the phone (Flutter app) can hit from anywhere.

Deploy:
    modal deploy poc/app/modal_web.py

URL pattern:
    https://arielpollack--lingbot-map-web-fastapi.modal.run

Persistent run records live in a Modal Dict (no Volume / file I/O needed).
The handler uses ModalClient to dispatch to the GPU worker app exactly the
same way the local server does.
"""

from __future__ import annotations

import os
from pathlib import Path

import modal

APP_NAME = "lingbot-map-web"
RUNS_DICT_NAME = "lingbot-map-runs"

# Same secret as the worker — has R2 + HF + cloudflare credentials.
worker_secret = modal.Secret.from_name("huggingface-secret")

# Light image — no torch, no Open3D. Just FastAPI + boto3 + modal.
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "modal",
        "fastapi",
        "uvicorn",
        "python-multipart",
        "boto3",
    )
    .env({"PYTHONPATH": "/workspace/lingbot-map"})
    .workdir("/workspace/lingbot-map")
)
if modal.is_local():
    _REPO_ROOT = Path(__file__).resolve().parents[2]
    image = image.add_local_dir(
        str(_REPO_ROOT / "poc"),
        remote_path="/workspace/lingbot-map/poc",
        ignore=["__pycache__", "*.pyc", ".data", ".env"],
    )

app = modal.App(APP_NAME, image=image)
runs_dict = modal.Dict.from_name(RUNS_DICT_NAME, create_if_missing=True)


def _apply_secret_aliases() -> None:
    """Map huggingface-secret env var names to what config.py expects."""
    aliases = {
        "R2_BUCKET": "R2_BUCKET_NAME",
        "R2_ACCOUNT_ID": "CLOUDFLARE_ACCOUNT_ID",
    }
    for ours, theirs in aliases.items():
        if not os.getenv(ours) and os.getenv(theirs):
            os.environ[ours] = os.environ[theirs]


@app.function(
    image=image,
    secrets=[worker_secret],
    min_containers=0,
    timeout=900,  # large uploads take time
)
@modal.concurrent(max_inputs=10)
@modal.asgi_app()
def fastapi_app():
    """Serve the existing poc.app.main FastAPI application."""
    _apply_secret_aliases()

    # Swap the file-based RunStore for the Modal Dict-backed one BEFORE
    # importing main (which builds module-level singletons).
    from poc.app import runs as runs_module
    from poc.app.runs import RunStoreDict

    _backed_store = RunStoreDict(runs_dict)

    # Monkeypatch so the import-time `RunStore(Path(config.data_dir) / "runs.json")`
    # in main.py returns our dict-backed store instead of touching disk.
    class _ProxyRunStore:
        def __new__(cls, *args, **kwargs):
            return _backed_store

    runs_module.RunStore = _ProxyRunStore  # type: ignore[assignment]

    from poc.app.main import app as fastapi_app  # build after the swap
    return fastapi_app
