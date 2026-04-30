# GSplat Diagnostics Status

**Locked baseline:** LingBot point-cloud / GLB export remains the baseline. The diagnostic work is scoped to the 3DGS phase and artifact upload path.

## Implemented

- `poc/worker/gsplat.py`
  - Configurable COLMAP init confidence threshold, init point count, and random seed.
  - 3DGS train log capture.
  - Train-view render helper using Graphdeco `render.py`.
  - Train-view MAE/MSE/PSNR metrics.

- `poc/worker/run_reconstruction.py`
  - Optional `splat_diagnostics` artifacts:
    - `colmap_workspace.tar.gz`
    - `3dgs_train.log`
    - `3dgs_render_train.log`
    - `train_view_metrics.json`
    - `train_view_renders.tar.gz`
  - Optional `splat_eval_train_views` metric pass.

- `poc/worker/pipeline.py`
  - Uploads diagnostic artifacts to `runs/{run_id}/diagnostics/`.

## Validated

- Local tests: `.venv/bin/python -m pytest -q`
  - Result: `11 passed`

- Modal probe after deploy:
  - Pipeline import: ok
  - R2 connect: ok
  - 3DGS `train.py`: ok

- Phase 1 diagnostic run:
  - Run id: `diag-c7e4e1ca9f0e`
  - Uploaded: COLMAP workspace and train log.

- Phase 2 metric run:
  - Run id: `diag2-7c41ec8ca5e0`
  - 60 frames, 100 iterations.
  - Metrics: PSNR `12.370862137111825`, MAE `0.11623999189806593`, MSE `0.057931368289958485`.
  - Uploaded all diagnostic artifacts.

- Controlled init threshold experiment:
  - Run id: `exp-conf-e83fdfabee`
  - Only changed `splat_init_conf_threshold` to `3.144076347351074`.
  - Metrics: PSNR `12.325831059374412`, MAE `0.11802152083948586`, MSE `0.05853517135810626`.
  - Result: worse than baseline; do not adopt.

- Controlled init point-count experiment:
  - Run id: `exp-pts-f0856fdab96`
  - Only changed `splat_init_points` to `200000`.
  - Metrics: PSNR `12.212435648214859`, MAE `0.11776941683255562`, MSE `0.06008366765068747`.
  - Result: worse than baseline and larger artifact; do not adopt.

## Pending

- 7k iteration baseline:
  - Run id: `base7k-7f8fca9dd12`
  - Modal call id: `fc-01KQD9KD8WFBVZMD1J2NYJ752W`
  - Polling was interrupted. Re-query with the Monday cert bundle loaded:

```bash
set -a && source poc/.env && set +a
.venv/bin/python - <<'PY'
import json, modal

call_id = "fc-01KQD9KD8WFBVZMD1J2NYJ752W"
call = modal.FunctionCall.from_id(call_id)
try:
    result = call.get(timeout=0)
except TimeoutError:
    print(json.dumps({"status": "IN_PROGRESS", "call_id": call_id}, indent=2))
else:
    print(json.dumps({"status": "COMPLETED", "call_id": call_id, "output": result}, indent=2))
PY
```

Expected env from `poc/.env`:

- `GRPC_DEFAULT_SSL_ROOTS_FILE_PATH=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem`
- `SSL_CERT_FILE=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem`
- `REQUESTS_CA_BUNDLE=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem`
- `CURL_CA_BUNDLE=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem`
- `AWS_CA_BUNDLE=/Users/ariel/develop/lingbot-map/poc/.data/ca-bundle.pem`

## Next Gate

If the 7k train-view PSNR is still low, the problem is not just early iterations. Next phase should test pose/geometry quality directly by comparing cross-view consistency or running a real SfM/VGGSfM/COLMAP baseline. If 7k train-view PSNR is decent but free-view output is bad, focus on viewer conventions and splat orientation/scale.
