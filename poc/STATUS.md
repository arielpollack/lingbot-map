# POC deployment status

Generated 2026-04-29 after autonomous Task 8–10 finish.

## Wired up

- **Local app** running at `http://127.0.0.1:7860` (uvicorn pid logged in `/tmp/poc-app.log`).
- **Cloudflare R2** — bucket `vitour`, account `c0edb6332d9f6c142c777433fff96255`. Inputs land at `inputs/{run_id}/{filename}`, outputs at `runs/{run_id}/{scene.glb,metadata.json}`. Sharing the vitour bucket is fine because key prefixes do not collide.
- **Docker image** `arielpollack/lingbot-map-runpod-poc:latest` — built from `poc/worker/Dockerfile`, pushed to Docker Hub. Digest `sha256:1cb4a5990a3f43a3c3c7eece5e29a6043ef778c97b72d991795325b55d29eef9`. 9.16 GB.
- **Runpod template** `xbjhs9ie0x` (`lingbot-map-runpod-poc`) — points at the image above with R2 + RUNPOD_* env vars set.
- **Runpod Serverless endpoint** `zs2z4gps6jqgn6` (`lingbot-map-poc`) — workersMin=0, workersMax=2, idleTimeout=5s, GPU pool: 4090 / A5000 / L4 / 3090 / A40 / A6000. Health endpoint replies (`workers.throttled=1` until first job warms one).
- **HuggingFace model** `robbyant/lingbot-map/lingbot-map-long.pt` confirmed public; the worker downloads it on first cold start (~minutes).
- **`poc/.env`** populated with the working creds. Git-ignored. Pulled from `~/develop/vitour/.env` per user note (rotation planned before going live).

## Verified locally

- `pytest tests/poc -q` → 8/8 pass.
- All plan import paths resolve (after lazy-loading `lingbot_map.vis/__init__.py` so torch-free consumers can use `glb_export`).
- App boots with dummy env: `/`, `/viewer?run=…`, `/api/runs` all return 200.
- App boots with real env: same routes serve, `R2Client` initialises against the real bucket.
- Worker container imports `handle_job` and constructs `R2Client` against the live bucket.

## Not yet verified (needs user)

- A real upload through `/` against the live Runpod endpoint. First job will pay a cold-start cost (~3–5 min image pull on Runpod, plus first-time HF model download ~minutes). Subsequent jobs reuse the warm worker until idle timeout (5 s).
- The actual GLB output viewed in the FPS viewer at `/viewer?run=<id>`.

## Try it

```bash
# the server is already running; if not:
cd ~/develop/lingbot-map
source .venv/bin/activate
set -a && source poc/.env && set +a
uvicorn poc.app.main:app --reload --port 7860
```

1. Open `http://127.0.0.1:7860`.
2. Upload an apartment-walkthrough video (mp4 / mov). Defaults: fps=10, mode=streaming, mask_sky=off.
3. The row turns from `submitted` → `running` → `completed` as the local app polls `/api/runs/{id}` every 5 s.
4. Click **view** → GLB opens in a Three.js scene. Click the canvas to pointer-lock; WASD + mouse-look + Space/Shift to fly around.

## Resources to watch

- Endpoint dashboard: <https://www.runpod.io/console/serverless/user/endpoint/zs2z4gps6jqgn6>
- Logs in real time: query `https://api.runpod.ai/v2/zs2z4gps6jqgn6/health` or open the endpoint UI.
- App log: `/tmp/poc-app.log`.

## Cleanup if abandoning

```bash
# stop the app
pkill -f 'uvicorn poc.app.main:app'

# delete Runpod endpoint and template (frees nothing material, no per-hour cost while idle)
curl -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/endpoints/zs2z4gps6jqgn6
curl -X DELETE -H "Authorization: Bearer $RUNPOD_API_KEY" https://rest.runpod.io/v1/templates/xbjhs9ie0x
```
