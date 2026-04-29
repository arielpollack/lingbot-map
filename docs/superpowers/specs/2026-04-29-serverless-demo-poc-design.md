# Serverless Demo POC Design

Date: 2026-04-29

## Goal

Build a functional proof of concept for running LingBot-Map from a local web page while executing GPU inference on Runpod Serverless and storing large inputs/outputs in Cloudflare R2.

The POC should:

- Accept arbitrary uploaded video files, including portrait videos.
- Submit reconstruction jobs to Runpod Serverless.
- Store uploaded videos and generated artifacts in Cloudflare R2.
- Serve a local web page for uploading videos, polling runs, and viewing completed runs.
- Let the user walk through completed reconstructions with FPS-style controls.

This is not a production product. The UI should be plain and functional.

## Chosen Approach

Use a static-artifact pipeline:

1. A local FastAPI app receives the browser upload.
2. The local app uploads the original video to R2.
3. The local app submits an async Runpod Serverless job using `/run`.
4. The worker downloads the video from R2, downloads/caches the Hugging Face checkpoint, runs LingBot-Map headlessly, exports a GLB scene, and uploads artifacts to R2.
5. The local app polls Runpod `/status` and stores minimal run state locally.
6. The local viewer loads the completed GLB and presents it with Three.js pointer-lock controls.

This avoids live remote viewers, WebRTC/WebSocket routing, and product-grade account features.

## External API Notes

- Runpod queue endpoints support async jobs with `/run` and status polling via `/status`. Async results are retained briefly by Runpod, so durable output must live in R2.
- Runpod worker images are packaged with a Dockerfile, a handler file, and requirements.
- R2 is S3-compatible, so both the local app and worker can use `boto3`.
- Browser direct access to R2 results requires either a public/custom-domain bucket path or generated GET URLs plus CORS. For the POC, the local app can proxy artifact reads if CORS slows development.

References:

- https://docs.runpod.io/serverless/endpoints/send-requests
- https://docs.runpod.io/serverless/workers/create-dockerfile
- https://developers.cloudflare.com/r2/api/s3/presigned-urls/
- https://developers.cloudflare.com/r2/buckets/cors/

## Repo Additions

Add a `poc/` directory:

```text
poc/
  app/
    main.py
    runpod_client.py
    r2.py
    runs.py
    static/
      index.html
      viewer.html
      app.js
      viewer.js
      styles.css
  worker/
    Dockerfile
    requirements.txt
    handler.py
    run_reconstruction.py
  README.md
  .env.example
```

Keep core LingBot-Map model code mostly unchanged. Add reusable helper functions only if needed to run inference without opening the existing Viser viewer.

## Local App

The local app is the control plane and static web server.

Responsibilities:

- Read config from environment:
  - `RUNPOD_API_KEY`
  - `RUNPOD_ENDPOINT_ID`
  - `R2_ACCOUNT_ID`
  - `R2_ACCESS_KEY_ID`
  - `R2_SECRET_ACCESS_KEY`
  - `R2_BUCKET`
  - `R2_PUBLIC_BASE_URL` optional
- Serve `GET /`.
- Accept `POST /api/runs` with multipart video upload and simple options.
- Upload input to `inputs/{run_id}/{filename}` in R2.
- Submit Runpod async job with input keys and options.
- Poll job status via `GET /api/runs/{run_id}`.
- List known runs via `GET /api/runs`.
- Proxy completed artifacts via `GET /api/runs/{run_id}/artifact/{name}` when no public R2 URL is configured.

Run state can be a JSON file under `poc/.data/runs.json`. SQLite is unnecessary for the POC.

## Runpod Worker

The worker is a single Runpod Serverless handler.

Input shape:

```json
{
  "run_id": "uuid",
  "input_video_key": "inputs/uuid/video.mp4",
  "output_prefix": "runs/uuid/",
  "options": {
    "fps": 10,
    "first_k": null,
    "stride": 1,
    "mode": "streaming",
    "mask_sky": false,
    "conf_threshold": 1.5,
    "downsample_factor": 10,
    "point_size": 0.00001
  }
}
```

Worker steps:

1. Download input video from R2 to a temporary run directory.
2. Resolve the checkpoint from Hugging Face. Default to repo `robbyant/lingbot-map` and checkpoint `lingbot-map-long`.
3. Run inference headlessly.
4. Export `scene.glb`.
5. Write `metadata.json` with run options, source dimensions, frame count, output keys, timing, and warnings.
6. Upload artifacts to R2 under `runs/{run_id}/`.
7. Return output keys in the Runpod job result.

The worker should not start Viser and should not require a web port.

## Headless Reconstruction Helper

Refactor or wrap `demo.py` so the worker can run the existing pipeline without opening `PointCloudViewer.run()`.

Needed behavior:

- Use existing `load_images`, `load_model`, `postprocess`, and `prepare_for_visualization` logic where possible.
- Accept video input.
- Preserve portrait inputs through extraction. Do not reject video based on aspect ratio.
- Export a GLB from predictions without relying on interactive GUI state.
- Keep temporary frames and intermediate files inside the run work directory.

The simplest implementation is a new helper module in `poc/worker/run_reconstruction.py` that imports demo functions and calls a new small export utility. If that gets awkward, add one focused function to the package such as `lingbot_map.vis.export_predictions_to_glb`.

## Portrait Video Handling

OpenCV frame extraction should keep the decoded frame dimensions as-is. The LingBot preprocessing path already crops/resizes to the model input size, so portrait video support means:

- The upload path accepts portrait files.
- Extraction does not rotate, letterbox, or reject portrait frames.
- Metadata records original video width, height, fps, and frame count.
- The viewer loads the resulting GLB like any other scene.

If mobile videos contain rotation metadata that OpenCV ignores on a test clip, use FFmpeg extraction in the worker as a follow-up. Do not block the first POC on this unless the demo video requires it.

## Viewer

Use a static Three.js viewer.

Responsibilities:

- Load `scene.glb`.
- Show a plain status/error message.
- Use pointer lock for mouse-look.
- Support:
  - `W/A/S/D` horizontal movement
  - mouse look
  - `Space` up
  - `Shift` down
  - `Esc` unlock pointer
  - optional speed slider or numeric input
- Start near the scene center or first camera position if metadata contains one.

Collision and physics are out of scope. This is free-fly FPS navigation through a point cloud/mesh artifact.

## Error Handling

Local app:

- Validate required environment variables on startup.
- Reject empty uploads.
- Store failed upload/submission errors in run state.
- Return JSON errors for API calls.

Worker:

- Return structured failure messages.
- Upload `metadata.json` even when possible failures occur after input download.
- Include logs in Runpod logs, not in R2, unless easy.

Viewer:

- Show an error if GLB loading fails.
- Keep controls functional without fancy UI.

## Testing And Verification

Local-only checks:

- Start the FastAPI app.
- Upload a small landscape video or existing test clip.
- Upload a small portrait video.
- Confirm R2 receives input keys.
- Mock or dry-run Runpod submission if endpoint is not configured.

Worker checks:

- Run the handler locally against an example folder or short video if CUDA/checkpoint are available.
- Confirm `scene.glb` and `metadata.json` are generated.
- Confirm artifacts upload to R2.

Viewer checks:

- Load a local or proxied GLB.
- Confirm pointer lock starts on click.
- Confirm WASD/mouse/Space/Shift change the camera.

End-to-end demo:

- Provide keys in `.env`.
- Start local app.
- Upload video.
- Watch status move from uploaded to submitted to completed.
- Open viewer and walk through the generated scene.

## Non-Goals

- Authentication.
- Multi-user run isolation.
- Production database.
- Fancy design.
- Live streaming progress visuals.
- Live remote Viser.
- Automatic Runpod endpoint creation.
- Robust resumable multipart upload.
- Full mobile/touch viewer controls.

## Open Implementation Defaults

- Use `uvicorn` for the local server.
- Use `boto3` for R2.
- Use `requests` or `httpx` for Runpod calls.
- Use a CUDA PyTorch base image for the worker.
- Install `lingbot-map[vis]`, `runpod`, `boto3`, and Hugging Face dependencies in the worker image.
- Prefer `lingbot-map-long` from Hugging Face for checkpoint quality unless download time makes iteration painful.
