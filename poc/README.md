# LingBot-Map Serverless POC

## Local app

```bash
python -m pip install fastapi uvicorn python-multipart boto3 requests pytest
cp poc/.env.example poc/.env
set -a
source poc/.env
set +a
uvicorn poc.app.main:app --reload --port 7860
```

Open `http://127.0.0.1:7860`.

## Worker image

```bash
docker build -f poc/worker/Dockerfile -t lingbot-map-runpod-poc .
docker tag lingbot-map-runpod-poc YOUR_DOCKERHUB_USER/lingbot-map-runpod-poc:latest
docker push YOUR_DOCKERHUB_USER/lingbot-map-runpod-poc:latest
```

Create a Runpod Serverless endpoint with that image. Configure the same R2 environment variables on the endpoint.

## Run flow

1. Upload a video in the local page.
2. The local app uploads it to R2 and submits a Runpod async job.
3. The worker downloads the video, downloads the LingBot-Map checkpoint from Hugging Face, exports `scene.glb`, and uploads results to R2.
4. The local page polls status. When complete, click `view`.
