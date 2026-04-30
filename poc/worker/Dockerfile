# Thin app image. The base carries lingbot-map + 3DGS CUDA wheels + train.py;
# we only layer in the worker code so iteration on handler.py is seconds, not
# 15 minutes.
#
# Update the base tag below whenever you rebuild Dockerfile.base.
ARG BASE_TAG=3dgs-latest
FROM arielpollack/lingbot-map-runpod-poc-base:${BASE_TAG}

WORKDIR /workspace/lingbot-map

COPY poc ./poc

CMD ["python", "-m", "poc.worker.handler"]
