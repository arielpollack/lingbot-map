FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

WORKDIR /workspace/lingbot-map

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md LICENSE.txt demo.py ./
COPY lingbot_map ./lingbot_map
COPY poc ./poc

RUN pip install --no-cache-dir -e ".[vis]" \
    && pip install --no-cache-dir -r poc/worker/requirements.txt

CMD ["python", "-u", "poc/worker/handler.py"]
