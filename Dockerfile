# Phase 4, Part 4 — containerizes the FastAPI backend (Part 3) so it can run
# alongside Qdrant under docker-compose instead of a manually-activated venv.
FROM python:3.11-slim

WORKDIR /app

# libgl1 is commonly needed by OpenCV-based Grad-CAM overlay code even in a
# "headless" container; build-essential covers any package in requirements.txt
# that needs to compile from source. Trim this list once you know your exact
# dependency set doesn't need it.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/
COPY models/ ./models/
# Adjust these two COPY lines to your actual repo layout if checkpoints,
# configs, or data live somewhere else. You do NOT need to copy your raw
# Kaggle dataset or RAG source documents in here — Qdrant's collection is
# already indexed and persisted from Phase 3; the container only needs the
# code, the trained checkpoint, and whatever config files `src/agent/config.py`
# expects to find on disk.

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]