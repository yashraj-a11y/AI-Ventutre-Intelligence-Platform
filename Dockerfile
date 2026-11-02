# Day 18 · Phase B — AI Venture Intelligence Platform image.
#
# ONE lean container: the Day-14 FastAPI app + all 13 layers (FAISS is IN-PROCESS, not a service).
# Multi-stage: a builder compiles the (heavy) deps into a venv; a slim runtime copies only that venv +
# the source. The trained artifacts (models, 407 MB FAISS index, embeddings, data) are NOT baked in —
# they are MOUNTED at runtime (see docker-compose.yml). The SBERT model + NLTK corpora ARE baked, so the
# first request pays no cold download.
#
# NOTE: `docker build` is the user's / CI's step — this file was written + statically validated (its
# build commands were run in the project venv), but the image was not built in the authoring environment.

# ----------------------------------------------------------------------------- builder
FROM python:3.14-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app

# CPU-ONLY torch FIRST (avoids the multi-GB CUDA wheel we never use), then the rest of requirements.
# torch is already pinned in requirements.txt; installing it from the CPU index up front means the
# subsequent `-r requirements.txt` sees it satisfied and doesn't pull the default (CUDA) build.
RUN pip install --upgrade pip \
    && pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

# ----------------------------------------------------------------------------- runtime
FROM python:3.14-slim AS runtime

# OMP_NUM_THREADS=1 is mandatory (faiss + torch coexistence); the app also sets it, this is defence in
# depth + covers any subprocess. Caches point at baked, world-readable image paths.
ENV OMP_NUM_THREADS=1 \
    TOKENIZERS_PARALLELISM=false \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PATH="/opt/venv/bin:$PATH" \
    HF_HOME=/opt/hf-cache \
    NLTK_DATA=/opt/nltk_data

WORKDIR /app

COPY --from=builder /opt/venv /opt/venv
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Bake the NLTK corpora + the SBERT model into the image (as root, into world-readable cache dirs) so a
# running container never downloads at first request.
RUN mkdir -p /opt/hf-cache /opt/nltk_data \
    && python scripts/download_nltk_data.py \
    && python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2', device='cpu')" \
    && chmod -R a+rX /opt/hf-cache /opt/nltk_data

# Run as a non-root user (least privilege). The mounted artifacts/ + data/ are read-only to the app.
RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/logs \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Single worker: each worker loads the full ~1 GB warm stack; scale by adding CONTAINERS, not workers.
# Artifacts (artifacts/, data/) are provided by a volume mount at runtime — NOT copied into this image.
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
