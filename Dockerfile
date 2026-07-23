# ═══════════════════════════════════════════════════════════════════════════════
# Single Dockerfile — ingestion / asr / diarisation
# Uses Python venv for reliable cross-stage package copying.
#
# NOTE ON GPU:
#   docker build has NO GPU access — model weights cannot be downloaded
#   to GPU during build. At container RUNTIME, WHISPER_DEVICE=cuda and
#   DIAR_DEVICE=cuda ensure all inference runs on the RTX 3050.
#   Model weights download on first container start and are cached in
#   the hf_cache Docker volume (never re-downloaded on restart).
# ═══════════════════════════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────────────────────────
# BASE — CPU  (ingestion service)
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base-cpu

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential gcc git \
        ffmpeg curl libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip==24.0


# ───────────────────────────────────────────────────────────────────────────────
# BASE — CUDA  (asr + diarisation services)
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS base-cuda

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-dev python3.11-venv python3-pip \
        build-essential gcc git \
        ffmpeg libsndfile1 libsndfile1-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip==24.0


# ───────────────────────────────────────────────────────────────────────────────
# INGESTION BUILDER
# ───────────────────────────────────────────────────────────────────────────────
FROM base-cpu AS ingestion-builder

RUN pip install --no-cache-dir --timeout=300 \
        celery==5.4.0 \
        kombu==5.3.4 \
        fastapi==0.111.0 \
        uvicorn[standard]==0.29.0 \
        python-multipart==0.0.9 \
        aiofiles==23.2.1 \
        redis==4.6.0

RUN pip install --no-cache-dir --timeout=300 \
        ffmpeg-python==0.2.0 \
        numpy==1.26.4 \
        scipy==1.13.0 \
        soundfile==0.12.1

RUN pip install --no-cache-dir --timeout=600 \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        torch==2.3.0+cpu \
        torchaudio==2.3.0+cpu

RUN pip install --no-cache-dir --timeout=300 \
        noisereduce==3.0.3 \
        librosa==0.10.2 \
        audioread==3.0.1


# ───────────────────────────────────────────────────────────────────────────────
# INGESTION RUNTIME  (target: ingestion)
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS ingestion

RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg curl libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ingestion-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Silero VAD model — downloads on first request, cached in hf_cache volume
WORKDIR /app
COPY ingestion_service/app.py \
     ingestion_service/extractor.py \
     ingestion_service/vad.py \
     observability.py ./

ENV STORAGE_DIR=/data/sessions
RUN mkdir -p /data/sessions

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]


# ───────────────────────────────────────────────────────────────────────────────
# ASR BUILDER
# ───────────────────────────────────────────────────────────────────────────────
FROM base-cuda AS asr-builder

RUN pip install --no-cache-dir --timeout=300 \
        celery==5.4.0 \
        kombu==5.3.4 \
        redis==4.6.0

RUN pip install --no-cache-dir --timeout=600 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        torch==2.3.0+cu121 \
        torchaudio==2.3.0+cu121

RUN pip install --no-cache-dir --timeout=300 \
        numpy==1.26.4 \
        scipy==1.13.0 \
        matplotlib==3.8.4 \
        soundfile==0.12.1

RUN pip install --no-cache-dir --timeout=300 \
        faster-whisper==1.0.3 \
        requests==2.32.3


# ───────────────────────────────────────────────────────────────────────────────
# ASR RUNTIME  (target: asr)
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS asr

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY --from=asr-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Verify the venv copied correctly — fails the build immediately if broken
RUN python -c "import faster_whisper; print('faster_whisper OK')"
RUN python -c "import torch; print('torch OK, CUDA available at runtime:', torch.cuda.is_available())"

# Whisper large-v3 weights (~3 GB) download on first container start.
# Cached in hf_cache Docker volume — never re-downloaded on restart.
# At runtime WHISPER_DEVICE=cuda → all inference runs on RTX 3050.

WORKDIR /app
COPY asr_service/app.py \
     asr_service/whisper_chunker.py \
     asr_service/word_timestamps.py \
     observability.py ./

ENV STORAGE_DIR=/data/sessions
ENV WHISPER_MODEL=medium
ENV WHISPER_DEVICE=cuda
ENV WHISPER_COMPUTE=int8_float16
ENV WHISPER_WORKERS=1
ENV REDIS_URL=redis://redis:6379/0

RUN mkdir -p /data/sessions
VOLUME ["/root/.cache/huggingface"]

CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", \
     "--loglevel=info", "--queues=asr", "--concurrency=1"]


# ───────────────────────────────────────────────────────────────────────────────
# DIARISATION BUILDER
# ───────────────────────────────────────────────────────────────────────────────
FROM base-cuda AS diarisation-builder

RUN pip install --no-cache-dir --timeout=300 \
        celery==5.4.0 \
        kombu==5.3.4 \
        redis==4.6.0

RUN pip install --no-cache-dir --timeout=600 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        torch==2.3.0+cu121 \
        torchaudio==2.3.0+cu121

RUN pip install --no-cache-dir --timeout=300 \
        numpy==1.26.4 \
        scipy==1.13.0 \
        matplotlib==3.8.4 \
        soundfile==0.12.1

RUN pip install --no-cache-dir --timeout=600 \
        huggingface-hub==0.24.7 \
        pyannote.audio==3.3.2


# ───────────────────────────────────────────────────────────────────────────────
# DIARISATION RUNTIME  (target: diarisation)
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS diarisation

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY --from=diarisation-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Verify venv
RUN python -c "import pyannote.audio; print('pyannote.audio OK')"
RUN python -c "import torch; print('torch OK')"

# pyannote models download on first container start via HUGGINGFACE_TOKEN.
# Cached in hf_cache volume — never re-downloaded on restart.

WORKDIR /app
COPY diarisation_service/app.py \
     diarisation_service/aligner.py \
     observability.py ./

ENV STORAGE_DIR=/data/sessions
ENV REDIS_URL=redis://redis:6379/0
ENV DIAR_MODEL=pyannote/speaker-diarization-3.1
ENV DIAR_DEVICE=cuda
ENV MIN_SPEAKERS=1
ENV MAX_SPEAKERS=6

RUN mkdir -p /data/sessions
VOLUME ["/root/.cache/huggingface"]

CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", \
     "--loglevel=info", "--queues=diarisation", "--concurrency=1"]


# ───────────────────────────────────────────────────────────────────────────────
# NLP BUILDER  (Stage 5)
# ───────────────────────────────────────────────────────────────────────────────
FROM base-cuda AS nlp-builder

RUN pip install --no-cache-dir --timeout=300 \
        celery==5.4.0 \
        kombu==5.3.4 \
        redis==4.6.0

# Copy models into the image after training


# PyTorch CUDA — shared with asr/diarisation builder cache
RUN pip install --no-cache-dir --timeout=600 \
        --index-url https://download.pytorch.org/whl/cu121 \
        --extra-index-url https://pypi.org/simple \
        torch==2.3.0+cu121 \
        torchaudio==2.3.0+cu121

# Transformers + NLP stack
RUN pip install --no-cache-dir --timeout=300 \
        transformers==4.41.2 \
        accelerate==0.30.1 \
        sentencepiece==0.2.0 \
        protobuf==4.25.3

# spaCy + transformer model
RUN pip install --no-cache-dir --timeout=300 \
    spacy==3.7.4 \
    spacy-transformers==1.3.5 \
    https://github.com/explosion/spacy-models/releases/download/en_core_web_trf-3.7.3/en_core_web_trf-3.7.3-py3-none-any.whl

# KeyBERT + sentence-transformers (LDA uses sklearn — no gensim/C++ needed)
RUN pip install --no-cache-dir --timeout=300 \
        keybert==0.8.5 \
        sentence-transformers==3.0.1

# Misc NLP utilities
RUN pip install --no-cache-dir --timeout=300 \
        requests==2.32.3 \
        numpy==1.26.4 \
        scipy==1.13.0 \
        scikit-learn==1.5.0


# ───────────────────────────────────────────────────────────────────────────────
# NLP RUNTIME  (target: nlp)
# ───────────────────────────────────────────────────────────────────────────────
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04 AS nlp

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv \
        libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.11 /usr/bin/python \
 && ln -sf /usr/bin/python3.11 /usr/bin/python3

COPY --from=nlp-builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Verify key imports
RUN python -c "import spacy; nlp=spacy.load('en_core_web_trf'); print('spaCy OK')"
RUN python -c "from keybert import KeyBERT; print('KeyBERT OK')"
RUN python -c "from transformers import pipeline; print('transformers OK')"

WORKDIR /app

COPY nlp_service/app.py \
     nlp_service/topics.py \
     nlp_service/entities.py \
     nlp_service/action_items.py \
     nlp_service/pain_points.py \
     nlp_service/sentiment.py \
     observability.py ./

# models/ directory — populated at training time, optional at build time
RUN mkdir -p /app/models/pain_point_classifier

ENV STORAGE_DIR=/data/sessions
ENV REDIS_URL=redis://redis:6379/0
ENV LLM_MODEL=llama3.2:1b
ENV OLLAMA_URL=http://ollama:11434

RUN mkdir -p /data/sessions
VOLUME ["/root/.cache/huggingface"]

CMD ["python", "-m", "celery", "-A", "app.celery_app", "worker", \
     "--loglevel=info", "--queues=nlp", "--concurrency=2"]


# ───────────────────────────────────────────────────────────────────────────────
# GENAI SERVICE  (Stage 6)  target: genai
# Lightweight CPU-only service — no GPU needed.
# The LLM runs in the Ollama container (GPU) or OpenAI API (cloud).
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS genai

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip==24.0

RUN pip install --no-cache-dir --timeout=300 \
        fastapi==0.111.0 \
        uvicorn[standard]==0.29.0 \
        aiohttp==3.9.5 \
        openai==1.35.3 \
        python-dotenv==1.0.1

WORKDIR /app

COPY genai_service/app.py \
     genai_service/report_builder.py \
     genai_service/streamer.py \
     observability.py ./

ENV STORAGE_DIR=/data/sessions
ENV OLLAMA_URL=http://ollama:11434
ENV OLLAMA_MODEL=llama3
ENV LLM_BACKEND=ollama
ENV OPENAI_MODEL=gpt-4o

RUN mkdir -p /data/sessions

EXPOSE 8001

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8001", "--workers", "1"]


# ───────────────────────────────────────────────────────────────────────────────
# API GATEWAY  (target: gateway)
# Lightweight CPU-only — just FastAPI + Redis + aiohttp + Prometheus
# ───────────────────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS gateway

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"
RUN pip install --upgrade pip==24.0

RUN pip install --no-cache-dir --timeout=300 \
        fastapi==0.111.0 \
        uvicorn[standard]==0.29.0 \
        aiohttp==3.9.5 \
        redis==4.6.0 \
        celery==5.4.0 \
        kombu==5.3.4 \
        prometheus-client==0.20.0 \
        python-multipart==0.0.9 \
        psycopg[binary]==3.2.1

RUN python -c "import fastapi, aiohttp, redis, celery, prometheus_client, psycopg; print('gateway deps OK')"

WORKDIR /app
COPY api_gateway/main.py \
     api_gateway/metrics.py \
     api_gateway/persistence.py \
     observability.py ./

ENV REDIS_URL=redis://redis:6379/0
ENV INGESTION_URL=http://ingestion:8000
ENV GENAI_URL=http://genai:8001
ENV STORAGE_DIR=/data/sessions
ENV DATABASE_URL=postgresql://meeting:meeting@postgres:5432/meeting_ai
ENV SEARCH_BACKEND=meilisearch
ENV SEARCH_URL=http://meilisearch:7700

RUN mkdir -p /data/sessions

EXPOSE 8080
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "2"]


FROM node:20-alpine AS frontend

WORKDIR /app/frontend

# Install dependencies first for Docker layer caching
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm install

# Copy frontend source
COPY frontend/ ./

# Expose Vite dev server port
EXPOSE 5173

# --host 0.0.0.0 is already set in vite.config.js server.host
# VITE_PROXY_TARGET is injected via docker-compose environment block
CMD ["npm", "run", "dev"]
