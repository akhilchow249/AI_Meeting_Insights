# MeetingIQ Frontend

React + Vite + TailwindCSS + WaveSurfer.js — fully wired to the FastAPI gateway.

## Quick Start

### Docker (recommended — runs all services)

```bash
cp .env.docker.example .env
# edit .env — set HUGGINGFACE_TOKEN at minimum
docker-compose up --build
```

Open **http://localhost:5173**

The frontend container proxies all `/api/*`, `/health`, `/metrics` requests to
`http://gateway:8080` via Vite's dev server proxy (Docker-internal DNS).

### Local dev (frontend only, gateway running separately)

```bash
npm install
VITE_PROXY_TARGET=http://localhost:8080 npm run dev
```

---

## Wiring Diagram

```
Browser (localhost:5173)
  │  relative /api/* requests
  ▼
Vite dev server  (frontend container :5173)
  │  proxy: VITE_PROXY_TARGET=http://gateway:8080
  ▼
API Gateway  (meeting_gateway :8080)
  ├── POST /api/upload          → ingestion :8000
  ├── GET  /api/sessions/*/progress (SSE)  ← Redis pub/sub
  ├── GET  /api/sessions/*/report/stream (SSE) → genai :8001
  ├── GET  /api/sessions/*/transcript/diarised
  ├── GET  /api/sessions/*/nlp
  ├── GET  /api/sessions/*/video  (Range-aware)
  ├── GET  /api/sessions/*/audio  (WAV for WaveSurfer)
  └── GET  /api/sessions  (library list)
```

## API Client: `src/api.js`

Every gateway endpoint has a typed wrapper. Key corrections vs the original:

| Function | Endpoint | Note |
|---|---|---|
| `uploadVideo` | `POST /api/upload` | Uses `BASE` (not `VITE_API_BASE_URL`) |
| `openProgressStream` | `GET /api/sessions/{id}/progress` (SSE) | Does not auto-close on error (SSE retries) |
| `openReportStream` | `GET /api/sessions/{id}/report/stream` | **/stream suffix** — not /report |
| `fetchDiarisedTranscript` | `GET /api/sessions/{id}/transcript/diarised` | New — used by Report screen |
| `fetchNlpResults` | `GET /api/sessions/{id}/nlp` | New — pain points, actions, topics |
| `fetchSessions` | `GET /api/sessions?limit=N&offset=N` | No ?q= server-side filter |

## SSE Event Reference (from `main.py _progress_sse_generator`)

| `type` field | When fired | Handled in |
|---|---|---|
| `state_catchup` | On SSE connect — current stage state | Pipeline.jsx |
| `transcript_preview_catchup` | On SSE connect — buffered word batches | Pipeline.jsx |
| `transcript_preview` | Live during ASR (Stage 3) | Pipeline.jsx |
| `heartbeat` | Every 15s (HEARTBEAT_INTERVAL) | Pipeline.jsx (ignored) |
| `pipeline_complete` | All stages done | Pipeline.jsx → `onComplete` |
| plain `{stage, status, percent}` | Every stage transition | Pipeline.jsx |

## Stage Key Mapping

Frontend `STAGE_IDX` in `Pipeline.jsx` maps backend stage keys to array indices:

| Backend key | Frontend label | Index |
|---|---|---|
| `video_ingestion` | Video Ingestion | 0 |
| `audio_extraction` | Audio Extraction | 1 |
| `transcription` | Speech-to-Text | 2 |
| `diarisation` | Speaker Diarisation | 3 |
| `nlp_analysis` | NLP Analysis | 4 |
| `genai_report` | AI Report Generation | 5 |
| `indexing` | Indexing | 6 |

## Retry Logic

Only `asr` and `nlp` are retryable via `POST /api/sessions/{id}/retry/{stage}`
(as defined in `RETRYABLE` in `main.py`). The frontend maps:
- `transcription` stage key → `asr` backend key
- `nlp_analysis` stage key  → `nlp` backend key

## Environment Variables

| Variable | Where set | Purpose |
|---|---|---|
| `VITE_PROXY_TARGET` | docker-compose / shell | Vite proxy destination (`http://gateway:8080`) |
| `VITE_API_BASE` | docker-compose / .env | Browser API prefix (empty = relative URLs) |
| `HUGGINGFACE_TOKEN` | .env | pyannote model access |
| `OPENAI_API_KEY` | .env | Optional OpenAI GenAI backend |
