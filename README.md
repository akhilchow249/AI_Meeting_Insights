# 🎙️ AI Meeting Insights Platform

An end-to-end, enterprise-grade microservices system designed for processing, transcribing, diarizing, and generating AI insights from meeting recordings. Built with custom AI models, local LLMs, full-text search engines, and full observability (Prometheus, Grafana, Loki, Promtail).

---

## 🏗️ System Architecture & Services Overview

The platform consists of distributed microservices and infrastructure components orchestrated via Docker Compose:

### 🧩 Core Microservices

* **API Gateway** (`meeting-ai-gateway`): Central entry point routing requests to internal services — **Port `8080`**
* **Ingestion Service** (`meeting-ai-ingestion`): Handles media uploads and audio pre-processing — **Port `8000`**
* **GenAI Service** (`meeting-ai-genai`): Handles LLM prompt execution & action item generation — **Port `8001`**
* **ASR Service** (`meeting-ai-asr`): Automatic Speech Recognition for transcript generation
* **Diarisation Service** (`meeting-ai-diarisation`): Identifies and separates distinct speakers in audio
* **NLP Service** (`meeting-ai-nlp`): Summarization and keyword extraction pipeline
* **Frontend** (`meeting-ai-frontend`): Web dashboard to upload media and view meeting analytics — **Port `5173`**

### 💾 Storage & Search Infrastructure

* **PostgreSQL** (`postgres:16-alpine`): Primary relational data store — **Port `5432`**
* **Redis** (`redis:7.2-alpine`): Caching and async job queue management — **Port `6379`**
* **Meilisearch** (`getmeili/meilisearch:v1.12`): High-performance full-text search engine — **Port `7700`**
* **Ollama** (`ollama/ollama:latest`): Local LLM inference engine — **Port `11434`**

### 📊 Observability & Telemetry

* **Prometheus** (`prom/prometheus:v2.54.1`): Scrapes and monitors system performance metrics — **Port `9090`**
* **Grafana** (`grafana/grafana:11.1.5`): Visualization dashboards for system health — **Port `3000`**
* **Loki** (`grafana/loki:3.1.0`): Centralized log aggregation — **Port `3100`**
* **Promtail** (`grafana/promtail:3.1.0`): Ships container logs directly to Loki

---

## 🌐 Port Mapping Reference

| Service | Container / Image | Port Mapping | Access URL |
| :--- | :--- | :--- | :--- |
| **Frontend** | `meeting-ai-frontend` | `5173:5173` | `http://localhost:5173` |
| **API Gateway** | `meeting-ai-gateway` | `8080:8080` | `http://localhost:8080` |
| **Ingestion Service** | `meeting-ai-ingestion` | `8000:8000` | `http://localhost:8000` |
| **GenAI Service** | `meeting-ai-genai` | `8001:8001` | `http://localhost:8001` |
| **Grafana** | `grafana/grafana:11.1.5` | `3000:3000` | `http://localhost:3000` |
| **Prometheus** | `prom/prometheus:v2.54.1` | `9090:9090` | `http://localhost:9090` |
| **Loki** | `grafana/loki:3.1.0` | `3100:3100` | `http://localhost:3100` |
| **Meilisearch** | `getmeili/meilisearch:v1.12` | `7700:7700` | `http://localhost:7700` |
| **Ollama** | `ollama/ollama:latest` | `11434:11434` | `http://localhost:11434` |
| **PostgreSQL** | `postgres:16-alpine` | `5432:5432` | `localhost:5432` |
| **Redis** | `redis:7.2-alpine` | `6379:6379` | `localhost:6379` |

---

## 📂 Repository Structure

```text
AI_Meeting_Insights/
├── api_gateway/            # API gateway service
├── asr_service/            # Speech-to-text processing (Whisper/custom)
├── diarisation_service/    # Speaker diarization models (PyAnnote/HF)
├── frontend/               # Frontend user interface
├── genai_service/          # Generative AI summary & insights pipeline
├── ingestion_service/      # Audio ingestion & media pre-processing
├── nlp_service/            # Text NLP utilities
├── observability/          # Prometheus & Promtail configs
│   ├── prometheus/
│   └── promtail/
├── .gitignore              # Ignored files & secrets rules
├── Dockerfile              # Base Docker configuration
├── Requirements.txt        # Top-level Python dependencies
├── docker-compose.yml      # Orchestration for all services
└── README.md               # Project documentation
```

---

## 🔑 Step-by-Step: How to Get a Hugging Face Access Token

The **Diarisation** and **ASR** services require PyAnnote / Hugging Face model weights to operate.

1. Navigate to [Hugging Face](https://huggingface.co/) and sign up or log into your account.
2. Go to your User Profile (top right icon) and select **Settings**.
3. Click on **Access Tokens** from the left-side menu (or go directly to [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)).
4. Click **Create new token**.
5. Give your token a name (e.g., `Meeting-AI-Token`), select **Read** or **Write** scope, and click **Generate a token**.
6. **Copy the generated token** (`hf_...`). You will place this in your `.env` file in the next step.

> ⚠️ **Note:** Make sure you accept the user terms for `pyannote/speaker-diarization-3.1` and `pyannote/segmentation-3.0` on Hugging Face using the same account.

---

## 🚀 Getting Started & Installation

### 1. Prerequisites
* **Docker Desktop** installed & running.
* **Python 3.9+** installed locally.
* **Git** installed.

### 2. Clone the Repository
```bash
git clone [https://github.com/akhilchow249/AI_Meeting_Insights.git](https://github.com/akhilchow249/AI_Meeting_Insights.git)
cd AI_Meeting_Insights
```

### 3. Install Python Dependencies
Set up a local environment (optional) and install requirements:

```bash
# Optional: Create a local virtual environment
python -m venv .venv

# Activate on Windows PowerShell:
.\.venv\Scripts\Activate.ps1

# Activate on Linux/macOS:
# source .venv/bin/activate

# Install requirements
pip install -r Requirements.txt
```

### 4. Configure Environment Variables
Create a `.env` file in the root folder of the project:

```bash
cp .env.example .env
```

Open `.env` and add your Hugging Face token alongside your database credentials:

```env
# Hugging Face Configuration
HUGGINGFACE_TOKEN=hf_your_actual_token_here

# Database Configuration
POSTGRES_USER=postgres
POSTGRES_PASSWORD=postgres
POSTGRES_DB=meeting_ai

# Search Engine
MEILI_MASTER_KEY=masterKey

# Port Overrides (Optional)
GATEWAY_PORT=8080
INGESTION_PORT=8000
GENAI_PORT=8001
FRONTEND_PORT=5173
```

---

## 🐳 Running with Docker Compose

To start the complete platform including microservices, databases, search engines, and observability stack:

### 1. Start all containers in detached mode:
```bash
docker compose up -d
```

### 2. Check the status of all containers:
```bash
docker compose ps
```

### 3. View logs for a specific service:
```bash
# View all logs
docker compose logs -f

# View logs for a specific service (e.g., API Gateway)
docker compose logs -f gateway
```

### 4. Stop the services:
```bash
docker compose down
```

---

## 📈 Observability & Dashboards

Once Docker Compose is running, access the monitoring stack:

* **Grafana Dashboards**: Navigate to `http://localhost:3000` (Default credentials: `admin` / `admin`).
* **Prometheus Target Metrics**: Navigate to `http://localhost:9090/targets` to verify service health.
* **Loki Logs**: Query application logs directly in Grafana under the **Explore** tab selecting `Loki` as the datasource.

---

## 👤 Author

* **Akhil Chowdary** - [@akhilchow249](https://github.com/akhilchow249)
