"""
genai-service/streamer.py
──────────────────────────
Async token streamer supporting two LLM backends:
  - Ollama  (local, privacy-preserving, free — llama3, mistral, etc.)
  - OpenAI  (cloud, highest quality — gpt-4o)

Both backends use async HTTP with aiohttp so the FastAPI event loop
is never blocked while waiting for tokens.

Each backend yields events as dicts:
  {"type": "token",   "content": "<token text>"}
  {"type": "error",   "message": "<error description>"}

The caller (app.py) wraps these in SSE format and pushes them to
the browser.

Retry logic
───────────
Both backends retry up to MAX_RETRIES times on connection errors.
On Ollama, if the model is not yet pulled, a helpful error is emitted.
"""

from __future__ import annotations
import asyncio
import json
import logging
from enum import Enum
from typing import AsyncGenerator

import aiohttp

logger = logging.getLogger(__name__)

MAX_RETRIES   = 2
RETRY_DELAY   = 3.0   # seconds between retries


class LLMBackend(str, Enum):
    OLLAMA = "ollama"
    OPENAI = "openai"


class ReportStreamer:
    """
    Streams LLM tokens from either Ollama or OpenAI.

    Parameters
    ----------
    backend        : LLMBackend
    ollama_url     : str   Base URL of Ollama API (e.g. http://ollama:11434)
    ollama_model   : str   Model name (e.g. "llama3", "mistral")
    openai_api_key : str   OpenAI API key (only needed for OPENAI backend)
    openai_model   : str   OpenAI model name (e.g. "gpt-4o")
    temperature    : float Generation temperature (default 0.3 for factual reports)
    max_tokens     : int   Maximum tokens to generate
    """

    def __init__(
        self,
        backend:        LLMBackend = LLMBackend.OLLAMA,
        ollama_url:     str        = "http://ollama:11434",
        ollama_model:   str        = "llama3",
        openai_api_key: str        = "",
        openai_model:   str        = "gpt-4o",
        temperature:    float      = 0.1,
        max_tokens:     int        = 1000,
    ):
        self.backend        = backend
        self.ollama_url     = ollama_url.rstrip("/")
        self.ollama_model   = ollama_model
        self.openai_api_key = openai_api_key
        self.openai_model   = openai_model
        self.temperature    = temperature
        self.max_tokens     = max_tokens

    async def stream(self, prompt: str) -> AsyncGenerator[dict, None]:
        """
        Yield token events from the configured LLM backend.

        Usage:
            async for event in streamer.stream(prompt):
                if event["type"] == "token":
                    print(event["content"], end="", flush=True)
        """
        for attempt in range(MAX_RETRIES + 1):
            try:
                if self.backend == LLMBackend.OLLAMA:
                    async for event in self._stream_ollama(prompt):
                        yield event
                else:
                    async for event in self._stream_openai(prompt):
                        yield event
                return  # success

            except aiohttp.ClientConnectorError as exc:
                if attempt < MAX_RETRIES:
                    logger.warning(
                        "LLM connection failed (attempt %d/%d): %s — retrying in %ss",
                        attempt + 1, MAX_RETRIES + 1, exc, RETRY_DELAY,
                    )
                    await asyncio.sleep(RETRY_DELAY)
                else:
                    yield {"type": "error",
                           "message": f"Cannot connect to LLM backend after "
                                      f"{MAX_RETRIES + 1} attempts: {exc}"}

            except Exception as exc:
                yield {"type": "error", "message": f"LLM streaming error: {exc}"}
                return

    # ── Ollama backend ────────────────────────────────────────────────────────

    async def _stream_ollama(self, prompt: str) -> AsyncGenerator[dict, None]:
        """
        Stream tokens from Ollama /api/generate endpoint.

        Ollama streams newline-delimited JSON objects:
          {"model":"llama3","response":" The","done":false}
          {"model":"llama3","response":"","done":true}
        """
        url     = f"{self.ollama_url}/api/generate"
        payload = {
            "model":   self.ollama_model,
            "prompt":  prompt,
            "stream":  True,
            "keep_alive": "30m",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
                "num_ctx": 3072,
                "stop": [],
            },
        }

        timeout = aiohttp.ClientTimeout(
            total=None,        # no total timeout — report can take several minutes
            connect=10,
            sock_read=120,     # 2 min read timeout per chunk
        )

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=payload) as resp:
                if resp.status == 404:
                    yield {
                        "type": "error",
                        "message": (
                            f"Ollama model '{self.ollama_model}' not found. "
                            f"Run: docker exec meeting_ollama ollama pull {self.ollama_model}"
                        ),
                    }
                    return

                if resp.status != 200:
                    body = await resp.text()
                    yield {"type": "error",
                           "message": f"Ollama returned HTTP {resp.status}: {body[:200]}"}
                    return

                async for raw_line in resp.content:
                    line = raw_line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    token = obj.get("response", "")
                    if token:
                        yield {"type": "token", "content": token}

                    if obj.get("done"):
                        break

    # ── OpenAI backend ────────────────────────────────────────────────────────

    async def _stream_openai(self, prompt: str) -> AsyncGenerator[dict, None]:
        """
        Stream tokens from OpenAI Chat Completions API with stream=True.

        OpenAI streams server-sent events:
          data: {"choices":[{"delta":{"content":" The"},"finish_reason":null}]}
          data: [DONE]
        """
        if not self.openai_api_key:
            yield {"type": "error",
                   "message": "OPENAI_API_KEY is not set. "
                               "Set it in your .env file or use LLM_BACKEND=ollama."}
            return

        url     = "https://api.openai.com/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {self.openai_api_key}",
            "Content-Type":  "application/json",
        }
        payload = {
            "model":       self.openai_model,
            "messages":    [{"role": "user", "content": prompt}],
            "stream":      True,
            "temperature": self.temperature,
            "max_tokens":  self.max_tokens,
        }

        timeout = aiohttp.ClientTimeout(total=None, connect=10, sock_read=120)

        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, headers=headers, json=payload) as resp:
                if resp.status == 401:
                    yield {"type": "error",
                           "message": "OpenAI API key is invalid or expired."}
                    return

                if resp.status == 429:
                    yield {"type": "error",
                           "message": "OpenAI rate limit exceeded. "
                                      "Wait a moment or switch to LLM_BACKEND=ollama."}
                    return

                if resp.status != 200:
                    body = await resp.text()
                    yield {"type": "error",
                           "message": f"OpenAI returned HTTP {resp.status}: {body[:200]}"}
                    return

                async for raw_line in resp.content:
                    line = raw_line.decode("utf-8").strip()
                    if not line or line == "data: [DONE]":
                        continue
                    if not line.startswith("data: "):
                        continue

                    try:
                        obj    = json.loads(line[6:])   # strip "data: " prefix
                        delta  = obj["choices"][0]["delta"]
                        token  = delta.get("content", "")
                        reason = obj["choices"][0].get("finish_reason")
                    except (json.JSONDecodeError, KeyError, IndexError):
                        continue

                    if token:
                        yield {"type": "token", "content": token}

                    if reason == "stop":
                        break
