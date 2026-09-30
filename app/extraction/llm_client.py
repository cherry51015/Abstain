"""
Async client for any OpenAI-compatible chat-completions endpoint (Groq by
default), written directly on httpx rather than through an SDK so the
reliability behaviour is explicit and testable:

  - per-request timeout
  - retries only on retryable failures (429, 5xx, timeouts, connection
    errors), exponential backoff with full jitter, honouring Retry-After
  - a concurrency cap shared by all callers, so self-consistency sampling
    and concurrent API requests cannot stampede the provider's rate limit
  - optional client-side pacing (requests per minute), so a known quota is
    respected up front instead of discovered through 429s
  - an optional content-addressed response cache (see cache.py)
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
import time
from dataclasses import dataclass

import httpx

from app import observability as obs
from app.extraction.cache import ResponseCache

logger = logging.getLogger("abstain.llm")
_RETRYABLE_STATUS = {408, 409, 425, 429, 500, 502, 503, 504}


class LLMError(Exception):
    """Non-retryable failure, or retries exhausted."""


@dataclass(frozen=True)
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    latency_s: float
    cached: bool


class OpenAICompatibleClient:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        base_url: str = "https://api.groq.com/openai/v1",
        timeout_s: float = 20.0,
        max_retries: int = 3,
        max_concurrency: int = 4,
        backoff_base_s: float = 1.0,
        backoff_cap_s: float = 30.0,
        max_requests_per_minute: float | None = None,
        extra_body: dict | None = None,
        cache: ResponseCache | None = None,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        self.model = model
        self.max_retries = max_retries
        self.backoff_base_s = backoff_base_s
        self.backoff_cap_s = backoff_cap_s
        self.extra_body = extra_body or {}
        self.cache = cache
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._min_interval_s = 60.0 / max_requests_per_minute if max_requests_per_minute else 0.0
        self._next_slot = 0.0
        self._pace_lock = asyncio.Lock()
        self._http = httpx.AsyncClient(
            base_url=base_url,
            headers={"Authorization": f"Bearer {api_key}"},
            timeout=httpx.Timeout(timeout_s),
            transport=transport,
        )

    async def aclose(self) -> None:
        await self._http.aclose()

    def cache_key(self, messages: list[dict], temperature: float, sample_index: int) -> str:
        payload = json.dumps([self.model, messages, temperature, sample_index], sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()

    async def complete(self, messages: list[dict], *, temperature: float, sample_index: int = 0,
                       max_tokens: int = 800) -> LLMResponse:
        key = self.cache_key(messages, temperature, sample_index)
        if self.cache and (hit := self.cache.get(key)) is not None:
            obs.LLM_CALLS.labels(outcome="cache_hit").inc()
            return LLMResponse(text=hit, prompt_tokens=0, completion_tokens=0, latency_s=0.0, cached=True)

        body = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "response_format": {"type": "json_object"},
            **self.extra_body,
        }
        async with self._semaphore:
            response = await self._post_with_retries(body)
        if self.cache:
            self.cache.put(key, response.text)
        return response

    async def _post_with_retries(self, body: dict) -> LLMResponse:
        last_error: str = ""
        for attempt in range(self.max_retries + 1):
            await self._pace()
            started = time.perf_counter()
            try:
                r = await self._http.post("/chat/completions", json=body)
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
                retry_after = None
            else:
                latency = time.perf_counter() - started
                if r.status_code == 200:
                    data = r.json()
                    usage = data.get("usage") or {}
                    obs.LLM_CALLS.labels(outcome="ok").inc()
                    obs.LLM_LATENCY.observe(latency)
                    obs.LLM_TOKENS.labels(kind="prompt").inc(usage.get("prompt_tokens", 0))
                    obs.LLM_TOKENS.labels(kind="completion").inc(usage.get("completion_tokens", 0))
                    return LLMResponse(
                        text=data["choices"][0]["message"]["content"] or "",
                        prompt_tokens=usage.get("prompt_tokens", 0),
                        completion_tokens=usage.get("completion_tokens", 0),
                        latency_s=latency,
                        cached=False,
                    )
                last_error = f"HTTP {r.status_code}: {r.text[:200]}"
                if r.status_code not in _RETRYABLE_STATUS:
                    obs.LLM_CALLS.labels(outcome="error").inc()
                    raise LLMError(last_error)
                retry_after = _retry_after_seconds(r)

            if attempt == self.max_retries:
                break
            obs.LLM_CALLS.labels(outcome="retry").inc()
            delay = retry_after if retry_after is not None else random.uniform(
                0, min(self.backoff_cap_s, self.backoff_base_s * 2 ** attempt))
            logger.warning("llm call failed, retrying", extra={"attempt": attempt + 1, "error": last_error,
                                                                "delay_s": round(delay, 2)})
            await asyncio.sleep(min(delay, self.backoff_cap_s))

        obs.LLM_CALLS.labels(outcome="exhausted").inc()
        raise LLMError(f"retries exhausted: {last_error}")

    async def _pace(self) -> None:
        """Reserve the next send slot so requests are spaced min_interval apart."""
        if not self._min_interval_s:
            return
        async with self._pace_lock:
            now = time.perf_counter()
            slot = max(now, self._next_slot)
            self._next_slot = slot + self._min_interval_s
        # asyncio's timers run on time.monotonic(), which ticks at ~15.6 ms on Windows, so a
        # single sleep can wake early and send ahead of the quota. Re-check on the
        # high-resolution clock until the reserved slot has really arrived.
        while (remaining := slot - time.perf_counter()) > 0:  # noqa: ASYNC110 - waiting on a clock, not an event
            await asyncio.sleep(remaining)


def _retry_after_seconds(r: httpx.Response) -> float | None:
    value = r.headers.get("retry-after")
    if value is None:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        return None
