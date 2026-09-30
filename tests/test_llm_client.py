from __future__ import annotations

import asyncio
import time

import httpx
import pytest

from app.extraction.cache import ResponseCache
from app.extraction.llm_client import LLMError, OpenAICompatibleClient
from tests.conftest import chat_response

MSGS = [{"role": "user", "content": "json please"}]


def client_with(responses: list, **kw) -> tuple[OpenAICompatibleClient, list]:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        r = responses.pop(0)
        if isinstance(r, Exception):
            raise r
        return r

    kw.setdefault("backoff_base_s", 0.001)
    return OpenAICompatibleClient(api_key="k", model="m", transport=httpx.MockTransport(handler), **kw), calls


def test_success_returns_text_and_usage():
    client, calls = client_with([chat_response('{"a": 1}')])
    r = asyncio.run(client.complete(MSGS, temperature=0.5))
    assert r.text == '{"a": 1}' and r.prompt_tokens == 100 and not r.cached
    assert calls[0].headers["authorization"] == "Bearer k"


def test_retries_429_honouring_retry_after():
    client, calls = client_with([chat_response("slow down", 429, {"retry-after": "0.05"}), chat_response("{}")])
    started = time.perf_counter()
    asyncio.run(client.complete(MSGS, temperature=0.5))
    assert len(calls) == 2 and time.perf_counter() - started >= 0.05


def test_retries_timeouts_and_5xx():
    client, calls = client_with([httpx.ReadTimeout("t"), chat_response("oops", 503), chat_response("{}")])
    asyncio.run(client.complete(MSGS, temperature=0.5))
    assert len(calls) == 3


def test_does_not_retry_client_errors():
    client, calls = client_with([chat_response("bad request", 400)])
    with pytest.raises(LLMError, match="400"):
        asyncio.run(client.complete(MSGS, temperature=0.5))
    assert len(calls) == 1


def test_gives_up_after_max_retries():
    client, calls = client_with([chat_response("x", 500) for _ in range(3)], max_retries=2)
    with pytest.raises(LLMError, match="exhausted"):
        asyncio.run(client.complete(MSGS, temperature=0.5))
    assert len(calls) == 3


def test_cache_hit_skips_the_network(tmp_path):
    cache = ResponseCache(tmp_path / "c.sqlite")
    client, calls = client_with([chat_response('{"x": 1}')], cache=cache)
    first = asyncio.run(client.complete(MSGS, temperature=0.5, sample_index=0))
    second = asyncio.run(client.complete(MSGS, temperature=0.5, sample_index=0))
    assert len(calls) == 1 and second.cached and second.text == first.text


def test_different_sample_index_is_a_different_cache_entry(tmp_path):
    client, calls = client_with([chat_response("{}"), chat_response("{}")], cache=ResponseCache(tmp_path / "c"))
    asyncio.run(client.complete(MSGS, temperature=0.5, sample_index=0))
    asyncio.run(client.complete(MSGS, temperature=0.5, sample_index=1))
    assert len(calls) == 2


def test_pacing_spaces_requests():
    client, calls = client_with([chat_response("{}") for _ in range(3)], max_requests_per_minute=600)  # 0.1 s apart

    async def burst():
        await asyncio.gather(*(client.complete(MSGS, temperature=0.5, sample_index=i) for i in range(3)))

    started = time.perf_counter()
    asyncio.run(burst())
    assert time.perf_counter() - started >= 0.2
