from __future__ import annotations

import asyncio
from decimal import Decimal

import httpx

from app.domain import Tri
from app.extraction.llm_client import OpenAICompatibleClient
from app.extraction.llm_extractor import LLMExtractor
from app.service import DisputeService
from tests.conftest import SHIFTED_DOCS, chat_response, llm_payload, make_case

GOOD = llm_payload(delivery_confirmed=("yes", "handed to the addressee"),
                   signed_by_cardholder=("yes", "POD signature: Priya Nair"),
                   customer_acknowledged_receipt=("yes", "Collected it from the front desk"))


def service(catalog, model, engine, handler=None, mode="cascade", budget=5.0):
    llm = None
    calls = []
    if handler is not None:
        def wrapped(request):
            calls.append(request)
            return handler(request)
        client = OpenAICompatibleClient(api_key="k", model="m", transport=httpx.MockTransport(wrapped),
                                        max_retries=0)
        llm = LLMExtractor(client, n_samples=3)
    return DisputeService(catalog=catalog, model=model, engine=engine, llm=llm, mode=mode, llm_budget_s=budget), calls


def test_cascade_skips_llm_when_rules_read_everything(catalog, model, engine):
    svc, calls = service(catalog, model, engine, handler=lambda r: chat_response(GOOD))
    ev = asyncio.run(svc.evaluate(make_case()))
    assert calls == [] and ev.extraction.source == "rules" and ev.unread_facts == []


def test_cascade_calls_llm_for_unfamiliar_phrasing(catalog, model, engine):
    svc, calls = service(catalog, model, engine, handler=lambda r: chat_response(GOOD))
    ev = asyncio.run(svc.evaluate(make_case(documents=SHIFTED_DOCS)))
    assert len(calls) == 3 and ev.extraction.source == "cascade"
    assert ev.facts.delivery_confirmed == Tri.YES and ev.facts.customer_acknowledged_receipt == Tri.YES
    assert ev.degraded_reason is None


def test_llm_failure_falls_back_to_rules_with_wider_uncertainty(catalog, model, engine):
    healthy, _ = service(catalog, model, engine, handler=lambda r: chat_response(GOOD))
    broken, _ = service(catalog, model, engine, handler=lambda r: chat_response("down", 503))
    case = make_case(documents=SHIFTED_DOCS)
    ok = asyncio.run(healthy.evaluate(case))
    bad = asyncio.run(broken.evaluate(case))
    assert bad.extraction.source == "rules_fallback" and bad.degraded_reason == "llm_failed"
    assert set(bad.unread_facts) >= {"delivery_confirmed", "customer_acknowledged_receipt"}
    assert bad.estimate.std > ok.estimate.std
    assert any("Degraded run" in r for r in bad.decision.reasons)


def test_llm_timeout_is_a_degraded_run_not_an_error(catalog, model, engine):
    async def slow(request):
        await asyncio.sleep(1)
        return chat_response(GOOD)

    svc, _ = service(catalog, model, engine, handler=slow, budget=0.05)
    ev = asyncio.run(svc.evaluate(make_case(documents=SHIFTED_DOCS)))
    assert ev.degraded_reason == "llm_timeout"


def test_imputation_is_deterministic_per_case(catalog, model, engine):
    svc, _ = service(catalog, model, engine)
    case = make_case(documents=SHIFTED_DOCS, amount_inr=Decimal("20000"))
    a, b = asyncio.run(svc.evaluate(case)), asyncio.run(svc.evaluate(case))
    assert (a.estimate.mean, a.estimate.std, a.decision.action) == (b.estimate.mean, b.estimate.std, b.decision.action)
