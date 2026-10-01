from __future__ import annotations

import dataclasses

import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import create_app
from tests.conftest import make_case

UNREADABLE_AUTH_RECORD = {
    "doc_type": "order_record",
    "text": "Order #4411\nAuth response: address_check=PASS, cvc_check=PASS\n"
            "Checkout session from an ISP in Pune; cardholder resides in Pune.",
}


@pytest.fixture
def client_factory(tmp_path):
    def make(**settings_overrides) -> TestClient:
        overrides = {"database_url": f"sqlite:///{tmp_path / 'api.db'}", "llm_api_key": None, "api_key": None,
                     **settings_overrides}
        settings = dataclasses.replace(Settings(), **overrides)
        return TestClient(create_app(settings, create_schema=True))
    return make


@pytest.fixture
def client(client_factory):
    with client_factory() as c:
        yield c


def body(**overrides) -> dict:
    return make_case(**overrides).model_dump(mode="json")


def escalating_body(case_id: str, amount: int = 90_000) -> dict:
    # Fraud dispute whose auth record uses phrasing the rules cannot read (and no LLM is configured):
    # the facts are imputed, P(win) is near break-even with a wide spread, so a review is worth its cost.
    return body(case_id=case_id, reason_code="10.4", amount_inr=amount, documents=[UNREADABLE_AUTH_RECORD])


def test_health_ready_and_request_id(client):
    assert client.get("/health").json() == {"status": "ok"}
    r = client.get("/ready", headers={"X-Request-ID": "abc123"})
    assert r.json()["status"] == "ready" and r.headers["X-Request-ID"] == "abc123"


def test_evaluate_persists_and_is_readable(client):
    r = client.post("/v1/disputes/evaluate", json=body())
    assert r.status_code == 201
    out = r.json()
    assert out["action"] in {"CONTEST", "CONCEDE", "ESCALATE"} and out["model_version"].startswith("wm-")
    d = client.get("/v1/disputes/case_1").json()
    assert d["latest"]["evaluation_id"] == out["evaluation_id"] and len(d["history"]) == 1


def test_reevaluation_appends_history(client):
    client.post("/v1/disputes/evaluate", json=body())
    client.post("/v1/disputes/evaluate", json=body(amount_inr=7000))
    assert len(client.get("/v1/disputes/case_1").json()["history"]) == 2


def test_idempotency_key_replays_and_rejects_reuse(client):
    h = {"Idempotency-Key": "key-1"}
    first = client.post("/v1/disputes/evaluate", json=body(), headers=h)
    again = client.post("/v1/disputes/evaluate", json=body(), headers=h)
    assert first.status_code == 201 and again.status_code == 200
    assert again.json()["idempotent_replay"] and again.json()["evaluation_id"] == first.json()["evaluation_id"]
    assert client.post("/v1/disputes/evaluate", json=body(amount_inr=1234), headers=h).status_code == 409


@pytest.mark.parametrize("override", [
    {"merchant_id": "nope"}, {"reason_code": "99.9"}, {"amount_inr": -5}, {"response_deadline_days_left": -1},
    {"documents": [{"doc_type": "selfie", "text": "x"}]}, {"cardholder_name": ""},
])
def test_bad_input_is_422(client, override):
    r = client.post("/v1/disputes/evaluate", json={**body(), **override})
    assert r.status_code == 422


def test_same_case_id_for_a_different_merchant_conflicts(client):
    client.post("/v1/disputes/evaluate", json=body())
    assert client.post("/v1/disputes/evaluate", json=body(merchant_id="mch_01")).status_code == 409


def test_unknown_dispute_is_404(client):
    r = client.get("/v1/disputes/missing")
    assert r.status_code == 404 and "request_id" in r.json()


def test_escalation_queue_is_ranked_by_review_value_and_review_closes_it(client):
    for cid, amount in [("small", 20_000), ("big", 120_000)]:
        assert client.post("/v1/disputes/evaluate", json=escalating_body(cid, amount)).json()["action"] == "ESCALATE"
    queue = client.get("/v1/escalations").json()
    assert [i["case_id"] for i in queue["items"]] == ["big", "small"] and queue["total"] == 2

    r = client.post("/v1/disputes/big/review", json={"reviewer": "asha", "action": "CONTEST"})
    assert r.status_code == 200
    assert [i["case_id"] for i in client.get("/v1/escalations").json()["items"]] == ["small"]
    assert client.post("/v1/disputes/big/review", json={"reviewer": "x", "action": "CONCEDE"}).status_code == 409


def test_review_of_a_non_escalated_case_conflicts(client):
    r = client.post("/v1/disputes/evaluate", json=body(amount_inr=300))  # uneconomic -> CONCEDE
    assert r.json()["action"] == "CONCEDE"
    assert client.post("/v1/disputes/case_1/review", json={"reviewer": "a", "action": "CONTEST"}).status_code == 409


def test_queue_pagination(client):
    for i in range(3):
        client.post("/v1/disputes/evaluate", json=escalating_body(f"c{i}", 50_000 + i * 10_000))
    page = client.get("/v1/escalations?limit=2&offset=2").json()
    assert page["total"] == 3 and len(page["items"]) == 1 and page["items"][0]["rank"] == 3


def test_outcomes_feed_reports_and_calibration(client):
    client.post("/v1/disputes/evaluate", json=body(case_id="a"))
    client.post("/v1/disputes/evaluate", json=body(case_id="b", documents=[]))
    assert client.post("/v1/disputes/a/outcome", json={"outcome": "won"}).status_code == 200
    assert client.post("/v1/disputes/b/outcome", json={"outcome": "lost"}).status_code == 200
    assert client.post("/v1/disputes/b/outcome", json={"outcome": "won"}).status_code == 409
    assert client.post("/v1/disputes/zzz/outcome", json={"outcome": "won"}).status_code == 404

    report = client.get("/v1/reports/portfolio").json()
    assert report["root_causes"]["resolved"] == 2 and report["root_causes"]["lost"] == 1
    assert "delivery_confirmed" in dict(report["root_causes"]["top_missing"])
    assert client.get("/v1/monitoring/calibration").json()["n"] == 2
    assert client.get("/v1/reports/merchants/mch_05").json()["scope"] == "mch_05"
    assert client.get("/v1/reports/merchants/nope").status_code == 422


def test_write_endpoints_require_api_key_when_configured(client_factory):
    with client_factory(api_key="s3cret") as c:
        assert c.post("/v1/disputes/evaluate", json=body()).status_code == 401
        assert c.post("/v1/disputes/evaluate", json=body(), headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.post("/v1/disputes/evaluate", json=body(), headers={"X-API-Key": "s3cret"}).status_code == 201
        assert c.get("/v1/disputes/case_1").status_code == 200  # reads stay open


def test_metrics_endpoint_exposes_decisions(client):
    client.post("/v1/disputes/evaluate", json=body())
    text = client.get("/metrics").text
    assert "abstain_decisions_total" in text and "abstain_http_request_seconds" in text


def test_demo_seed_loads_once_and_feeds_insights(tmp_path):
    from app.catalog import load_catalog
    from app.db import Base, build_engine, build_session_factory
    from app.demo import seed_demo_portfolio
    from app.engine.decision_engine import DecisionEngine
    from app.scoring.win_model import WinModel

    url = f"sqlite:///{tmp_path / 'demo.db'}"
    engine = build_engine(url)
    Base.metadata.create_all(engine)
    sessions = build_session_factory(engine)
    args = (sessions, load_catalog(), WinModel.load(), DecisionEngine())
    assert seed_demo_portfolio(*args, limit=40) == 40
    assert seed_demo_portfolio(*args, limit=40) == 0          # never loads twice

    settings = dataclasses.replace(Settings(), database_url=url, llm_api_key=None, api_key=None)
    with TestClient(create_app(settings)) as c:
        assert c.get("/v1/reports/portfolio").json()["root_causes"]["resolved"] == 40
        assert c.get("/ready").json()["demo_data"] is False
