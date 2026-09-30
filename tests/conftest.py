from __future__ import annotations

import json
from decimal import Decimal

import httpx
import pytest

from app.catalog import load_catalog
from app.domain import DisputeCase, DocType, EvidenceDocument
from app.engine.decision_engine import DecisionEngine
from app.scoring.win_model import WinModel

CARDHOLDER = "Priya Nair"

# Phrasing the rules were written against ...
DEV_DOCS = [
    EvidenceDocument(doc_type=DocType.CARRIER_RECORD,
                     text="Tracking ID: DL123456789\nStatus: DELIVERED on 12-Mar 14:05.\nSigned by: P. Nair"),
    EvidenceDocument(doc_type=DocType.CUSTOMER_MESSAGES,
                     text="From customer (13-Mar): Received the box today. I'm still raising a dispute with my bank."),
]
# ... and phrasing they have never seen.
SHIFTED_DOCS = [
    EvidenceDocument(doc_type=DocType.CARRIER_RECORD,
                     text="Tracking ID: DL123456789\nConsignment status: handed to the addressee.\n"
                          "POD signature: Priya Nair"),
    EvidenceDocument(doc_type=DocType.CUSTOMER_MESSAGES,
                     text="From customer (13-Mar): Collected it from the front desk on Tuesday."),
]


def make_case(**overrides) -> DisputeCase:
    base = dict(case_id="case_1", merchant_id="mch_05", reason_code="13.1", amount_inr=Decimal("5000"),
                cardholder_name=CARDHOLDER, response_deadline_days_left=10, is_repeat_dispute=False,
                documents=DEV_DOCS)
    base.update(overrides)
    return DisputeCase(**base)


@pytest.fixture(scope="session")
def catalog():
    return load_catalog()


@pytest.fixture(scope="session")
def model():
    return WinModel.load()


@pytest.fixture
def engine():
    return DecisionEngine()


def llm_payload(**values) -> str:
    """A well-formed extraction response; values are (value, quote) pairs."""
    facts = {name: {"value": v, "quote": q} for name, (v, q) in values.items()}
    return json.dumps({"facts": facts})


def chat_response(content: str, status: int = 200, headers: dict | None = None) -> httpx.Response:
    body = {"choices": [{"message": {"content": content}}], "usage": {"prompt_tokens": 100, "completion_tokens": 50}}
    return httpx.Response(status, json=body if status == 200 else {"error": content}, headers=headers or {})
