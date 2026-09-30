from __future__ import annotations

import pytest

from app.domain import DocType, EvidenceDocument, EvidenceFacts, Tri
from app.extraction.aggregate import agreement, majority
from app.extraction.llm_extractor import build_messages, parse_and_ground
from app.extraction.rules import _names_match, extract_with_rules
from tests.conftest import CARDHOLDER, DEV_DOCS, llm_payload, make_case


def doc(t: DocType, text: str) -> EvidenceDocument:
    return EvidenceDocument(doc_type=t, text=text)


# ---------------------------------------------------------------- rules

@pytest.mark.parametrize("text,expected", [
    ("Status: DELIVERED on 12-Mar 14:05.", Tri.YES),
    ("Shipment could not be delivered: address not found. RTO initiated.", Tri.NO),
    ("Delivery failed after 3 attempts; parcel returned to seller.", Tri.NO),
    ("Label created.", Tri.UNKNOWN),
])
def test_rules_delivery_handles_negation(text, expected):
    facts = extract_with_rules([doc(DocType.CARRIER_RECORD, text)], CARDHOLDER)
    assert facts.delivery_confirmed == expected


@pytest.mark.parametrize("signed,expected", [
    ("Priya Nair", True), ("P. Nair", True), ("Priya N.", True), ("PRIYA NAIR", True),
    ("Priya Nayak", False), ("Rahul Nair", False), ("R. Menon", False),
])
def test_signature_name_matching(signed, expected):
    assert _names_match(signed, CARDHOLDER) is expected


def test_avs_requires_both_checks():
    order = "Order #1\nAddress verification: full match; security code: not provided"
    assert extract_with_rules([doc(DocType.ORDER_RECORD, order)], CARDHOLDER).avs_cvv_match == Tri.NO


def test_amount_comparison():
    ok = extract_with_rules([doc(DocType.INVOICE, "Invoice total Rs 4,500; card charged Rs 4,500.")], CARDHOLDER)
    bad = extract_with_rules([doc(DocType.INVOICE, "Invoice total Rs 3,200; card charged Rs 4,500.")], CARDHOLDER)
    assert ok.amount_matches_agreement == Tri.YES and bad.amount_matches_agreement == Tri.NO


def test_rules_read_dev_phrasing():
    facts = extract_with_rules(DEV_DOCS, CARDHOLDER)
    assert (facts.delivery_confirmed, facts.signed_by_cardholder, facts.customer_acknowledged_receipt) == \
        (Tri.YES, Tri.YES, Tri.YES)


# ---------------------------------------------------------------- LLM output parsing + grounding

def test_grounded_answers_are_kept():
    raw = llm_payload(delivery_confirmed=("yes", "Status: DELIVERED on 12-Mar"),
                      signed_by_cardholder=("yes", "signed by: p. nair"))
    facts, ungrounded = parse_and_ground(raw, make_case())
    assert facts.delivery_confirmed == Tri.YES and facts.signed_by_cardholder == Tri.YES and ungrounded == 0


def test_hallucinated_quote_is_downgraded_to_unknown():
    raw = llm_payload(delivery_confirmed=("yes", "Delivered and signed by the cardholder in person"))
    facts, ungrounded = parse_and_ground(raw, make_case())
    assert facts.delivery_confirmed == Tri.UNKNOWN and ungrounded == 1


def test_missing_facts_default_to_unknown():
    facts, _ = parse_and_ground(llm_payload(), make_case())
    assert facts == EvidenceFacts()


@pytest.mark.parametrize("raw", ["not json", '{"facts": {"delivery_confirmed": {"value": "maybe"}}}', "[]"])
def test_malformed_output_raises(raw):
    with pytest.raises(ValueError):
        parse_and_ground(raw, make_case())


def test_prompt_fences_documents_and_names_every_fact(catalog):
    messages = build_messages(make_case(), catalog.reason_code("13.1"))
    user = messages[1]["content"]
    assert "<documents>" in user and "untrusted" in messages[0]["content"]
    for name in EvidenceFacts.model_fields:
        assert name in user


# ---------------------------------------------------------------- self-consistency aggregation

def test_majority_and_agreement():
    y, n = EvidenceFacts(delivery_confirmed=Tri.YES), EvidenceFacts(delivery_confirmed=Tri.NO)
    assert majority([y, y, n]).delivery_confirmed == Tri.YES
    assert agreement([y, y, n])["delivery_confirmed"] == pytest.approx(0.667, abs=1e-3)


def test_majority_tie_resolves_to_unknown():
    y, n = EvidenceFacts(delivery_confirmed=Tri.YES), EvidenceFacts(delivery_confirmed=Tri.NO)
    assert majority([y, n]).delivery_confirmed == Tri.UNKNOWN
