"""Contradictions within the merchant's own evidence, restricted to the facts
that matter for the dispute's category."""
from __future__ import annotations

from app.domain import EvidenceFacts, Tri

# (fact_a, value_a, fact_b, value_b, description)
_RULES = [
    ("delivery_confirmed", Tri.YES, "signed_by_cardholder", Tri.NO,
     "carrier confirms delivery but the signature is not the cardholder's"),
    ("delivery_confirmed", Tri.NO, "customer_acknowledged_receipt", Tri.YES,
     "carrier says not delivered but the customer says it arrived"),
    ("avs_cvv_match", Tri.YES, "ip_consistent_with_cardholder", Tri.NO,
     "AVS/CVV matched but the order came from an inconsistent location"),
]


def find_conflicts(facts: EvidenceFacts, relevant_facts: list[str]) -> list[str]:
    relevant = set(relevant_facts)
    return [
        desc for a, va, b, vb, desc in _RULES
        if a in relevant and b in relevant and facts.get(a) == va and facts.get(b) == vb
    ]
