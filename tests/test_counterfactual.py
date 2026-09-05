"""
test_counterfactual.py

Uses a small FAKE scorer, not the real EvidenceScorer's fallback heuristic,
deliberately. The fallback heuristic pins uncertainty at a fixed value that
sits at the LOW-confidence boundary (see evidence_scorer.py), which means
it never trusts itself enough to auto-CONTEST — so testing counterfactual
flips against it would never actually observe a flip, and the test would
be exercising the fallback's conservatism rather than find_minimal_flip's
own logic. This fake gives find_minimal_flip a scorer whose confidence
band is fully controllable, so the flip-detection mechanics can actually
be exercised end to end.
"""
from __future__ import annotations
import pytest

from app.engine.counterfactual import find_minimal_flip
from app.engine.decision_engine import DecisionEngine
from app.engine.models import Action, DisputeCase, EvidenceAssessment, Merchant


class FakeScorer:
    """assess() returns HIGH-confidence strong evidence once a specific
    required item is present, LOW-confidence weak evidence otherwise —
    lets a test set up an exact, deterministic flip condition."""
    def __init__(self, flips_on: str):
        self.flips_on = flips_on

    def assess(self, *, reason_code, reason_label, required_evidence, present_evidence, missing_evidence):
        if self.flips_on in present_evidence:
            return EvidenceAssessment(strength=0.9, uncertainty=0.05, source="llm")
        return EvidenceAssessment(strength=0.3, uncertainty=0.5, source="llm")


def make_merchant() -> Merchant:
    return Merchant(
        merchant_id="mch_test", historical_win_rate=0.5,
        current_chargeback_rate_pct=0.3, network_threshold_pct=0.9,
        ops_cost_per_contest_inr=300.0, risk_tolerance="moderate",
    )


def make_case() -> DisputeCase:
    return DisputeCase(
        case_id="case_cf", merchant_id="mch_test", reason_code="13.1",
        dispute_amount_inr=8000.0, response_deadline_days_left=10,
    )


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()


def test_no_missing_evidence_reports_no_flip_possible(engine):
    scorer = FakeScorer(flips_on="delivery_proof")
    case, merchant = make_case(), make_merchant()
    baseline_assessment = scorer.assess(
        reason_code="13.1", reason_label="test", required_evidence=["delivery_proof"],
        present_evidence=["delivery_proof"], missing_evidence=[],
    )
    baseline = engine.decide(case, merchant, baseline_assessment)

    result = find_minimal_flip(
        engine=engine, scorer=scorer, case=case, merchant=merchant,
        reason_code="13.1", reason_label="test", required_evidence=["delivery_proof"],
        present_evidence=["delivery_proof"], missing_evidence=[], baseline_decision=baseline,
    )
    assert result.flips is False
    assert result.checked_items == []
    assert "No missing evidence" in result.note


def test_single_missing_item_flip_is_detected(engine):
    scorer = FakeScorer(flips_on="delivery_proof")
    case, merchant = make_case(), make_merchant()
    required = ["delivery_proof", "tracking_number"]
    present = ["tracking_number"]
    missing = ["delivery_proof"]

    baseline_assessment = scorer.assess(
        reason_code="13.1", reason_label="test", required_evidence=required,
        present_evidence=present, missing_evidence=missing,
    )
    baseline = engine.decide(case, merchant, baseline_assessment)
    assert baseline.action != Action.CONTEST  # sanity check on the fixture itself

    result = find_minimal_flip(
        engine=engine, scorer=scorer, case=case, merchant=merchant,
        reason_code="13.1", reason_label="test", required_evidence=required,
        present_evidence=present, missing_evidence=missing, baseline_decision=baseline,
    )
    assert result.flips is True
    assert result.flipping_evidence == "delivery_proof"
    assert result.resulting_action == Action.CONTEST
    assert result.ev_delta_inr > 0


def test_no_single_item_flips_a_robust_decision(engine):
    # flips_on references an item that's never in missing_evidence, so no
    # candidate the loop tries can ever satisfy the fake's flip condition.
    scorer = FakeScorer(flips_on="some_evidence_never_checked")
    case, merchant = make_case(), make_merchant()
    required = ["delivery_proof", "tracking_number"]
    present: list[str] = []
    missing = ["delivery_proof", "tracking_number"]

    baseline_assessment = scorer.assess(
        reason_code="13.1", reason_label="test", required_evidence=required,
        present_evidence=present, missing_evidence=missing,
    )
    baseline = engine.decide(case, merchant, baseline_assessment)

    result = find_minimal_flip(
        engine=engine, scorer=scorer, case=case, merchant=merchant,
        reason_code="13.1", reason_label="test", required_evidence=required,
        present_evidence=present, missing_evidence=missing, baseline_decision=baseline,
    )
    assert result.flips is False
    assert set(result.checked_items) == set(missing)
    assert "would need multiple evidence items together" in result.note