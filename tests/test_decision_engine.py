"""
test_decision_engine.py

Tests the deterministic core in isolation — constructs EvidenceAssessment
directly rather than going through EvidenceScorer, so these tests exercise
decision_engine.py's own logic only, not the LLM/fallback path (that's
evidence_scorer's own concern, and its fallback determinism is exercised
indirectly via test_portfolio_pairs.py and the eval harness).

Each test targets exactly one gate, in the order decide() actually checks
them, so a failure here points straight at which gate broke.
"""
from __future__ import annotations
import pytest

from app.engine.decision_engine import DecisionEngine, EngineConfig
from app.engine.models import Action, DisputeCase, EvidenceAssessment, Merchant


def make_merchant(**overrides) -> Merchant:
    defaults = dict(
        merchant_id="mch_test", historical_win_rate=0.5,
        current_chargeback_rate_pct=0.3, network_threshold_pct=0.9,
        ops_cost_per_contest_inr=500.0, risk_tolerance="moderate",
    )
    defaults.update(overrides)
    return Merchant(**defaults)


def make_case(**overrides) -> DisputeCase:
    defaults = dict(
        case_id="case_test", merchant_id="mch_test", reason_code="13.1",
        dispute_amount_inr=5000.0, response_deadline_days_left=10,
        is_repeat_dispute=False, conflicting_evidence=False,
    )
    defaults.update(overrides)
    return DisputeCase(**defaults)


def make_evidence(**overrides) -> EvidenceAssessment:
    defaults = dict(strength=0.8, uncertainty=0.10, source="llm", key_gaps=[], reasoning="")
    defaults.update(overrides)
    return EvidenceAssessment(**defaults)


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()


# ---------- Gate 1: feasibility ----------

def test_feasibility_gate_forces_concede_when_days_left_below_minimum(engine):
    case = make_case(response_deadline_days_left=1)
    decision = engine.decide(case, make_merchant(), make_evidence(strength=0.95, uncertainty=0.05))
    assert decision.action == Action.CONCEDE
    assert "day(s) left to respond" in decision.reasons[0]


def test_feasibility_gate_allows_processing_at_exact_minimum(engine):
    cfg = EngineConfig(min_days_to_respond=2)
    e = DecisionEngine(cfg)
    case = make_case(response_deadline_days_left=2)  # exactly at the minimum, should NOT trigger the gate
    decision = e.decide(case, make_merchant(), make_evidence(strength=0.95, uncertainty=0.05))
    assert "day(s) left to respond" not in decision.reasons[0]


# ---------- Gate 2: trivial amount ----------

def test_trivial_amount_forces_concede_regardless_of_evidence():
    merchant = make_merchant(ops_cost_per_contest_inr=1000.0)
    case = make_case(dispute_amount_inr=500.0)  # even a 100% win can't clear ops cost
    engine = DecisionEngine()
    decision = engine.decide(case, merchant, make_evidence(strength=1.0, uncertainty=0.01))
    assert decision.action == Action.CONCEDE
    assert "Not economically contestable" in decision.reasons[0]


# ---------- Gate 3: confidence ----------

def test_low_confidence_forces_escalate_even_with_strong_point_estimate(engine):
    case = make_case()
    decision = engine.decide(case, make_merchant(), make_evidence(strength=0.9, uncertainty=0.5))
    assert decision.action == Action.ESCALATE
    assert decision.confidence_label == "LOW"


def test_conflicting_evidence_forces_escalate_even_at_high_confidence(engine):
    case = make_case(conflicting_evidence=True)
    decision = engine.decide(case, make_merchant(), make_evidence(strength=0.9, uncertainty=0.05))
    assert decision.action == Action.ESCALATE
    assert "conflicting" in decision.reasons[-1].lower()


# ---------- Gate 4: EV-based decision ----------

def test_high_confidence_contests_when_ev_positive(engine):
    case = make_case(dispute_amount_inr=10000.0)
    merchant = make_merchant(ops_cost_per_contest_inr=300.0)
    decision = engine.decide(case, merchant, make_evidence(strength=0.85, uncertainty=0.05))
    assert decision.confidence_label == "HIGH"
    assert decision.action == Action.CONTEST


def test_high_confidence_concedes_when_ev_negative(engine):
    case = make_case(dispute_amount_inr=1000.0)
    merchant = make_merchant(ops_cost_per_contest_inr=800.0)
    decision = engine.decide(case, merchant, make_evidence(strength=0.3, uncertainty=0.05))
    assert decision.confidence_label == "HIGH"
    assert decision.action == Action.CONCEDE


def test_medium_confidence_escalates_when_ev_falls_inside_margin(engine):
    # Deliberately balanced so EV_contest lands close to zero, inside the
    # medium-confidence margin — should escalate rather than guess.
    case = make_case(dispute_amount_inr=1200.0)
    merchant = make_merchant(ops_cost_per_contest_inr=600.0)
    decision = engine.decide(case, merchant, make_evidence(strength=0.55, uncertainty=0.25))
    assert decision.confidence_label == "MEDIUM"
    assert decision.action == Action.ESCALATE


# ---------- Ops cost / repeat dispute ----------

def test_repeat_dispute_increases_ops_cost(engine):
    merchant = make_merchant(ops_cost_per_contest_inr=500.0)
    first = engine.decide(make_case(is_repeat_dispute=False), merchant, make_evidence())
    second = engine.decide(make_case(is_repeat_dispute=True), merchant, make_evidence())
    assert second.ops_cost_inr > first.ops_cost_inr
    assert second.ops_cost_inr == pytest.approx(500.0 * 1.6)


# ---------- Input validation ----------

def test_unrecognized_risk_tolerance_rejected_at_construction():
    with pytest.raises(ValueError):
        make_merchant(risk_tolerance="reckless")


def test_negative_amount_rejected_at_construction():
    with pytest.raises(ValueError):
        make_case(dispute_amount_inr=-100.0)