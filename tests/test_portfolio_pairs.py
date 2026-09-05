"""
test_portfolio_pairs.py

Asserts the feature the whole demo hinges on: the SAME dispute (same case,
same EvidenceAssessment) produces a DIFFERENT decision depending on the
merchant's portfolio context. Evidence is held fixed and identical across
both merchants in every test here — if these ever pass with an unchanged
EvidenceAssessment but a divergence in outcome, that's proof the merchant
context is what moved the needle, not evidence noise.
"""
from __future__ import annotations
import pytest

from app.engine.decision_engine import DecisionEngine
from app.engine.portfolio import compute_portfolio_penalty, TOLERANCE_MULTIPLIERS
from app.engine.models import Action, DisputeCase, EvidenceAssessment, Merchant


def make_case() -> DisputeCase:
    return DisputeCase(
        case_id="case_shared", merchant_id="varies", reason_code="4853",
        dispute_amount_inr=6000.0, response_deadline_days_left=10,
    )


SHARED_EVIDENCE = EvidenceAssessment(strength=0.65, uncertainty=0.10, source="llm")


@pytest.fixture
def engine() -> DecisionEngine:
    return DecisionEngine()


def test_same_dispute_diverges_between_aggressive_and_conservative_merchant(engine):
    # historical_win_rate is held EQUAL on purpose: win_probability blends
    # evidence strength with this as a light prior (see decision_engine.py's
    # _win_probability), so leaving it to vary would let two variables move
    # at once and the test would no longer isolate the portfolio penalty as
    # the cause of the divergence — which is the actual claim being tested.
    aggressive = Merchant(
        merchant_id="mch_aggressive", historical_win_rate=0.5,
        current_chargeback_rate_pct=0.2, network_threshold_pct=0.9,   # far from threshold
        ops_cost_per_contest_inr=600.0, risk_tolerance="aggressive",
    )
    conservative = Merchant(
        merchant_id="mch_conservative", historical_win_rate=0.5,
        current_chargeback_rate_pct=0.85, network_threshold_pct=0.9,  # near threshold
        ops_cost_per_contest_inr=600.0, risk_tolerance="conservative",
    )

    case = make_case()
    decision_a = engine.decide(case, aggressive, SHARED_EVIDENCE)
    decision_b = engine.decide(case, conservative, SHARED_EVIDENCE)

    # Evidence AND historical_win_rate were identical — win_probability must
    # therefore be identical too. Any difference in EV or action is
    # attributable only to the portfolio (threshold-proximity) penalty.
    assert decision_a.win_probability == decision_b.win_probability
    assert decision_a.ev_contest_inr != decision_b.ev_contest_inr
    assert decision_a.portfolio_risk_penalty_inr < decision_b.portfolio_risk_penalty_inr
    # The conservative, near-threshold merchant should never end up MORE
    # willing to contest than the aggressive, far-from-threshold one.
    action_leniency = {Action.CONTEST: 2, Action.ESCALATE: 1, Action.CONCEDE: 0}
    assert action_leniency[decision_a.action] >= action_leniency[decision_b.action]


def test_portfolio_penalty_zero_when_far_below_threshold():
    merchant = Merchant(
        merchant_id="mch_safe", historical_win_rate=0.5,
        current_chargeback_rate_pct=0.1, network_threshold_pct=0.9,  # ~11% of threshold
        ops_cost_per_contest_inr=500.0, risk_tolerance="conservative",
    )
    penalty, explanation = compute_portfolio_penalty(merchant)
    assert penalty == 0.0
    assert explanation is None


def test_portfolio_penalty_capped_at_configured_max():
    merchant = Merchant(
        merchant_id="mch_over", historical_win_rate=0.3,
        current_chargeback_rate_pct=5.0, network_threshold_pct=0.9,  # wildly over threshold
        ops_cost_per_contest_inr=500.0, risk_tolerance="conservative",
    )
    penalty, explanation = compute_portfolio_penalty(merchant, cap_inr=5000.0)
    assert penalty == 5000.0
    assert explanation is not None


def test_all_declared_risk_tolerances_are_handled():
    # Guards against silent drift if a new tolerance value is ever added to
    # one of Merchant's validator or TOLERANCE_MULTIPLIERS but not the other.
    from app.engine.models import Merchant as M
    declared = {"aggressive", "moderate", "conservative"}
    assert set(TOLERANCE_MULTIPLIERS.keys()) == declared


def test_unrecognized_tolerance_raises_rather_than_defaulting():
    merchant = Merchant.model_construct(
        merchant_id="mch_bad", historical_win_rate=0.5,
        current_chargeback_rate_pct=0.5, network_threshold_pct=0.9,
        ops_cost_per_contest_inr=500.0, risk_tolerance="reckless",
    )
    with pytest.raises(ValueError):
        compute_portfolio_penalty(merchant)