"""Same dispute, different merchant -> the decision can legitimately differ.

The divergence comes from merchant economics (contest cost), track record
(a model feature) and risk appetite (how much human oversight the merchant
buys), not from a "chargeback-ratio penalty" (removed): winning a
representment does not remove a dispute from the network's ratio, so that
penalty priced a cost contesting does not have.
"""
from __future__ import annotations

from decimal import Decimal

from app.domain import Action, Category, EvidenceFacts, Tri
from app.scoring.win_model import WinEstimate
from tests.conftest import make_case


def test_track_record_moves_p_win(model, catalog):
    facts = [EvidenceFacts(delivery_confirmed=Tri.YES)]
    strong = model.estimate(Category.NOT_RECEIVED, facts, catalog.merchant("mch_07"))  # 78% historical win rate
    weak = model.estimate(Category.NOT_RECEIVED, facts, catalog.merchant("mch_06"))    # 29%
    assert strong.mean > weak.mean


def test_risk_appetite_changes_escalation_for_identical_estimates(engine, catalog):
    case = make_case(amount_inr=Decimal("4000"))
    # EV is near zero with a wide spread: the value of a review (~₹270) sits between the
    # conservative (₹150) and aggressive (₹600) review-cost thresholds.
    estimate = WinEstimate(mean=0.2, std=0.2, n_draws=1)
    base = catalog.merchant("mch_05")
    conservative = engine.decide(case, base.model_copy(update={"risk_tolerance": "conservative"}), estimate)
    aggressive = engine.decide(case, base.model_copy(update={"risk_tolerance": "aggressive"}), estimate)
    assert conservative.action == Action.ESCALATE and aggressive.action != Action.ESCALATE


def test_contest_cost_differs_by_merchant(engine, catalog):
    case = make_case()
    assert engine.contest_cost(case, catalog.merchant("mch_07")) > engine.contest_cost(case, catalog.merchant("mch_06"))
