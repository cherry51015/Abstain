from __future__ import annotations

from decimal import Decimal

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from app.domain import Action
from app.engine.decision_engine import DecisionEngine, PolicyConfig, expected_positive_part, value_of_information
from app.scoring.win_model import WinEstimate
from tests.conftest import make_case


def est(mean: float, std: float) -> WinEstimate:
    return WinEstimate(mean=mean, std=std, n_draws=1)


def merchant(catalog, tolerance: str = "moderate"):
    return catalog.merchant("mch_05").model_copy(update={"risk_tolerance": tolerance})


# ---------------------------------------------------------------- VOI math

@pytest.mark.parametrize("m,s", [(0.0, 1.0), (500.0, 800.0), (-1200.0, 900.0), (3000.0, 100.0)])
def test_expected_positive_part_matches_monte_carlo(m, s):
    x = np.random.default_rng(0).normal(m, s, 400_000)
    assert expected_positive_part(m, s) == pytest.approx(np.maximum(x, 0).mean(), rel=0.02, abs=2.0)


def test_voi_is_zero_without_uncertainty():
    assert value_of_information(1000.0, 0.0) == 0.0
    assert value_of_information(-1000.0, 0.0) == 0.0


@given(m=st.floats(-1e5, 1e5), s=st.floats(0, 1e5))
def test_voi_is_non_negative(m, s):
    assert value_of_information(m, s) >= 0.0


@given(m=st.floats(-1e4, 1e4), s1=st.floats(1, 1e4), s2=st.floats(1, 1e4))
def test_voi_grows_with_uncertainty(m, s1, s2):
    lo, hi = sorted([s1, s2])
    assert value_of_information(m, lo) <= value_of_information(m, hi) + 1e-6


# ---------------------------------------------------------------- gates

def test_feasibility_gate_concedes_even_a_certain_win(engine, catalog):
    d = engine.decide(make_case(response_deadline_days_left=1), merchant(catalog), est(0.99, 0.0))
    assert d.action == Action.CONCEDE and "Feasibility" in d.reasons[-1]


def test_uneconomic_amount_is_conceded_at_any_probability(engine, catalog):
    d = engine.decide(make_case(amount_inr=Decimal("500")), merchant(catalog), est(1.0, 0.0))
    assert d.action == Action.CONCEDE and "not contestable" in d.reasons[-1]


def test_repeat_dispute_raises_contest_cost(engine, catalog):
    base = engine.decide(make_case(), merchant(catalog), est(0.6, 0.02))
    repeat = engine.decide(make_case(is_repeat_dispute=True), merchant(catalog), est(0.6, 0.02))
    assert repeat.contest_cost_inr == pytest.approx(base.contest_cost_inr * 1.6)


# ---------------------------------------------------------------- EV policy

def test_confident_positive_ev_contests(engine, catalog):
    d = engine.decide(make_case(), merchant(catalog), est(0.8, 0.02))
    assert d.action == Action.CONTEST and d.confidence == "HIGH"


def test_confident_negative_ev_concedes(engine, catalog):
    d = engine.decide(make_case(), merchant(catalog), est(0.05, 0.02))
    assert d.action == Action.CONCEDE


def test_uncertain_high_stakes_case_escalates(engine, catalog):
    d = engine.decide(make_case(amount_inr=Decimal("90000")), merchant(catalog), est(0.02, 0.25))
    assert d.action == Action.ESCALATE
    assert d.review_value_inr > d.review_cost_inr


def test_escalates_exactly_when_voi_exceeds_review_cost(engine, catalog):
    for std in np.linspace(0.0, 0.4, 41):
        d = engine.decide(make_case(), merchant(catalog), est(0.2, float(std)))
        assert (d.action == Action.ESCALATE) == (d.review_value_inr > d.review_cost_inr)


def test_conservative_merchants_escalate_at_least_as_often(engine, catalog):
    case = make_case(amount_inr=Decimal("6000"))
    for std in np.linspace(0.0, 0.4, 21):
        cons = engine.decide(case, merchant(catalog, "conservative"), est(0.2, float(std)))
        aggr = engine.decide(case, merchant(catalog, "aggressive"), est(0.2, float(std)))
        if aggr.action == Action.ESCALATE:
            assert cons.action == Action.ESCALATE


def test_conflicts_widen_uncertainty_instead_of_forcing_escalation(engine, catalog):
    cheap = engine.decide(make_case(amount_inr=Decimal("1500")), merchant(catalog), est(0.9, 0.01),
                          conflicts=["signature mismatch"])
    assert cheap.p_win_std == PolicyConfig().conflict_std_floor
    assert cheap.action != Action.ESCALATE  # not worth a human at this amount
    big = engine.decide(make_case(amount_inr=Decimal("60000")), merchant(catalog), est(0.3, 0.01),
                        conflicts=["signature mismatch"])
    assert big.action == Action.ESCALATE


@settings(max_examples=200)
@given(mu=st.floats(0, 1), delta=st.floats(0, 1), std=st.floats(0, 0.5),
       amount=st.integers(200, 200_000))
def test_higher_win_probability_never_turns_contest_into_concede(mu, delta, std, amount):
    from app.catalog import load_catalog
    engine, m = DecisionEngine(), load_catalog().merchant("mch_05")
    case = make_case(amount_inr=Decimal(amount))
    lo = engine.decide(case, m, est(mu, std))
    hi = engine.decide(case, m, est(min(1.0, mu + delta), std))
    if lo.action == Action.CONTEST:
        assert hi.action != Action.CONCEDE
