"""
portfolio.py

Computes the portfolio-context risk penalty that makes the SAME dispute
resolve differently for different merchants (the portfolio-aware decision
feature). Pulled out of decision_engine.py into its own module because this
specific calculation is the one the whole "same dispute, different merchant,
different decision" demo hinges on — it deserves to be independently
testable and independently explainable, not buried inside a longer method.
"""
from __future__ import annotations

from .models import Merchant

TOLERANCE_MULTIPLIERS = {"aggressive": 0.5, "moderate": 1.0, "conservative": 1.6}
PROXIMITY_FLOOR = 0.5      # penalty only starts accruing once a merchant is >50% of the way to their threshold
PENALTY_CAP_INR = 5000.0


def compute_portfolio_penalty(
    merchant: Merchant,
    cap_inr: float = PENALTY_CAP_INR,
    proximity_floor: float = PROXIMITY_FLOOR,
) -> tuple[float, str | None]:
    """Returns (penalty_inr, explanation_or_None). Raises on an unrecognized
    risk_tolerance rather than silently defaulting — a merchant record with
    a typo'd tolerance should fail loudly, not get an arbitrary multiplier.
    cap_inr/proximity_floor default to this module's constants but are
    overridable so callers (e.g. DecisionEngine's EngineConfig) stay the
    single place tuning actually happens, without duplicating the formula."""
    if merchant.risk_tolerance not in TOLERANCE_MULTIPLIERS:
        raise ValueError(
            f"Unrecognized risk_tolerance {merchant.risk_tolerance!r} on "
            f"merchant {merchant.merchant_id}; expected one of {list(TOLERANCE_MULTIPLIERS)}."
        )

    proximity = merchant.threshold_proximity  # 0..1.5, defined on Merchant itself
    multiplier = TOLERANCE_MULTIPLIERS[merchant.risk_tolerance]
    raw = max(0.0, proximity - proximity_floor) * multiplier
    penalty = min(cap_inr, raw * cap_inr)

    if penalty <= 0:
        return 0.0, None

    explanation = (
        f"Merchant at {proximity:.0%} of network chargeback-rate threshold "
        f"({merchant.risk_tolerance} tolerance) — portfolio risk penalty "
        f"₹{penalty:,.0f} added to the cost of contesting."
    )
    return round(penalty, 2), explanation