"""
Deterministic decision policy. Imports no LLM or extraction code: it consumes
a WinEstimate (mean and std of P(win)) plus case economics, and nothing else.

The policy is expected-value maximization over three actions:

    X            = P(win) * amount - contest_cost        (value of contesting)
    V(CONCEDE)   = 0
    V(CONTEST)   = E[X]
    V(ESCALATE)  = E[max(X, 0)] - review_cost            (a reviewer resolves
                                                         the uncertainty, then
                                                         picks the better action)

With P(win) ~ Normal(mu, sigma), X ~ Normal(m, s) where m = mu*A - C and
s = sigma*A, and E[max(X, 0)] has a closed form. Escalating beats the best
automatic action exactly when the value of information

    VOI = E[max(X, 0)] - max(m, 0)

exceeds the review cost. So "abstain" is not a separate confidence threshold
bolted on: it is priced the same way as the other two actions. Two
assumptions are stated rather than hidden: the reviewer is treated as fully
resolving the uncertainty (VOI is an upper bound), and the Normal
approximation ignores that P(win) is bounded in [0, 1].

Two hard gates run before the EV comparison: operational feasibility and
economic contestability. Contradictory evidence is not a gate: it widens the
uncertainty and the value-of-information test decides (the eval showed a hard
"always escalate on conflict" rule cost more in review time than it saved).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from pydantic import BaseModel

from app.domain import Action, DisputeCase, Merchant
from app.scoring.win_model import WinEstimate


@dataclass(frozen=True)
class PolicyConfig:
    min_days_to_respond: int = 2
    repeat_dispute_cost_multiplier: float = 1.6     # pre-arbitration/arbitration fees
    human_review_cost_inr: float = 300.0            # fully loaded analyst cost per case
    # Scales the review cost per merchant risk appetite: conservative merchants
    # buy more human oversight, aggressive ones prefer automation.
    review_cost_multiplier: dict[str, float] = field(
        default_factory=lambda: {"aggressive": 2.0, "moderate": 1.0, "conservative": 0.5}
    )
    high_confidence_z: float = 2.0                  # |m| / s above this -> HIGH confidence
    # Contradictory evidence makes the point estimate less trustworthy, so it
    # widens the uncertainty; whether that is worth a human is still decided
    # by the value of information, not by a hard override.
    conflict_std_floor: float = 0.20


class Decision(BaseModel):
    action: Action
    confidence: str                    # HIGH / MEDIUM / LOW
    p_win: float
    p_win_std: float
    contest_cost_inr: float
    ev_contest_inr: float
    review_value_inr: float            # VOI: expected value of a human looking at this
    review_cost_inr: float
    conflicts: list[str] = []
    reasons: list[str]


def _phi(z: float) -> float:
    return math.exp(-0.5 * z * z) / math.sqrt(2 * math.pi)


def _Phi(z: float) -> float:
    return 0.5 * (1 + math.erf(z / math.sqrt(2)))


def expected_positive_part(m: float, s: float) -> float:
    """E[max(X, 0)] for X ~ Normal(m, s)."""
    if s <= 0:
        return max(m, 0.0)
    z = m / s
    return m * _Phi(z) + s * _phi(z)


def value_of_information(m: float, s: float) -> float:
    return max(0.0, expected_positive_part(m, s) - max(m, 0.0))


class DecisionEngine:
    def __init__(self, config: PolicyConfig | None = None):
        self.config = config or PolicyConfig()

    def contest_cost(self, case: DisputeCase, merchant: Merchant) -> float:
        cost = float(merchant.ops_cost_per_contest_inr)
        if case.is_repeat_dispute:
            cost *= self.config.repeat_dispute_cost_multiplier
        return cost

    def decide(
        self,
        case: DisputeCase,
        merchant: Merchant,
        estimate: WinEstimate,
        conflicts: list[str] | None = None,
    ) -> Decision:
        cfg = self.config
        amount = float(case.amount_inr)
        cost = self.contest_cost(case, merchant)
        reasons: list[str] = []
        mu, sigma = estimate.mean, estimate.std
        if conflicts:
            reasons.append("Contradictory evidence (" + "; ".join(conflicts) + f"): uncertainty widened to "
                           f"at least ±{cfg.conflict_std_floor:.2f}.")
            sigma = max(sigma, cfg.conflict_std_floor)
        m, s = mu * amount - cost, sigma * amount
        voi = value_of_information(m, s)
        review_cost = cfg.human_review_cost_inr * cfg.review_cost_multiplier[merchant.risk_tolerance]
        if case.is_repeat_dispute:
            reasons.append(f"Repeat/arbitration dispute: contest cost x{cfg.repeat_dispute_cost_multiplier} = ₹{cost:,.0f}.")

        def out(action: Action, confidence: str) -> Decision:
            return Decision(action=action, confidence=confidence, p_win=round(mu, 4), p_win_std=round(sigma, 4),
                            conflicts=conflicts or [],
                            contest_cost_inr=round(cost, 2), ev_contest_inr=round(m, 2),
                            review_value_inr=round(voi, 2), review_cost_inr=round(review_cost, 2),
                            reasons=reasons)

        # Gate 1: we cannot assemble and submit evidence in time.
        if case.response_deadline_days_left < cfg.min_days_to_respond:
            reasons.append(f"{case.response_deadline_days_left} day(s) left to respond, below the "
                           f"{cfg.min_days_to_respond}-day minimum. Feasibility overrides EV.")
            return out(Action.CONCEDE, "HIGH")

        # Gate 2: even a certain win would not cover the cost of contesting.
        if amount <= cost:
            reasons.append(f"Amount ₹{amount:,.0f} does not exceed contest cost ₹{cost:,.0f}; "
                           f"not contestable at any win probability.")
            return out(Action.CONCEDE, "HIGH")

        reasons.append(f"P(win) = {mu:.2f} ± {sigma:.2f}; EV(contest) = ₹{m:,.0f} "
                       f"(amount ₹{amount:,.0f}, cost ₹{cost:,.0f}).")
        if voi > review_cost:
            reasons.append(f"Value of a human review ₹{voi:,.0f} exceeds its cost ₹{review_cost:,.0f} "
                           f"({merchant.risk_tolerance} merchant): escalate.")
            return out(Action.ESCALATE, "LOW")

        action = Action.CONTEST if m > 0 else Action.CONCEDE
        confident = s == 0 or abs(m) / s >= cfg.high_confidence_z
        reasons.append(f"Review value ₹{voi:,.0f} ≤ review cost ₹{review_cost:,.0f}: decide automatically "
                       f"({'EV > 0, contest' if m > 0 else 'EV ≤ 0, concede'}).")
        return out(action, "HIGH" if confident else "MEDIUM")
