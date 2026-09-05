"""
decision_engine.py

The deterministic core of Abstain. Deliberately has ZERO dependency on any
LLM client — not even indirectly. It consumes only validated, structured
inputs (EvidenceAssessment, Merchant, DisputeCase) and returns a Decision.

This isolation is the architecture's central claim: the LLM can read,
retrieve, and draft — but it cannot decide where money goes. Enforcing that
as an import boundary, not just a design doc sentence, is the point.

Design choice on the EV model (deliberately simpler and more correct than
"EV_contest - EV_concede"): conceding never costs anything further — the
disputed amount is already gone the moment the chargeback lands. Contesting
is the only action with a marginal cost/gain, so:

    EV_contest = P(win) * amount - ops_cost - portfolio_risk_penalty
    EV_concede = 0                      (baseline: accept the existing loss)

Decision follows EV_contest relative to 0, gated by two things EV alone
doesn't capture: operational feasibility (can we even respond in time) and
confidence (is the estimate trustworthy enough to act on automatically).
"""
from __future__ import annotations
from dataclasses import dataclass

from .models import Action, Decision, DisputeCase, EvidenceAssessment, Merchant
from .portfolio import compute_portfolio_penalty


@dataclass(frozen=True)
class EngineConfig:
    """All thresholds live here — nothing magic-numbered inline below.
    Tune these against the eval harness, not by feel."""
    min_days_to_respond: int = 2          # below this, contesting isn't operationally feasible
    high_confidence_uncertainty_max: float = 0.15   # uncertainty below this = trust the point estimate
    medium_confidence_ev_margin_fraction: float = 0.35
    low_confidence_uncertainty_min: float = 0.35    # uncertainty above this = don't trust it, escalate
    trivial_amount_floor_multiplier: float = 1.0    # if amount < ops_cost * this, contesting can't pay off
    portfolio_penalty_max_inr: float = 5000.0       # cap on how much threshold-proximity can inflate cost
    repeat_dispute_fee_multiplier: float = 1.6      # arbitration/second-stage fees run higher


class DecisionEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()

    # ---------- public entrypoint ----------
    def decide(
        self,
        case: DisputeCase,
        merchant: Merchant,
        evidence: EvidenceAssessment,
    ) -> Decision:
        reasons: list[str] = []

        ops_cost = self._ops_cost(case, merchant, reasons)
        portfolio_penalty = self._portfolio_penalty(merchant, reasons)
        win_prob = self._win_probability(evidence, merchant)

        ev_contest = win_prob * case.dispute_amount_inr - ops_cost - portfolio_penalty

        # --- Gate 1: operational feasibility overrides everything else ---
        if case.response_deadline_days_left < self.config.min_days_to_respond:
            reasons.append(
                f"Only {case.response_deadline_days_left} day(s) left to respond — "
                f"below the {self.config.min_days_to_respond}-day minimum needed to "
                f"assemble and submit evidence. Feasibility overrides EV."
            )
            return self._finalize(case, Action.CONCEDE, win_prob, evidence.uncertainty,
                                   ev_contest, ops_cost, portfolio_penalty, reasons)

        # --- Gate 2: trivial amount — EV can't clear ops cost even at max plausible win rate ---
        best_case_ev = 1.0 * case.dispute_amount_inr - ops_cost - portfolio_penalty
        if best_case_ev <= 0:
            reasons.append(
                f"Even at a 100% win probability, expected recovery (₹{case.dispute_amount_inr:,.0f}) "
                f"cannot exceed ops cost + portfolio penalty (₹{ops_cost + portfolio_penalty:,.0f}). "
                f"Not economically contestable regardless of evidence."
            )
            return self._finalize(case, Action.CONCEDE, win_prob, evidence.uncertainty,
                                   ev_contest, ops_cost, portfolio_penalty, reasons)

        # --- Gate 3: confidence — don't act automatically on an untrustworthy estimate ---
        confidence_label = self._confidence_label(evidence.uncertainty)
        if confidence_label == "LOW":
            reasons.append(
                f"Uncertainty ({evidence.uncertainty:.2f}) is too high to trust the point "
                f"estimate (P(win)={win_prob:.2f}) for an automatic decision, regardless of EV sign."
            )
            return self._finalize(case, Action.ESCALATE, win_prob, evidence.uncertainty,
                                   ev_contest, ops_cost, portfolio_penalty, reasons,
                                   confidence_label=confidence_label)

        if case.conflicting_evidence:
            reasons.append(
                "Evidence signals point in different directions (conflicting_evidence flag set) — "
                "escalating regardless of EV, since an automated evidence draft could misrepresent the case."
            )
            return self._finalize(case, Action.ESCALATE, win_prob, evidence.uncertainty,
                                   ev_contest, ops_cost, portfolio_penalty, reasons,
                                   confidence_label=confidence_label)

        # --- Gate 4: EV-based decision with confidence-scaled zone for MEDIUM confidence ---
        if confidence_label == "HIGH":
            action = Action.CONTEST if ev_contest > 0 else Action.CONCEDE
            reasons.append(
                f"High confidence (uncertainty={evidence.uncertainty:.2f}); "
                f"EV_contest={ev_contest:,.0f} INR {'> 0, contest.' if ev_contest > 0 else '<= 0, concede.'}"
            )
        else:  # MEDIUM confidence — require a wider EV margin before auto-acting
            margin = self.config.medium_confidence_ev_margin_fraction * (
    ops_cost + portfolio_penalty
)
            if ev_contest > margin:
                action = Action.CONTEST
                reasons.append(
                    f"Medium confidence; EV_contest={ev_contest:,.0f} clears the "
                    f"required margin (₹{margin:,.0f}) for an automatic contest."
                )
            elif ev_contest < -margin:
                action = Action.CONCEDE
                reasons.append(
                    f"Medium confidence; EV_contest={ev_contest:,.0f} is clearly negative "
                    f"beyond the margin — concede."
                )
            else:
                action = Action.ESCALATE
                reasons.append(
                    f"Medium confidence and EV_contest={ev_contest:,.0f} falls inside the "
                    f"±₹{margin:,.0f} margin — too close to call automatically."
                )

        return self._finalize(case, action, win_prob, evidence.uncertainty,
                               ev_contest, ops_cost, portfolio_penalty, reasons,
                               confidence_label=confidence_label)

    # ---------- helpers ----------
    def _ops_cost(self, case: DisputeCase, merchant: Merchant, reasons: list[str]) -> float:
        cost = merchant.ops_cost_per_contest_inr
        if case.is_repeat_dispute:
            cost *= self.config.repeat_dispute_fee_multiplier
            reasons.append(
                f"Repeat/arbitration dispute — ops cost multiplied "
                f"{self.config.repeat_dispute_fee_multiplier}x to ₹{cost:,.0f}."
            )
        return cost

    def _portfolio_penalty(self, merchant: Merchant, reasons: list[str]) -> float:
        """Same dispute, different merchant context: this is where that divergence
        actually gets computed — delegated to portfolio.py so the calculation has
        one home and is independently testable outside the engine."""
        penalty, explanation = compute_portfolio_penalty(
            merchant, cap_inr=self.config.portfolio_penalty_max_inr
        )
        if explanation:
            reasons.append(explanation)
        return penalty

    def _win_probability(self, evidence: EvidenceAssessment, merchant: Merchant) -> float:
        """Blend the evidence-driven estimate with the merchant's own historical
        win rate as a light prior, so a brand-new/thin-evidence case doesn't get
        scored in a vacuum. 70/30 weighting toward the case-specific signal."""
        blended = 0.7 * evidence.strength + 0.3 * merchant.historical_win_rate
        return round(max(0.0, min(1.0, blended)), 3)

    def _confidence_label(self, uncertainty: float) -> str:
        if uncertainty <= self.config.high_confidence_uncertainty_max:
            return "HIGH"
        if uncertainty >= self.config.low_confidence_uncertainty_min:
            return "LOW"
        return "MEDIUM"

    def _finalize(
        self, case, action, win_prob, uncertainty, ev_contest, ops_cost,
        portfolio_penalty, reasons, confidence_label: str | None = None,
    ) -> Decision:
        label = confidence_label or self._confidence_label(uncertainty)
        memo = self._write_memo(case, action, win_prob, ev_contest, ops_cost,
                                 portfolio_penalty, label, reasons)
        return Decision(
            case_id=case.case_id,
            action=action,
            win_probability=win_prob,
            uncertainty=uncertainty,
            ev_contest_inr=round(ev_contest, 2),
            ops_cost_inr=round(ops_cost, 2),
            portfolio_risk_penalty_inr=round(portfolio_penalty, 2),
            reasons=reasons,
            confidence_label=label,
            memo=memo,
        )

    @staticmethod
    def _write_memo(case, action, win_prob, ev_contest, ops_cost, portfolio_penalty,
                     confidence_label, reasons) -> str:
        lines = [f"RECOMMENDATION: {action.value}", "", "Why:"]
        lines += [f"  • {r}" for r in reasons]
        lines += [
            "",
            f"P(win): {win_prob:.0%}   Confidence: {confidence_label}",
            f"EV(contest): ₹{ev_contest:,.0f}  "
            f"(ops cost ₹{ops_cost:,.0f} + portfolio penalty ₹{portfolio_penalty:,.0f})",
        ]
        return "\n".join(lines)

    # ---------- counterfactual reuse ----------
    def counterfactual(
        self,
        case: DisputeCase,
        merchant: Merchant,
        evidence: EvidenceAssessment,
        hypothetical_strength: float,
        hypothetical_uncertainty: float | None = None,
    ) -> Decision:
        """Re-run the same deterministic logic with one input perturbed.
        This is deliberately just a call to `decide()` with a modified
        EvidenceAssessment — no separate code path to keep in sync."""
        hypothetical = evidence.model_copy(update={
            "strength": hypothetical_strength,
            "uncertainty": hypothetical_uncertainty if hypothetical_uncertainty is not None else evidence.uncertainty,
        })
        return self.decide(case, merchant, hypothetical)