"""
counterfactual.py

Answers: "what's the one piece of missing evidence that would actually
change this decision?" — not by asking an LLM to guess, but by re-running
the real pipeline (evidence_scorer + decision_engine) once per candidate
missing-evidence item and checking whether the action flips.

Deliberate scope limit, stated honestly rather than hidden: this checks
single-item additions only, not combinations. Checking all combinations
of missing evidence is a real possibility (2^n candidates) but grows fast
and mostly restates the same insight — the single most load-bearing gap
is what a merchant actually acts on. If nothing single-handedly flips the
decision, the result says so rather than silently returning nothing.
"""
from __future__ import annotations

from .decision_engine import DecisionEngine
from .demo_models import CounterfactualResult
from .evidence_scorer import EvidenceScorer
from .models import Decision, DisputeCase, Merchant


def find_minimal_flip(
    *,
    engine: DecisionEngine,
    scorer: EvidenceScorer,
    case: DisputeCase,
    merchant: Merchant,
    reason_code: str,
    reason_label: str,
    required_evidence: list[str],
    present_evidence: list[str],
    missing_evidence: list[str],
    baseline_decision: Decision,
) -> CounterfactualResult:
    if not missing_evidence:
        return CounterfactualResult(
            case_id=case.case_id,
            baseline_action=baseline_decision.action,
            flips=False,
            checked_items=[],
            note="No missing evidence — the evidence set is already complete for this dispute type.",
        )

    checked: list[str] = []
    best_flip: CounterfactualResult | None = None

    for candidate in missing_evidence:
        checked.append(candidate)
        hypothetical_present = sorted(set(present_evidence) | {candidate})
        hypothetical_missing = [e for e in missing_evidence if e != candidate]

        hypothetical_assessment = scorer.assess(
            reason_code=reason_code,
            reason_label=reason_label,
            required_evidence=required_evidence,
            present_evidence=hypothetical_present,
            missing_evidence=hypothetical_missing,
        )
        hypothetical_decision = engine.decide(case, merchant, hypothetical_assessment)

        if hypothetical_decision.action != baseline_decision.action:
            ev_delta = hypothetical_decision.ev_contest_inr - baseline_decision.ev_contest_inr
            candidate_result = CounterfactualResult(
                case_id=case.case_id,
                baseline_action=baseline_decision.action,
                flips=True,
                flipping_evidence=candidate,
                resulting_action=hypothetical_decision.action,
                ev_delta_inr=round(ev_delta, 2),
                checked_items=list(checked),
                note=f"Adding '{candidate}' alone flips {baseline_decision.action.value} "
                     f"-> {hypothetical_decision.action.value} "
                     f"(EV shifts by ₹{ev_delta:,.0f}).",
            )
            # Keep the flip with the largest EV swing if multiple single items flip it.
            if best_flip is None or (candidate_result.ev_delta_inr or 0) > (best_flip.ev_delta_inr or 0):
                best_flip = candidate_result

    if best_flip is not None:
        best_flip.checked_items = checked  # report all items actually tested, not just the winner
        return best_flip

    return CounterfactualResult(
        case_id=case.case_id,
        baseline_action=baseline_decision.action,
        flips=False,
        checked_items=checked,
        note="No single missing evidence item would change this decision — would need multiple evidence items together.",
    )