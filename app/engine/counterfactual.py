"""
"What evidence would change this decision?"

For a CONCEDE or ESCALATE, try making each relevant fact that is not already
favourable into 'yes', re-estimate P(win) with the same model and re-run the
same policy. It returns every *minimal* set of facts (single facts first,
then pairs of facts that do not already flip on their own) that turns the
decision into CONTEST, ranked by the expected-value gain. On a large dispute
this is the merchant's checklist of what is worth chasing, and each item says
what kind of action it needs:

  missing  no document covers the fact            -> go and collect it
  unread   a document exists but was not read     -> it may already be there
  adverse  the documents say 'no'                 -> an operations fix, not evidence

Deterministic and LLM-free: the what-if runs through the same code path as
the real decision, so the two cannot drift apart, and it costs no API calls.
"""
from __future__ import annotations

from collections.abc import Callable
from itertools import combinations
from typing import Literal

from pydantic import BaseModel

from app.domain import Action, DisputeCase, EvidenceFacts, Merchant, Tri
from app.engine.conflicts import find_conflicts
from app.engine.decision_engine import Decision, DecisionEngine
from app.extraction.aggregate import majority
from app.scoring.win_model import WinEstimate

EstimateFn = Callable[[list[EvidenceFacts]], WinEstimate]
FactKind = Literal["missing", "unread", "adverse"]


class FlipOption(BaseModel):
    evidence: list[str]
    kinds: dict[str, FactKind]
    p_win_after: float
    ev_contest_after_inr: float
    ev_gain_inr: float


class Counterfactual(BaseModel):
    flips: bool
    evidence_needed: list[str] = []          # the best option (largest EV gain)
    resulting_action: Action | None = None
    ev_gain_inr: float | None = None
    options: list[FlipOption] = []           # every minimal flipping set, best first
    note: str


def find_flip(
    *,
    engine: DecisionEngine,
    estimate: EstimateFn,
    case: DisputeCase,
    merchant: Merchant,
    relevant_facts: list[str],
    fact_samples: list[EvidenceFacts],
    baseline: Decision,
    max_set_size: int = 2,
    max_options: int = 5,
    unread: list[str] | tuple[str, ...] = (),
) -> Counterfactual:
    if baseline.action == Action.CONTEST:
        return Counterfactual(flips=False, note="Already CONTEST; nothing to obtain.")

    representative = majority(fact_samples)
    candidates = [f for f in relevant_facts if representative.get(f) != Tri.YES]
    if not candidates:
        return Counterfactual(flips=False, note="Every relevant fact is already favourable; the economics decide.")

    options: list[FlipOption] = []
    flipping_singles: set[str] = set()
    for size in range(1, max_set_size + 1):
        for combo in combinations(candidates, size):
            if size > 1 and flipping_singles & set(combo):
                continue  # not minimal: a subset already flips the decision
            samples = [f.model_copy(update={x: Tri.YES for x in combo}) for f in fact_samples]
            decision = engine.decide(case, merchant, estimate(samples), find_conflicts(majority(samples), relevant_facts))
            if decision.action != Action.CONTEST:
                continue
            if size == 1:
                flipping_singles.add(combo[0])
            options.append(FlipOption(
                evidence=list(combo),
                kinds={f: _kind(f, representative, unread) for f in combo},
                p_win_after=decision.p_win,
                ev_contest_after_inr=decision.ev_contest_inr,
                ev_gain_inr=round(decision.ev_contest_inr - baseline.ev_contest_inr, 2),
            ))

    if not options:
        return Counterfactual(flips=False,
                              note=f"No set of up to {max_set_size} additional facts makes contesting worthwhile.")
    options.sort(key=lambda o: (len(o.evidence), -o.ev_gain_inr))
    options = options[:max_options]
    best = max(options, key=lambda o: o.ev_gain_inr)
    note = (f"{' and '.join(_describe(f, best.kinds[f]) for f in best.evidence)} would flip "
            f"{baseline.action.value} -> CONTEST (P(win) {baseline.p_win:.2f} -> {best.p_win_after:.2f}, "
            f"EV +₹{best.ev_gain_inr:,.0f}).")
    if len(options) > 1:
        note += f" {len(options) - 1} other option(s) would also flip it."
    return Counterfactual(flips=True, evidence_needed=best.evidence, resulting_action=Action.CONTEST,
                          ev_gain_inr=best.ev_gain_inr, options=options, note=note)


def _kind(fact: str, facts: EvidenceFacts, unread) -> FactKind:
    if fact in unread:
        return "unread"
    return "adverse" if facts.get(fact) == Tri.NO else "missing"


def _describe(fact: str, kind: FactKind) -> str:
    return {
        "unread": f"confirming {fact} (a document may already say so; it could not be read)",
        "adverse": f"{fact} being yes instead of no (an operations fix, not evidence to collect)",
        "missing": f"obtaining evidence for {fact}",
    }[kind]
