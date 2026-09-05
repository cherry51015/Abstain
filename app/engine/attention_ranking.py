"""
attention_ranking.py

Answers: given N cases sitting in the ESCALATE queue and one analyst with
limited time, which case should they look at first?

Deliberately NOT "highest amount first" or "oldest first" — those are the
naive defaults every unranked queue defaults to. The metric here rewards
cases where a human is likely to actually change the outcome (high
uncertainty) AND where that outcome matters (high amount). A ₹500 case
with wildly uncertain evidence is a lower priority than a ₹90,000 case
with moderately uncertain evidence, even though the first "feels" more
unresolved.

human_review_value = uncertainty * dispute_amount_inr - human_review_cost_inr

This is a deliberately simple, auditable formula — not a learned ranking
model — for the same reason the decision engine itself is deterministic:
a ranking that decides whose case gets attention should be explainable to
the analyst it's ranking for, not a black box.
"""
from __future__ import annotations
from dataclasses import dataclass

from .demo_models import AttentionRankedCase
from .models import Decision, Action


@dataclass(frozen=True)
class AttentionConfig:
    human_review_cost_inr: float = 500.0   # rough fully-loaded cost of one analyst review


def rank_escalated_cases(
    decisions: list[Decision],
    dispute_amounts_inr: dict[str, float],
    config: AttentionConfig | None = None,
) -> list[AttentionRankedCase]:
    """
    decisions: full set of Decision objects from a batch run (only ESCALATE
               ones are ranked; others are ignored here).
    dispute_amounts_inr: case_id -> amount, passed separately rather than
               re-deriving from Decision, since Decision intentionally
               doesn't carry the raw case amount (keeps it a pure output
               of the engine, not a re-export of the input).
    """
    cfg = config or AttentionConfig()
    escalated = [d for d in decisions if d.action == Action.ESCALATE]

    scored: list[tuple[Decision, float]] = []
    for d in escalated:
        amount = dispute_amounts_inr.get(d.case_id)
        if amount is None:
            raise KeyError(
                f"No dispute amount provided for case_id={d.case_id!r} — "
                f"cannot compute human-review value without it."
            )
        value = d.uncertainty * amount - cfg.human_review_cost_inr
        scored.append((d, value))

    # Highest value first — most worth a human's limited time.
    scored.sort(key=lambda pair: pair[1], reverse=True)

    return [
        AttentionRankedCase(
            case_id=d.case_id,
            decision=d,
            human_review_value_inr=round(value, 2),
            rank=i + 1,
        )
        for i, (d, value) in enumerate(scored)
    ]