"""Additional shared types for the demo-facing modules (counterfactual, attention
ranking). Kept in a separate file from models.py rather than appended there —
these are downstream consumers of a Decision, not inputs to the engine, and
keeping that direction one-way is deliberate."""
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

from .models import Action, Decision


class CounterfactualResult(BaseModel):
    case_id: str
    baseline_action: Action
    flips: bool
    flipping_evidence: Optional[str] = None
    resulting_action: Optional[Action] = None
    ev_delta_inr: Optional[float] = None
    checked_items: list[str] = []
    note: str = ""


class AttentionRankedCase(BaseModel):
    case_id: str
    decision: Decision
    human_review_value_inr: float
    rank: int