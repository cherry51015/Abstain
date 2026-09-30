"""HTTP response/request models. Request bodies for evaluation reuse the
domain `DisputeCase` directly, so validation rules live in one place."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from app.db import EvaluationRow


class EvaluationOut(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    evaluation_id: int
    case_id: str
    created_at: datetime
    action: str
    confidence: str
    p_win: float
    p_win_std: float
    ev_contest_inr: float
    contest_cost_inr: float
    review_value_inr: float
    review_cost_inr: float
    facts: dict[str, str]
    fact_agreement: dict[str, float]
    unread_facts: list[str]
    conflicts: list[str]
    reasons: list[str]
    counterfactual: dict
    contributions: dict[str, float]
    audit_log: list[str]
    extraction_source: str
    degraded_reason: str | None
    llm_calls: int
    llm_tokens: int
    model_version: str
    latency_ms: int
    reviewed_as: str | None = None
    idempotent_replay: bool = False

    @classmethod
    def from_row(cls, row: EvaluationRow, *, replay: bool = False) -> EvaluationOut:
        return cls(
            evaluation_id=row.id, case_id=row.case_id, created_at=row.created_at, action=row.action.value,
            confidence=row.confidence, p_win=row.p_win, p_win_std=row.p_win_std,
            ev_contest_inr=float(row.ev_contest_inr), contest_cost_inr=float(row.contest_cost_inr),
            review_value_inr=float(row.review_value_inr), review_cost_inr=float(row.review_cost_inr),
            facts=row.facts, fact_agreement=row.fact_agreement, unread_facts=row.unread_facts,
            conflicts=row.conflicts, reasons=row.reasons, counterfactual=row.counterfactual,
            contributions=row.contributions, audit_log=row.audit_log or [], extraction_source=row.extraction_source,
            degraded_reason=row.degraded_reason, llm_calls=row.llm_calls, llm_tokens=row.llm_tokens,
            model_version=row.model_version, latency_ms=row.latency_ms,
            reviewed_as=row.review.action.value if row.review else None, idempotent_replay=replay,
        )


class DisputeOut(BaseModel):
    case_id: str
    merchant_id: str
    reason_code: str
    category: str
    amount_inr: float
    outcome: str | None
    latest: EvaluationOut
    history: list[dict]


class QueueItem(BaseModel):
    rank: int
    case_id: str
    evaluation_id: int
    amount_inr: float
    p_win: float
    p_win_std: float
    review_value_inr: float
    net_review_value_inr: float
    reasons: list[str]


class QueuePage(BaseModel):
    items: list[QueueItem]
    total: int
    limit: int
    offset: int


class ReviewIn(BaseModel):
    reviewer: str = Field(min_length=1, max_length=64)
    action: Literal["CONTEST", "CONCEDE"]
    note: str = Field(default="", max_length=2000)


class OutcomeIn(BaseModel):
    outcome: Literal["won", "lost"]
