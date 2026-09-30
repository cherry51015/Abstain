"""All database reads and writes. Route handlers never build queries themselves."""
from __future__ import annotations

import hashlib
import json
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.db import ActionEnum, DisputeRow, EvaluationRow, IdempotencyKeyRow, OutcomeEnum, OutcomeRow, ReviewRow
from app.domain import DisputeCase
from app.service import Evaluation


class ConflictError(Exception):
    """Request is inconsistent with the current state (maps to HTTP 409)."""


class NotFoundError(LookupError):
    pass


def _money(x: float) -> Decimal:
    return Decimal(str(round(x, 2)))


def request_hash(case: DisputeCase) -> str:
    canonical = json.dumps(case.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


# ---------------------------------------------------------------- idempotency

def find_idempotent(session: Session, key: str, req_hash: str) -> EvaluationRow | None:
    """Stripe-style semantics: same key + same body replays the stored
    result; same key + different body is a client error."""
    row = session.get(IdempotencyKeyRow, key)
    if row is None:
        return None
    if row.request_hash != req_hash:
        raise ConflictError("Idempotency-Key was already used with a different request body")
    return session.get(EvaluationRow, row.evaluation_id)


# ---------------------------------------------------------------- writes

def save_evaluation(session: Session, ev: Evaluation, *, req_hash: str, latency_ms: int,
                    idempotency_key: str | None = None) -> EvaluationRow:
    case, d = ev.case, ev.decision
    dispute = session.get(DisputeRow, case.case_id)
    if dispute is None:
        dispute = DisputeRow(case_id=case.case_id)
        session.add(dispute)
    elif dispute.merchant_id != case.merchant_id:
        raise ConflictError(f"case_id {case.case_id!r} already belongs to merchant {dispute.merchant_id!r}")
    dispute.merchant_id = case.merchant_id
    dispute.reason_code = case.reason_code
    dispute.category = ev.reason_code.category.value
    dispute.amount_inr = case.amount_inr

    row = EvaluationRow(
        case_id=case.case_id, input_hash=req_hash,
        action=ActionEnum(d.action.value), confidence=d.confidence,
        p_win=d.p_win, p_win_std=d.p_win_std,
        ev_contest_inr=_money(d.ev_contest_inr), contest_cost_inr=_money(d.contest_cost_inr),
        review_value_inr=_money(d.review_value_inr), review_cost_inr=_money(d.review_cost_inr),
        facts={k: v.value for k, v in ev.facts.model_dump().items()},
        fact_agreement=ev.fact_agreement, unread_facts=ev.unread_facts, conflicts=ev.conflicts,
        reasons=d.reasons, counterfactual=ev.counterfactual.model_dump(mode="json"),
        contributions=ev.contributions, audit_log=ev.audit_log,
        extraction_source=ev.extraction.source, degraded_reason=ev.degraded_reason,
        llm_calls=ev.extraction.llm_calls,
        llm_tokens=ev.extraction.prompt_tokens + ev.extraction.completion_tokens,
        model_version=ev.model_version, latency_ms=latency_ms,
        input_snapshot=case.model_dump(mode="json"),
        merchant_snapshot=ev.merchant.model_dump(mode="json"),
    )
    session.add(row)
    session.flush()
    if idempotency_key:
        session.add(IdempotencyKeyRow(key=idempotency_key, request_hash=req_hash, evaluation_id=row.id))
    return row


def record_review(session: Session, case_id: str, *, reviewer: str, action: ActionEnum, note: str) -> ReviewRow:
    latest = latest_evaluation(session, case_id)
    if latest.action != ActionEnum.ESCALATE:
        raise ConflictError(f"Latest evaluation of {case_id!r} is {latest.action.value}, not ESCALATE")
    if latest.review is not None:
        raise ConflictError(f"Evaluation {latest.id} was already reviewed by {latest.review.reviewer!r}")
    if action == ActionEnum.ESCALATE:
        raise ConflictError("A review must resolve to CONTEST or CONCEDE")
    review = ReviewRow(evaluation_id=latest.id, case_id=case_id, reviewer=reviewer, action=action, note=note)
    session.add(review)
    session.flush()
    return review


def record_outcome(session: Session, case_id: str, outcome: OutcomeEnum) -> OutcomeRow:
    if session.get(DisputeRow, case_id) is None:
        raise NotFoundError(f"No dispute {case_id!r}")
    row = session.get(OutcomeRow, case_id)
    if row is not None and row.outcome != outcome:
        raise ConflictError(f"Outcome for {case_id!r} already recorded as {row.outcome.value}")
    if row is None:
        row = OutcomeRow(case_id=case_id, outcome=outcome)
        session.add(row)
    return row


# ---------------------------------------------------------------- reads

def get_dispute(session: Session, case_id: str) -> DisputeRow:
    dispute = session.get(DisputeRow, case_id)
    if dispute is None:
        raise NotFoundError(f"No dispute {case_id!r}")
    return dispute


def latest_evaluation(session: Session, case_id: str) -> EvaluationRow:
    row = session.scalars(
        select(EvaluationRow).where(EvaluationRow.case_id == case_id).order_by(EvaluationRow.id.desc()).limit(1)
    ).first()
    if row is None:
        raise NotFoundError(f"No dispute {case_id!r}")
    return row


def _latest_ids():
    return select(func.max(EvaluationRow.id)).group_by(EvaluationRow.case_id).scalar_subquery()


def escalation_queue(session: Session, *, limit: int, offset: int) -> tuple[list[EvaluationRow], int]:
    """Open escalations (latest evaluation is ESCALATE and unreviewed), ranked
    by the net value of a review: VOI minus review cost. Not FIFO."""
    base = (
        select(EvaluationRow)
        .outerjoin(ReviewRow, ReviewRow.evaluation_id == EvaluationRow.id)
        .where(EvaluationRow.id.in_(_latest_ids()))
        .where(EvaluationRow.action == ActionEnum.ESCALATE)
        .where(ReviewRow.id.is_(None))
    )
    total = session.scalar(select(func.count()).select_from(base.subquery())) or 0
    rows = session.scalars(
        base.order_by((EvaluationRow.review_value_inr - EvaluationRow.review_cost_inr).desc(), EvaluationRow.id)
        .limit(limit).offset(offset)
    ).all()
    return list(rows), total


def resolved_cases(session: Session, merchant_id: str | None = None) -> list[tuple[EvaluationRow, OutcomeRow, DisputeRow]]:
    """Latest evaluation + recorded outcome for every resolved dispute."""
    q = (
        select(EvaluationRow, OutcomeRow, DisputeRow)
        .join(OutcomeRow, OutcomeRow.case_id == EvaluationRow.case_id)
        .join(DisputeRow, DisputeRow.case_id == EvaluationRow.case_id)
        .where(EvaluationRow.id.in_(_latest_ids()))
    )
    if merchant_id:
        q = q.where(DisputeRow.merchant_id == merchant_id)
    return [tuple(r) for r in session.execute(q).all()]
