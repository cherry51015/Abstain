"""
main.py

FastAPI entrypoint. Wires together at startup: reason-code/merchant
reference data, the hybrid retrieval index, the evidence scorer (live LLM
if GROQ_API_KEY is set, fallback heuristic otherwise), the deterministic
decision engine, the LangGraph pipeline, and DB persistence — into three
endpoints: evaluate a dispute, look one up, and get the analyst's ranked
worklist of escalated cases.
"""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func

from app.db import (
    build_engine, init_db, build_session_factory, session_scope,
    save_pipeline_result, DisputeORM, DecisionORM, AuditLogEntryORM,
)
from app.engine.attention_ranking import rank_escalated_cases
from app.engine.decision_engine import DecisionEngine
from app.engine.evidence_scorer import EvidenceScorer
from app.engine.models import Action, Decision, DisputeCase, Merchant
from app.graph.pipeline import build_pipeline, run_case
from app.llm_client import build_llm_client
from app.reporting.portfolio_intelligence import (
    analyze_merchant, analyze_portfolio, load_resolved_records,
    render_markdown as render_root_cause_markdown,
)
from app.retrieval.index_builder import build_index

DATASET_DIR = Path(__file__).resolve().parent.parent / "dataset"

app = FastAPI(title="Abstain — AI Risk Decision Engine for Chargebacks")
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Abstain — AI Risk Decision Engine for Chargebacks")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup() -> None:
    with open(DATASET_DIR / "merchants.json") as f:
        app.state.merchants = {m["merchant_id"]: m for m in json.load(f)}

    app.state.retrieval_index = build_index(DATASET_DIR / "reason_codes.json")

    llm_client = build_llm_client()
    app.state.llm_configured = llm_client is not None
    app.state.evidence_scorer = EvidenceScorer(llm_client=llm_client)
    app.state.decision_engine = DecisionEngine()
    app.state.pipeline = build_pipeline(
        app.state.retrieval_index, app.state.evidence_scorer, app.state.decision_engine
    )

    app.state.db_engine = build_engine()
    init_db(app.state.db_engine)
    app.state.session_factory = build_session_factory(app.state.db_engine)

    # Resolved outcomes for the root-cause reports. Sourced from the
    # synthetic dataset, not the live decisions table — see
    # portfolio_intelligence.load_resolved_records for why true_outcome
    # isn't something the live pipeline ever has at decision time.
    app.state.resolved_records = load_resolved_records(DATASET_DIR)


# ---------------- request/response schemas ----------------

class EvaluateRequest(BaseModel):
    case_id: str
    merchant_id: str
    reason_code: str
    dispute_amount_inr: float = Field(gt=0)
    response_deadline_days_left: int = Field(ge=0)
    is_repeat_dispute: bool = False
    conflicting_evidence: bool = False
    present_evidence: list[str] = []
    dispute_reason_text: str


class EvaluateResponse(BaseModel):
    case_id: str
    action: str | None
    win_probability: float | None
    uncertainty: float | None
    confidence_label: str | None
    memo: str | None
    counterfactual_note: str | None
    audit_log: list[str]
    error: str | None
    degraded_mode: bool  # True if this specific run used the fallback heuristic, not a live LLM


# ---------------- endpoints ----------------

@app.get("/health")
def health():
    return {"status": "ok", "llm_configured": app.state.llm_configured}


@app.post("/disputes/evaluate", response_model=EvaluateResponse)
def evaluate_dispute(req: EvaluateRequest):
    merchant_data = app.state.merchants.get(req.merchant_id)
    if merchant_data is None:
        raise HTTPException(status_code=404, detail=f"Unknown merchant_id: {req.merchant_id!r}")

    merchant = Merchant(**merchant_data)
    case = DisputeCase(
        case_id=req.case_id, merchant_id=req.merchant_id, reason_code=req.reason_code,
        dispute_amount_inr=req.dispute_amount_inr,
        response_deadline_days_left=req.response_deadline_days_left,
        is_repeat_dispute=req.is_repeat_dispute, conflicting_evidence=req.conflicting_evidence,
    )

    result = run_case(
        app.state.pipeline, case, merchant,
        present_evidence=req.present_evidence, dispute_reason_text=req.dispute_reason_text,
    )

    with session_scope(app.state.session_factory) as session:
        save_pipeline_result(session, result)

    return EvaluateResponse(
        case_id=req.case_id,
        action=result.decision.action.value if result.decision else None,
        win_probability=result.decision.win_probability if result.decision else None,
        uncertainty=result.decision.uncertainty if result.decision else None,
        confidence_label=result.decision.confidence_label if result.decision else None,
        memo=result.decision.memo if result.decision else None,
        counterfactual_note=result.counterfactual.note if result.counterfactual else None,
        audit_log=result.audit_log,
        error=result.error,
        degraded_mode=(
            result.evidence_assessment.source == "fallback_heuristic"
            if result.evidence_assessment else not app.state.llm_configured
        ),
    )


@app.get("/disputes/{case_id}")
def get_dispute(case_id: str):
    with session_scope(app.state.session_factory) as session:
        dispute = session.get(DisputeORM, case_id)
        if dispute is None:
            raise HTTPException(status_code=404, detail=f"No dispute found for case_id={case_id!r}")

        latest_decision = (
            session.query(DecisionORM)
            .filter(DecisionORM.case_id == case_id)
            .order_by(DecisionORM.created_at.desc())
            .first()
        )
        audit = (
            session.query(AuditLogEntryORM)
            .filter(AuditLogEntryORM.case_id == case_id)
            .order_by(AuditLogEntryORM.sequence.asc())
            .all()
        )

        return {
            "case_id": case_id,
            "merchant_id": dispute.merchant_id,
            "reason_code": dispute.reason_code,
            "dispute_amount_inr": dispute.dispute_amount_inr,
            "decision": {
                "action": latest_decision.action,
                "win_probability": latest_decision.win_probability,
                "confidence_label": latest_decision.confidence_label,
                "memo": latest_decision.memo,
                "evidence_source": latest_decision.evidence_source,
            } if latest_decision else None,
            "audit_log": [entry.message for entry in audit],
        }


@app.get("/reports/portfolio")
def get_portfolio_report():
    """'Why are we losing overall?' — every merchant, aggregated."""
    report = analyze_portfolio(app.state.resolved_records)
    return {"scope": "portfolio", "markdown": render_root_cause_markdown(report)}


@app.get("/reports/merchant/{merchant_id}")
def get_merchant_report(merchant_id: str):
    """'Why is THIS merchant losing?' — same underlying numbers as the
    portfolio report, scoped to one merchant. This is the view the demo's
    merchant-selector flow should call, not /reports/portfolio."""
    try:
        report = analyze_merchant(app.state.resolved_records, merchant_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    return {"scope": "merchant", "merchant_id": merchant_id, "markdown": render_root_cause_markdown(report)}


@app.get("/escalations/ranked")
def get_ranked_escalations():
    """The analyst's actual worklist: every case currently sitting at
    ESCALATE, ordered by expected value of human review — not FIFO, not
    amount alone. Reconstructs Decision objects from stored rows; `reasons`
    isn't persisted as a separate list (the memo already carries the full
    explanation), so it's reconstructed empty here — a known, deliberate
    simplification, not an oversight.

    save_pipeline_result() is append-only on purpose (fresh DecisionORM row
    per re-evaluation, for audit history) — so this query can't just filter
    DecisionORM.action == ESCALATE directly, or a case that was ESCALATE on
    an earlier run and has since been re-evaluated to CONTEST/CONCEDE would
    still show up here as a stale, already-resolved entry (and a case
    escalated twice would show up twice). We first find the latest decision
    id per case_id, then join and filter on that set only."""
    with session_scope(app.state.session_factory) as session:
        latest_decision_ids = (
            session.query(func.max(DecisionORM.id))
            .group_by(DecisionORM.case_id)
            .subquery()
        )
        rows = (
            session.query(DecisionORM, DisputeORM)
            .join(DisputeORM, DecisionORM.case_id == DisputeORM.case_id)
            .filter(DecisionORM.id.in_(latest_decision_ids))
            .filter(DecisionORM.action == Action.ESCALATE.value)
            .all()
        )

        decisions: list[Decision] = []
        amounts: dict[str, float] = {}
        for dec_row, dispute_row in rows:
            decisions.append(Decision(
                case_id=dec_row.case_id, action=Action(dec_row.action),
                win_probability=dec_row.win_probability, uncertainty=dec_row.uncertainty,
                ev_contest_inr=dec_row.ev_contest_inr, ops_cost_inr=dec_row.ops_cost_inr,
                portfolio_risk_penalty_inr=dec_row.portfolio_risk_penalty_inr,
                reasons=[], confidence_label=dec_row.confidence_label, memo=dec_row.memo,
            ))
            amounts[dec_row.case_id] = dispute_row.dispute_amount_inr

    ranked = rank_escalated_cases(decisions, amounts)
    return [
        {
            "rank": r.rank, "case_id": r.case_id,
            "human_review_value_inr": r.human_review_value_inr,
            "action": r.decision.action.value, "memo": r.decision.memo,
        }
        for r in ranked
    ]