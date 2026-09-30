from __future__ import annotations

import hmac
import time

from fastapi import APIRouter, Depends, Header, HTTPException, Query, Request, Response
from fastapi.concurrency import run_in_threadpool
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import sessionmaker

from app import insights
from app import repository as repo
from app.api.schemas import DisputeOut, EvaluationOut, OutcomeIn, QueueItem, QueuePage, ReviewIn
from app.db import ActionEnum, OutcomeEnum, session_scope
from app.domain import DisputeCase
from app.service import DisputeService

router = APIRouter()


# ---------------------------------------------------------------- dependencies

def get_service(request: Request) -> DisputeService:
    return request.app.state.service


def get_sessions(request: Request) -> sessionmaker:
    return request.app.state.session_factory


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)) -> None:
    """Write endpoints require X-API-Key when ABSTAIN_API_KEY is configured."""
    expected = request.app.state.settings.api_key
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="Missing or invalid X-API-Key")


# ---------------------------------------------------------------- health

@router.get("/health", tags=["ops"])
def health() -> dict:
    return {"status": "ok"}


@router.get("/ready", tags=["ops"])
def ready(request: Request, sessions: sessionmaker = Depends(get_sessions)) -> dict:
    with session_scope(sessions) as s:
        s.execute(text("SELECT 1"))
    service: DisputeService = request.app.state.service
    return {"status": "ready", "model_version": service.model.version, "extraction_mode": service.mode,
            "llm_configured": service.llm is not None}


@router.get("/v1/reference", tags=["reference"])
def reference(service: DisputeService = Depends(get_service)) -> dict:
    return {
        "merchants": [m.model_dump(mode="json") for m in service.catalog.merchants.values()],
        "reason_codes": [rc.model_dump(mode="json") for rc in service.catalog.reason_codes.values()],
    }


# ---------------------------------------------------------------- disputes

@router.post("/v1/disputes/evaluate", response_model=EvaluationOut, status_code=201,
             dependencies=[Depends(require_api_key)], tags=["disputes"])
async def evaluate(
    case: DisputeCase,
    response: Response,
    idempotency_key: str | None = Header(default=None, max_length=128),
    service: DisputeService = Depends(get_service),
    sessions: sessionmaker = Depends(get_sessions),
) -> EvaluationOut:
    req_hash = repo.request_hash(case)

    def lookup():
        with session_scope(sessions) as s:
            row = repo.find_idempotent(s, idempotency_key, req_hash)
            return EvaluationOut.from_row(row, replay=True) if row else None

    if idempotency_key and (replayed := await run_in_threadpool(lookup)):
        response.status_code = 200
        return replayed

    started = time.perf_counter()
    ev = await service.evaluate(case)
    latency_ms = int((time.perf_counter() - started) * 1000)

    def save():
        with session_scope(sessions) as s:
            row = repo.save_evaluation(s, ev, req_hash=req_hash, latency_ms=latency_ms,
                                       idempotency_key=idempotency_key)
            return EvaluationOut.from_row(row)

    try:
        return await run_in_threadpool(save)
    except IntegrityError:
        # A concurrent request with the same Idempotency-Key committed first.
        if idempotency_key and (replayed := await run_in_threadpool(lookup)):
            response.status_code = 200
            return replayed
        raise


@router.get("/v1/disputes/{case_id}", response_model=DisputeOut, tags=["disputes"])
def get_dispute(case_id: str, sessions: sessionmaker = Depends(get_sessions)) -> DisputeOut:
    with session_scope(sessions) as s:
        d = repo.get_dispute(s, case_id)
        latest = repo.latest_evaluation(s, case_id)
        return DisputeOut(
            case_id=d.case_id, merchant_id=d.merchant_id, reason_code=d.reason_code, category=d.category,
            amount_inr=float(d.amount_inr), outcome=d.outcome.outcome.value if d.outcome else None,
            latest=EvaluationOut.from_row(latest),
            history=[{"evaluation_id": e.id, "created_at": e.created_at, "action": e.action.value,
                      "p_win": e.p_win, "extraction_source": e.extraction_source,
                      "reviewed_as": e.review.action.value if e.review else None} for e in d.evaluations],
        )


@router.post("/v1/disputes/{case_id}/review", dependencies=[Depends(require_api_key)], tags=["disputes"])
def review(case_id: str, body: ReviewIn, sessions: sessionmaker = Depends(get_sessions)) -> dict:
    with session_scope(sessions) as s:
        r = repo.record_review(s, case_id, reviewer=body.reviewer, action=ActionEnum(body.action), note=body.note)
        return {"case_id": case_id, "evaluation_id": r.evaluation_id, "action": r.action.value, "reviewer": r.reviewer}


@router.post("/v1/disputes/{case_id}/outcome", dependencies=[Depends(require_api_key)], tags=["disputes"])
def outcome(case_id: str, body: OutcomeIn, sessions: sessionmaker = Depends(get_sessions)) -> dict:
    with session_scope(sessions) as s:
        row = repo.record_outcome(s, case_id, OutcomeEnum(body.outcome))
        return {"case_id": case_id, "outcome": row.outcome.value}


@router.get("/v1/escalations", response_model=QueuePage, tags=["review queue"])
def escalations(limit: int = Query(20, ge=1, le=100), offset: int = Query(0, ge=0),
                sessions: sessionmaker = Depends(get_sessions)) -> QueuePage:
    with session_scope(sessions) as s:
        rows, total = repo.escalation_queue(s, limit=limit, offset=offset)
        items = [
            QueueItem(rank=offset + i + 1, case_id=r.case_id, evaluation_id=r.id,
                      amount_inr=float(r.dispute.amount_inr), p_win=r.p_win, p_win_std=r.p_win_std,
                      review_value_inr=float(r.review_value_inr),
                      net_review_value_inr=float(r.review_value_inr - r.review_cost_inr), reasons=r.reasons)
            for i, r in enumerate(rows)
        ]
    return QueuePage(items=items, total=total, limit=limit, offset=offset)


# ---------------------------------------------------------------- insights

def _report(sessions: sessionmaker, service: DisputeService, merchant_id: str | None) -> dict:
    with session_scope(sessions) as s:
        everything = repo.resolved_cases(s)  # benchmarks always need the whole portfolio
        rows = [r for r in everything if r[2].merchant_id == merchant_id] if merchant_id else everything
        causes = insights.root_causes(rows, service.catalog)
        calib = insights.calibration(rows)
        econ = insights.realized_economics(rows)
        weak = insights.weaknesses(everything, service.catalog)
    title = f"Merchant {merchant_id}" if merchant_id else "Portfolio"
    if merchant_id:
        weak_view = {"merchant": weak["merchants"].get(merchant_id), "systemic": weak["systemic"]}
    else:
        weak_view = weak
    return {"scope": merchant_id or "portfolio", "root_causes": causes, "weaknesses": weak_view,
            "calibration": calib, "economics": econ,
            "markdown": insights.render_markdown(title, causes, calib, econ, weak, merchant_id)}


@router.get("/v1/reports/portfolio", tags=["insights"])
def portfolio_report(sessions: sessionmaker = Depends(get_sessions),
                     service: DisputeService = Depends(get_service)) -> dict:
    return _report(sessions, service, None)


@router.get("/v1/reports/merchants/{merchant_id}", tags=["insights"])
def merchant_report(merchant_id: str, sessions: sessionmaker = Depends(get_sessions),
                    service: DisputeService = Depends(get_service)) -> dict:
    service.catalog.merchant(merchant_id)  # 422 on unknown merchant
    return _report(sessions, service, merchant_id)


@router.get("/v1/monitoring/calibration", tags=["insights"])
def calibration(sessions: sessionmaker = Depends(get_sessions)) -> dict:
    with session_scope(sessions) as s:
        return insights.calibration(repo.resolved_cases(s))
