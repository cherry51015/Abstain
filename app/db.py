"""
db.py

Persistence layer: disputes, decisions, and the audit trail. Uses SQLAlchemy
2.0 declarative style. Works against Postgres (docker-compose) or SQLite
(local dev without Docker) via the same DB_URL-driven engine — no code
branches on which one you're using.

Design choice worth stating: merchant fields are stored as a SNAPSHOT on
each DisputeORM row, not as a live foreign-key join to a mutable merchant
table. An audit trail has to reflect what was true at decision time — if a
merchant's risk_tolerance changes next week, a decision made today should
still show the risk_tolerance that was actually used, not silently inherit
the update. Live merchant reference data (for making NEW decisions) is
loaded from merchants.json at app startup, same as reason_codes.json —
this table is for what was true when, not the live source of truth.
"""
from __future__ import annotations

import os
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator

from sqlalchemy import (
    create_engine, Column, String, Float, Integer, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import declarative_base, relationship, sessionmaker, Session

from app.engine.models import Decision as DecisionModel
from app.graph.state import PipelineState

Base = declarative_base()


class DisputeORM(Base):
    __tablename__ = "disputes"

    case_id = Column(String, primary_key=True)
    merchant_id = Column(String, nullable=False)
    reason_code = Column(String, nullable=False)
    dispute_amount_inr = Column(Float, nullable=False)
    response_deadline_days_left = Column(Integer, nullable=False)
    is_repeat_dispute = Column(Boolean, default=False)
    conflicting_evidence = Column(Boolean, default=False)
    dispute_reason_text = Column(Text, default="")

    # merchant snapshot at decision time — see module docstring
    merchant_risk_tolerance = Column(String, nullable=False)
    merchant_historical_win_rate = Column(Float, nullable=False)
    merchant_threshold_proximity = Column(Float, nullable=False)

    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    decisions = relationship("DecisionORM", back_populates="dispute", cascade="all, delete-orphan")
    audit_entries = relationship("AuditLogEntryORM", back_populates="dispute", cascade="all, delete-orphan")


class DecisionORM(Base):
    __tablename__ = "decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey("disputes.case_id"), nullable=False)

    action = Column(String, nullable=False)
    win_probability = Column(Float, nullable=False)
    uncertainty = Column(Float, nullable=False)
    ev_contest_inr = Column(Float, nullable=False)
    ops_cost_inr = Column(Float, nullable=False)
    portfolio_risk_penalty_inr = Column(Float, nullable=False)
    confidence_label = Column(String, nullable=False)
    memo = Column(Text, nullable=False)

    evidence_source = Column(String, nullable=True)  # "llm" or "fallback_heuristic" — flags degraded runs
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    dispute = relationship("DisputeORM", back_populates="decisions")


class AuditLogEntryORM(Base):
    __tablename__ = "audit_log_entries"

    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(String, ForeignKey("disputes.case_id"), nullable=False)
    sequence = Column(Integer, nullable=False)
    message = Column(Text, nullable=False)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    dispute = relationship("DisputeORM", back_populates="audit_entries")


# ---------------- engine / session plumbing ----------------

def get_db_url() -> str:
    """Defaults to a local SQLite file if DB_URL isn't set, so the app runs
    without requiring docker-compose's Postgres for quick local iteration."""
    return os.environ.get("DB_URL", "sqlite:///./abstain_local.db")


def build_engine(db_url: str | None = None):
    url = db_url or get_db_url()
    connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
    return create_engine(url, connect_args=connect_args)


def init_db(engine) -> None:
    Base.metadata.create_all(bind=engine)


def build_session_factory(engine) -> sessionmaker:
    return sessionmaker(bind=engine, autoflush=False, autocommit=False, future=True)


@contextmanager
def session_scope(session_factory: sessionmaker) -> Iterator[Session]:
    session = session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


# ---------------- persistence of a completed pipeline run ----------------

def save_pipeline_result(session: Session, state: PipelineState) -> None:
    """Persists a completed PipelineState: the dispute (with merchant
    snapshot), the decision (if one was reached), and the full audit log.
    Idempotent on case_id — re-running the same case_id updates rather than
    duplicates the dispute row, but always appends fresh decision/audit rows
    so history of repeated evaluations is preserved, not overwritten."""
    existing = session.get(DisputeORM, state.case.case_id)
    if existing is None:
        existing = DisputeORM(case_id=state.case.case_id)
        session.add(existing)

    existing.merchant_id = state.merchant.merchant_id
    existing.reason_code = state.case.reason_code
    existing.dispute_amount_inr = state.case.dispute_amount_inr
    existing.response_deadline_days_left = state.case.response_deadline_days_left
    existing.is_repeat_dispute = state.case.is_repeat_dispute
    existing.conflicting_evidence = state.case.conflicting_evidence
    existing.dispute_reason_text = state.dispute_reason_text
    existing.merchant_risk_tolerance = state.merchant.risk_tolerance
    existing.merchant_historical_win_rate = state.merchant.historical_win_rate
    existing.merchant_threshold_proximity = state.merchant.threshold_proximity

    if state.decision is not None:
        d: DecisionModel = state.decision
        session.add(DecisionORM(
            case_id=state.case.case_id,
            action=d.action.value,
            win_probability=d.win_probability,
            uncertainty=d.uncertainty,
            ev_contest_inr=d.ev_contest_inr,
            ops_cost_inr=d.ops_cost_inr,
            portfolio_risk_penalty_inr=d.portfolio_risk_penalty_inr,
            confidence_label=d.confidence_label,
            memo=d.memo,
            evidence_source=state.evidence_assessment.source if state.evidence_assessment else None,
        ))

    for i, message in enumerate(state.audit_log):
        session.add(AuditLogEntryORM(case_id=state.case.case_id, sequence=i, message=message))