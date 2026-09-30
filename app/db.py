"""
Persistence: SQLAlchemy 2.0 models and session plumbing. Schema changes go
through Alembic migrations (alembic/versions), never create_all() in prod.

Design notes:
  - Evaluations are append-only. Re-evaluating a dispute adds a row; the
    latest row per case is the current state, older rows are history.
  - Each evaluation stores a snapshot of its inputs (documents, merchant
    profile, model version). An audit has to show what was known when the
    decision was made, not what the reference data says today.
  - Money is Numeric(12, 2), never float.
"""
from __future__ import annotations

import enum
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from decimal import Decimal

from sqlalchemy import (
    JSON,
    DateTime,
    Enum,
    ForeignKey,
    Index,
    Integer,
    Numeric,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    text,
)
from sqlalchemy.engine import Engine
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship, sessionmaker


def utcnow() -> datetime:
    return datetime.now(UTC)


class Base(DeclarativeBase):
    pass


class ActionEnum(str, enum.Enum):
    CONTEST = "CONTEST"
    CONCEDE = "CONCEDE"
    ESCALATE = "ESCALATE"


class OutcomeEnum(str, enum.Enum):
    won = "won"
    lost = "lost"


Money = Numeric(12, 2)


class DisputeRow(Base):
    __tablename__ = "disputes"

    case_id: Mapped[str] = mapped_column(String(64), primary_key=True)
    merchant_id: Mapped[str] = mapped_column(String(32), index=True)
    reason_code: Mapped[str] = mapped_column(String(16))
    category: Mapped[str] = mapped_column(String(32))
    amount_inr: Mapped[Decimal] = mapped_column(Money)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    evaluations: Mapped[list[EvaluationRow]] = relationship(back_populates="dispute", order_by="EvaluationRow.id")
    outcome: Mapped[OutcomeRow | None] = relationship(back_populates="dispute")


class EvaluationRow(Base):
    __tablename__ = "evaluations"
    __table_args__ = (Index("ix_evaluations_case_id_id", "case_id", "id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("disputes.case_id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    input_hash: Mapped[str] = mapped_column(String(64), index=True)

    action: Mapped[ActionEnum] = mapped_column(Enum(ActionEnum, name="action_enum"))
    confidence: Mapped[str] = mapped_column(String(8))
    p_win: Mapped[float]
    p_win_std: Mapped[float]
    ev_contest_inr: Mapped[Decimal] = mapped_column(Money)
    contest_cost_inr: Mapped[Decimal] = mapped_column(Money)
    review_value_inr: Mapped[Decimal] = mapped_column(Money)
    review_cost_inr: Mapped[Decimal] = mapped_column(Money)

    facts: Mapped[dict] = mapped_column(JSON)
    fact_agreement: Mapped[dict] = mapped_column(JSON)
    unread_facts: Mapped[list] = mapped_column(JSON)
    conflicts: Mapped[list] = mapped_column(JSON)
    reasons: Mapped[list] = mapped_column(JSON)
    counterfactual: Mapped[dict] = mapped_column(JSON)
    contributions: Mapped[dict] = mapped_column(JSON)
    audit_log: Mapped[list] = mapped_column(JSON, default=list, server_default=text("'[]'"))

    extraction_source: Mapped[str] = mapped_column(String(24))
    degraded_reason: Mapped[str | None] = mapped_column(String(32))
    llm_calls: Mapped[int] = mapped_column(default=0)
    llm_tokens: Mapped[int] = mapped_column(default=0)
    model_version: Mapped[str] = mapped_column(String(32))
    latency_ms: Mapped[int] = mapped_column(default=0)
    input_snapshot: Mapped[dict] = mapped_column(JSON)       # case fields + documents
    merchant_snapshot: Mapped[dict] = mapped_column(JSON)

    dispute: Mapped[DisputeRow] = relationship(back_populates="evaluations")
    review: Mapped[ReviewRow | None] = relationship(back_populates="evaluation")


class ReviewRow(Base):
    """A human decision on an escalated evaluation."""
    __tablename__ = "reviews"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluations.id"), unique=True)
    case_id: Mapped[str] = mapped_column(ForeignKey("disputes.case_id"), index=True)
    reviewer: Mapped[str] = mapped_column(String(64))
    action: Mapped[ActionEnum] = mapped_column(Enum(ActionEnum, name="action_enum"))
    note: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    evaluation: Mapped[EvaluationRow] = relationship(back_populates="review")


class OutcomeRow(Base):
    """Network resolution, recorded weeks after the decision (e.g. by a reconciliation job)."""
    __tablename__ = "outcomes"

    case_id: Mapped[str] = mapped_column(ForeignKey("disputes.case_id"), primary_key=True)
    outcome: Mapped[OutcomeEnum] = mapped_column(Enum(OutcomeEnum, name="outcome_enum"))
    recorded_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    dispute: Mapped[DisputeRow] = relationship(back_populates="outcome")


class IdempotencyKeyRow(Base):
    __tablename__ = "idempotency_keys"
    __table_args__ = (UniqueConstraint("key", name="uq_idempotency_key"),)

    key: Mapped[str] = mapped_column(String(128), primary_key=True)
    request_hash: Mapped[str] = mapped_column(String(64))
    evaluation_id: Mapped[int] = mapped_column(ForeignKey("evaluations.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


# ---------------------------------------------------------------- plumbing

def build_engine(url: str) -> Engine:
    if url.startswith("postgres://"):  # Render/Heroku-style URLs
        url = "postgresql://" + url[len("postgres://"):]
    kwargs: dict = {"pool_pre_ping": True}
    if url.startswith("sqlite"):
        kwargs["connect_args"] = {"check_same_thread": False}
    return create_engine(url, **kwargs)


def build_session_factory(engine: Engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def session_scope(factory: sessionmaker[Session]) -> Iterator[Session]:
    session = factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
