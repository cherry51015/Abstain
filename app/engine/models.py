"""
Shared types for the decision engine and evidence scorer.
Kept dependency-free (stdlib + pydantic only) so decision_engine.py
never has a path to importing an LLM client, even transitively.
"""
from __future__ import annotations
from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field, field_validator


class Action(str, Enum):
    CONTEST = "CONTEST"
    CONCEDE = "CONCEDE"
    ESCALATE = "ESCALATE"


class Merchant(BaseModel):
    merchant_id: str
    historical_win_rate: float = Field(ge=0.0, le=1.0)
    current_chargeback_rate_pct: float = Field(ge=0.0)
    network_threshold_pct: float = Field(gt=0.0)
    ops_cost_per_contest_inr: float = Field(gt=0.0)
    risk_tolerance: str = Field(default="moderate")

    @field_validator("risk_tolerance")
    @classmethod
    def _valid_tolerance(cls, v: str) -> str:
        allowed = {"aggressive", "moderate", "conservative"}
        if v not in allowed:
            raise ValueError(f"risk_tolerance must be one of {allowed}, got {v!r}")
        return v

    @property
    def threshold_proximity(self) -> float:
        """0 = far from network chargeback-rate threshold, 1 = at/over it."""
        return max(0.0, min(1.5, self.current_chargeback_rate_pct / self.network_threshold_pct))


class EvidenceAssessment(BaseModel):
    """Output of evidence_scorer.py — the ONLY interface the decision engine
    is allowed to consume from the LLM side. No raw LLM output ever crosses
    this boundary unvalidated."""
    strength: float = Field(ge=0.0, le=1.0, description="Mean assessed evidence strength")
    uncertainty: float = Field(ge=0.0, le=1.0, description="Spread/disagreement across samples")
    source: str = Field(description="'llm' or 'fallback_heuristic'")
    key_gaps: list[str] = Field(default_factory=list)
    reasoning: str = ""


class DisputeCase(BaseModel):
    case_id: str
    merchant_id: str
    reason_code: str
    dispute_amount_inr: float = Field(gt=0.0)
    response_deadline_days_left: int = Field(ge=0)
    is_repeat_dispute: bool = False
    conflicting_evidence: bool = False


class Decision(BaseModel):
    case_id: str
    action: Action
    win_probability: float
    uncertainty: float
    ev_contest_inr: float
    ops_cost_inr: float
    portfolio_risk_penalty_inr: float
    reasons: list[str]
    confidence_label: str  # HIGH / MEDIUM / LOW
    memo: str  # human-readable risk-manager-style explanation