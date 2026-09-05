"""
state.py

State threaded through every node in the ingest -> retrieve -> score ->
decide -> counterfactual -> log pipeline. Each node reads what it needs
and returns only the fields it adds or changes; LangGraph merges that into
the running state, so nodes never need the full picture, just their slice.
"""
from __future__ import annotations
from typing import Optional

from pydantic import BaseModel, ConfigDict

from app.engine.models import DisputeCase, Merchant, EvidenceAssessment, Decision
from app.engine.demo_models import CounterfactualResult
from app.retrieval.index_builder import ReasonCodeDoc


class PipelineState(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # inputs — required to start a run
    case: DisputeCase
    merchant: Merchant
    present_evidence: list[str]
    dispute_reason_text: str   # free-text description; drives the retrieval step

    # populated progressively as nodes run
    retrieved_reason_code: Optional[ReasonCodeDoc] = None
    missing_evidence: list[str] = []
    evidence_assessment: Optional[EvidenceAssessment] = None
    decision: Optional[Decision] = None
    counterfactual: Optional[CounterfactualResult] = None
    audit_log: list[str] = []
    error: Optional[str] = None