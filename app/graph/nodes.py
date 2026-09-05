"""
nodes.py

Individual LangGraph node functions. Each node is a plain function of
PipelineState -> dict (LangGraph merges the returned dict into state).
Dependencies (retrieval index, scorer, engine) are injected via factory
functions rather than imported as module-level globals, so the graph is
testable with fakes and doesn't silently depend on hidden singletons.

Failure handling philosophy: a node that can't do its job sets state.error
and logs why, rather than raising. The graph always reaches `log`, so
every run — including a failed one — produces an auditable trail. This
matters specifically because the brief expects an honest exception list,
not a system that only ever shows its successes.
"""
from __future__ import annotations
from typing import Callable

from app.engine.decision_engine import DecisionEngine
from app.engine.evidence_scorer import EvidenceScorer
from app.engine.counterfactual import find_minimal_flip
from app.retrieval.hybrid_search import search as hybrid_search
from app.retrieval.index_builder import RetrievalIndex

from .state import PipelineState


def make_ingest_node() -> Callable[[PipelineState], dict]:
    def ingest_node(state: PipelineState) -> dict:
        log = [f"Ingested case {state.case.case_id} for merchant {state.merchant.merchant_id}."]
        return {"audit_log": state.audit_log + log}
    return ingest_node


def make_retrieve_node(index: RetrievalIndex, top_k: int = 1) -> Callable[[PipelineState], dict]:
    def retrieve_node(state: PipelineState) -> dict:
        try:
            results = hybrid_search(index, state.dispute_reason_text, top_k=top_k)
        except ValueError as exc:
            msg = f"Retrieval failed: {exc}"
            return {"error": msg, "audit_log": state.audit_log + [msg]}

        if not results:
            msg = "Retrieval returned no matching reason code."
            return {"error": msg, "audit_log": state.audit_log + [msg]}

        top = results[0].doc
        missing = [e for e in top.required_evidence if e not in state.present_evidence]
        log = [f"Retrieved reason code {top.code} ({top.label}) — "
               f"fused_score={results[0].fused_score}, dense_used={index.dense_enabled}."]
        return {
            "retrieved_reason_code": top,
            "missing_evidence": missing,
            "audit_log": state.audit_log + log,
        }
    return retrieve_node


def make_score_node(scorer: EvidenceScorer) -> Callable[[PipelineState], dict]:
    def score_node(state: PipelineState) -> dict:
        if state.retrieved_reason_code is None:
            msg = "Skipped scoring: no reason code resolved."
            return {"audit_log": state.audit_log + [msg]}

        rc = state.retrieved_reason_code
        assessment = scorer.assess(
            reason_code=rc.code, reason_label=rc.label,
            required_evidence=rc.required_evidence,
            present_evidence=state.present_evidence,
            missing_evidence=state.missing_evidence,
        )
        log = [f"Evidence assessed: strength={assessment.strength}, "
               f"uncertainty={assessment.uncertainty}, source={assessment.source}."]
        return {"evidence_assessment": assessment, "audit_log": state.audit_log + log}
    return score_node


def make_decide_node(engine: DecisionEngine) -> Callable[[PipelineState], dict]:
    def decide_node(state: PipelineState) -> dict:
        if state.evidence_assessment is None:
            msg = "Skipped decision: no evidence assessment available."
            return {"audit_log": state.audit_log + [msg]}

        decision = engine.decide(state.case, state.merchant, state.evidence_assessment)
        log = [f"Decision: {decision.action.value} (confidence={decision.confidence_label})."]
        return {"decision": decision, "audit_log": state.audit_log + log}
    return decide_node


def make_counterfactual_node(engine: DecisionEngine, scorer: EvidenceScorer) -> Callable[[PipelineState], dict]:
    """Only meaningful for non-CONTEST decisions — for CONTEST there's nothing
    to explain a merchant into fixing, so this is a deliberate no-op there,
    not a missed case."""
    def counterfactual_node(state: PipelineState) -> dict:
        if state.decision is None or state.retrieved_reason_code is None:
            return {}
        if state.decision.action.value == "CONTEST":
            return {"audit_log": state.audit_log + ["Counterfactual skipped: already CONTEST."]}

        rc = state.retrieved_reason_code
        result = find_minimal_flip(
            engine=engine, scorer=scorer, case=state.case, merchant=state.merchant,
            reason_code=rc.code, reason_label=rc.label, required_evidence=rc.required_evidence,
            present_evidence=state.present_evidence, missing_evidence=state.missing_evidence,
            baseline_decision=state.decision,
        )
        return {"counterfactual": result, "audit_log": state.audit_log + [result.note]}
    return counterfactual_node


def make_log_node() -> Callable[[PipelineState], dict]:
    def log_node(state: PipelineState) -> dict:
        # Actual persistence to Postgres happens in app/db.py's caller, not here —
        # this node stays storage-agnostic so the graph itself has no DB dependency.
        return {"audit_log": state.audit_log + ["Pipeline run complete."]}
    return log_node