"""
pipeline.py

Wires the nodes into the LangGraph state machine:
ingest -> retrieve -> score -> decide -> counterfactual -> log -> END

Linear graph, deliberately no conditional branching at the graph level —
each node already handles its own "can't proceed" case internally (see
nodes.py) and simply passes state through with an error/skip note logged,
so the graph always completes and always produces an audit trail, whether
the run succeeded or not.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from app.engine.decision_engine import DecisionEngine
from app.engine.evidence_scorer import EvidenceScorer
from app.engine.models import DisputeCase, Merchant
from app.retrieval.index_builder import RetrievalIndex

from .nodes import (
    make_ingest_node, make_retrieve_node, make_score_node,
    make_decide_node, make_counterfactual_node, make_log_node,
)
from .state import PipelineState


def build_pipeline(index: RetrievalIndex, scorer: EvidenceScorer, engine: DecisionEngine):
    graph = StateGraph(PipelineState)

    graph.add_node("ingest", make_ingest_node())
    graph.add_node("retrieve", make_retrieve_node(index))
    graph.add_node("score", make_score_node(scorer))
    graph.add_node("decide", make_decide_node(engine))
    # Node id deliberately differs from the PipelineState field name
    # ("counterfactual") they both relate to — LangGraph treats node ids and
    # state keys as the same namespace and rejects the collision.
    graph.add_node("counterfactual_step", make_counterfactual_node(engine, scorer))
    graph.add_node("log", make_log_node())

    graph.set_entry_point("ingest")
    graph.add_edge("ingest", "retrieve")
    graph.add_edge("retrieve", "score")
    graph.add_edge("score", "decide")
    graph.add_edge("decide", "counterfactual_step")
    graph.add_edge("counterfactual_step", "log")
    graph.add_edge("log", END)
    graph.add_edge("log", END)

    return graph.compile()


def run_case(
    compiled_graph,
    case: DisputeCase,
    merchant: Merchant,
    present_evidence: list[str],
    dispute_reason_text: str,
) -> PipelineState:
    initial = PipelineState(
        case=case, merchant=merchant,
        present_evidence=present_evidence, dispute_reason_text=dispute_reason_text,
    )
    result = compiled_graph.invoke(initial)
    # LangGraph returns a plain dict from .invoke(); re-validate into our
    # typed state rather than trusting the dict shape blindly.
    return PipelineState(**result)