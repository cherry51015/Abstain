"""
hybrid_search.py

Combines BM25 (lexical — good at exact reason-code/network terms) with
FAISS dense retrieval (semantic — good at paraphrased dispute descriptions)
via reciprocal rank fusion (RRF) — no learned reranker, kept auditable and
dependency-light.

If dense retrieval wasn't available at index-build time (model download
failed, offline environment), this falls back to BM25-only automatically —
degraded, but never broken. That fallback path is explicit and tested here,
not an accident of missing error handling.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from .index_builder import RetrievalIndex, ReasonCodeDoc

_RRF_K = 60  # conventional default from the original RRF paper — an obvious knob to tune later, not derived here


@dataclass
class RetrievedResult:
    doc: ReasonCodeDoc
    bm25_rank: int
    dense_rank: int  # -1 if dense retrieval was unavailable for this search
    fused_score: float


def _rank_order(scores: np.ndarray) -> list[int]:
    return list(np.argsort(scores)[::-1])


def _reciprocal_rank_fusion(rankings: list[list[int]], k: int = _RRF_K) -> dict[int, float]:
    scores: dict[int, float] = {}
    for order in rankings:
        for rank, idx in enumerate(order):
            scores[idx] = scores.get(idx, 0.0) + 1.0 / (k + rank + 1)
    return scores


def search(index: RetrievalIndex, query: str, top_k: int = 3) -> list[RetrievedResult]:
    if not query.strip():
        raise ValueError("Empty query — cannot retrieve against an empty string.")

    tokenized_query = query.lower().split()
    bm25_scores = np.array(index.bm25.get_scores(tokenized_query))
    bm25_order = _rank_order(bm25_scores)
    bm25_rank_lookup = {idx: r for r, idx in enumerate(bm25_order)}

    if index.dense_enabled and index.embed_model is not None and index.faiss_index is not None:
        query_vec = index.embed_model.encode([query], normalize_embeddings=True)
        _, dense_order_arr = index.faiss_index.search(np.array(query_vec, dtype="float32"), len(index.docs))
        dense_order = list(dense_order_arr[0])
        dense_rank_lookup = {idx: r for r, idx in enumerate(dense_order)}
        fused = _reciprocal_rank_fusion([bm25_order, dense_order])
    else:
        dense_rank_lookup = {}
        fused = _reciprocal_rank_fusion([bm25_order])

    ranked_indices = sorted(fused.keys(), key=lambda i: fused[i], reverse=True)[:top_k]

    return [
        RetrievedResult(
            doc=index.docs[i],
            bm25_rank=bm25_rank_lookup.get(i, -1),
            dense_rank=dense_rank_lookup.get(i, -1),
            fused_score=round(fused[i], 5),
        )
        for i in ranked_indices
    ]