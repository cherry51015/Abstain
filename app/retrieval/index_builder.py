"""
index_builder.py

Builds the hybrid retrieval index over reason_codes.json once at startup.
Both indices point at the same underlying corpus (one text blob per reason
code: label + notes + evidence requirement names), so BM25 and FAISS are
scoring the exact same documents, just with different signal — lexical vs.
semantic. This is what lets a free-text dispute description ("customer says
the item never showed up") resolve to the right structured reason code
without requiring the raw data to already carry a clean code field.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

try:
    import faiss
    from sentence_transformers import SentenceTransformer
    _DENSE_AVAILABLE = True
except ImportError:
    _DENSE_AVAILABLE = False

_EMBED_MODEL_NAME = "all-MiniLM-L6-v2"  # small, open-source, runs locally — no API call needed


@dataclass
class ReasonCodeDoc:
    code: str
    network: str
    label: str
    required_evidence: list[str]
    base_win_rate_range: tuple[float, float]
    notes: str
    text: str  # the blob actually indexed


@dataclass
class RetrievalIndex:
    docs: list[ReasonCodeDoc]
    bm25: BM25Okapi
    embed_model: object | None      # SentenceTransformer, or None if dense retrieval unavailable
    faiss_index: object | None      # faiss.Index, or None if dense retrieval unavailable
    dense_enabled: bool


def _doc_text(entry: dict) -> str:
    return f"{entry['label']}. {entry['notes']} Evidence needed: {', '.join(entry['required_evidence'])}."


def build_index(reason_codes_path: str | Path, enable_dense: bool = True) -> RetrievalIndex:
    with open(reason_codes_path) as f:
        raw = json.load(f)

    if not raw:
        raise ValueError(f"No reason codes found in {reason_codes_path} — cannot build an empty index.")

    docs = [
        ReasonCodeDoc(
            code=e["code"], network=e["network"], label=e["label"],
            required_evidence=e["required_evidence"],
            base_win_rate_range=tuple(e["base_win_rate_range"]),
            notes=e["notes"], text=_doc_text(e),
        )
        for e in raw
    ]

    tokenized = [d.text.lower().split() for d in docs]
    bm25 = BM25Okapi(tokenized)

    embed_model = None
    faiss_index = None
    dense_enabled = False

    if enable_dense and _DENSE_AVAILABLE:
        try:
            embed_model = SentenceTransformer(_EMBED_MODEL_NAME)
            embeddings = embed_model.encode([d.text for d in docs], normalize_embeddings=True)
            dim = embeddings.shape[1]
            faiss_index = faiss.IndexFlatIP(dim)  # inner product on normalized vecs = cosine similarity
            faiss_index.add(np.array(embeddings, dtype="float32"))
            dense_enabled = True
        except Exception:
            # Model download can fail offline / on a restricted network — degrade to
            # BM25-only rather than crashing index build. hybrid_search.py checks
            # dense_enabled and falls back accordingly, so this is a real, tested path,
            # not a silent guess.
            embed_model, faiss_index, dense_enabled = None, None, False

    return RetrievalIndex(docs=docs, bm25=bm25, embed_model=embed_model,
                           faiss_index=faiss_index, dense_enabled=dense_enabled)