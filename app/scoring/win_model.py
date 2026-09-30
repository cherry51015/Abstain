"""
Win-probability model: extracted facts -> P(win), with uncertainty.

One L2 logistic regression per dispute category over that category's
relevant facts (yes/no indicators) plus the merchant's track record. It is
trained offline (scripts/train_win_model.py) as a bootstrap ensemble and
shipped as plain JSON coefficients: inference is numpy-only, the artifact is
diffable and reviewable, and nothing is unpickled at startup.

Uncertainty is Monte Carlo over two sources:
  - extraction: each self-consistency sample of facts is one draw
  - model: each bootstrap member is one draw
so P(win) is reported as a mean and a standard deviation over the
(extraction sample x ensemble member) grid, rather than a self-reported
LLM confidence.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from app.domain import Category, EvidenceFacts, Merchant, Tri

DEFAULT_MODEL_PATH = Path(__file__).resolve().parents[2] / "models" / "win_model.json"


def feature_names(relevant_facts: list[str]) -> list[str]:
    names = []
    for f in relevant_facts:
        names += [f"{f}=yes", f"{f}=no"]
    return names + ["merchant_win_rate_centered"]


def featurize(facts: EvidenceFacts, relevant_facts: list[str], merchant_win_rate: float) -> np.ndarray:
    x = []
    for f in relevant_facts:
        v = facts.get(f)
        x += [1.0 if v == Tri.YES else 0.0, 1.0 if v == Tri.NO else 0.0]
    x.append(merchant_win_rate - 0.5)
    return np.asarray(x, dtype=float)


@dataclass(frozen=True)
class WinEstimate:
    mean: float
    std: float
    n_draws: int


@dataclass(frozen=True)
class CategoryModel:
    relevant_facts: list[str]
    feature_names: list[str]
    intercepts: np.ndarray      # (members,)
    coefs: np.ndarray           # (members, n_features)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (n_samples, n_features) -> probabilities (n_samples, members)."""
        z = X @ self.coefs.T + self.intercepts
        return 1.0 / (1.0 + np.exp(-z))


class WinModel:
    def __init__(self, categories: dict[Category, CategoryModel], version: str, metadata: dict,
                 fact_priors: dict[str, dict[str, float]] | None = None):
        self.categories = categories
        self.version = version
        self.metadata = metadata
        self.fact_priors = fact_priors or {}

    @classmethod
    def load(cls, path: Path = DEFAULT_MODEL_PATH) -> WinModel:
        raw = json.loads(Path(path).read_text())
        cats = {}
        for name, c in raw["categories"].items():
            cats[Category(name)] = CategoryModel(
                relevant_facts=c["relevant_facts"],
                feature_names=c["feature_names"],
                intercepts=np.asarray(c["intercepts"], dtype=float),
                coefs=np.asarray(c["coefs"], dtype=float),
            )
        return cls(cats, raw["version"], raw.get("metadata", {}), raw.get("fact_priors"))

    def impute(self, samples: list[EvidenceFacts], unread: list[str], seed: int, draws: int = 16) -> list[EvidenceFacts]:
        """Replace facts a document should cover but the extractor could not
        read with draws from the training priors: addressed with probability
        p_addressed, and then yes with probability p_yes. Uncertainty about
        what the document said then shows up in the P(win) spread instead of
        being silently treated as 'unknown'."""
        if not unread:
            return samples
        rng = np.random.default_rng(seed)
        out = []
        for s in samples:
            for _ in range(draws):
                update = {}
                for f in unread:
                    prior = self.fact_priors.get(f, {"p_addressed": 0.5, "p_yes": 0.5})
                    if rng.random() < prior["p_addressed"]:
                        update[f] = Tri.YES if rng.random() < prior["p_yes"] else Tri.NO
                out.append(s.model_copy(update=update))
        return out

    def estimate(self, category: Category, fact_samples: list[EvidenceFacts], merchant: Merchant) -> WinEstimate:
        if not fact_samples:
            raise ValueError("estimate() needs at least one fact sample")
        m = self.categories[category]
        X = np.stack([featurize(f, m.relevant_facts, merchant.historical_win_rate) for f in fact_samples])
        p = m.predict(X).ravel()
        return WinEstimate(mean=float(p.mean()), std=float(p.std()), n_draws=int(p.size))

    def contributions(self, category: Category, facts: EvidenceFacts, merchant: Merchant) -> dict[str, float]:
        """Per-feature log-odds contribution (ensemble mean), for explanations."""
        m = self.categories[category]
        x = featurize(facts, m.relevant_facts, merchant.historical_win_rate)
        mean_coef = m.coefs.mean(axis=0)
        return {n: round(float(c * v), 3) for n, c, v in zip(m.feature_names, mean_coef, x, strict=False) if v != 0.0}
