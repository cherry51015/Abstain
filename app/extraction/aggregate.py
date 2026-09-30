"""Combine self-consistency samples of extracted facts."""
from __future__ import annotations

from collections import Counter

from app.domain import FACT_NAMES, EvidenceFacts, Tri


def majority(samples: list[EvidenceFacts]) -> EvidenceFacts:
    """Per-fact majority vote; ties resolve to 'unknown' rather than guessing."""
    voted = {}
    for name in FACT_NAMES:
        counts = Counter(s.get(name) for s in samples).most_common()
        top = counts[0]
        tied = len(counts) > 1 and counts[1][1] == top[1]
        voted[name] = Tri.UNKNOWN if tied else top[0]
    return EvidenceFacts(**voted)


def agreement(samples: list[EvidenceFacts]) -> dict[str, float]:
    """Fraction of samples agreeing with the majority, per fact."""
    maj = majority(samples)
    return {n: round(sum(s.get(n) == maj.get(n) for s in samples) / len(samples), 3) for n in FACT_NAMES}
