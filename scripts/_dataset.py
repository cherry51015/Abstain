"""Shared loader for the generated JSONL splits (used by training and eval)."""
from __future__ import annotations

import json
from pathlib import Path

from app.domain import DisputeCase, EvidenceFacts

DATA = Path(__file__).resolve().parent.parent / "data"


def load_split(split: str) -> list[dict]:
    rows = []
    for line in (DATA / f"disputes_{split}.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        row["case"] = DisputeCase(**{k: row[k] for k in DisputeCase.model_fields if k in row})
        row["facts"] = EvidenceFacts(**row["true_facts"])
        rows.append(row)
    return rows
