"""Reference data: reason codes and merchants, loaded once and validated.

Reason codes arrive structured from the card network, so this is a lookup,
not a retrieval problem.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.domain import FACT_NAMES, Merchant, ReasonCode

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass(frozen=True)
class Catalog:
    reason_codes: dict[str, ReasonCode]
    merchants: dict[str, Merchant]

    def reason_code(self, code: str) -> ReasonCode:
        try:
            return self.reason_codes[code]
        except KeyError:
            raise UnknownReferenceError(f"Unknown reason code {code!r}") from None

    def merchant(self, merchant_id: str) -> Merchant:
        try:
            return self.merchants[merchant_id]
        except KeyError:
            raise UnknownReferenceError(f"Unknown merchant_id {merchant_id!r}") from None


class UnknownReferenceError(LookupError):
    pass


def load_catalog(data_dir: Path = DATA_DIR) -> Catalog:
    reason_codes = [ReasonCode(**r) for r in json.loads((data_dir / "reason_codes.json").read_text())]
    for rc in reason_codes:
        unknown = set(rc.relevant_facts) - set(FACT_NAMES)
        if unknown:
            raise ValueError(f"Reason code {rc.code} references undefined facts: {sorted(unknown)}")
    merchants = [Merchant(**m) for m in json.loads((data_dir / "merchants.json").read_text())]
    return Catalog(
        reason_codes={rc.code: rc for rc in reason_codes},
        merchants={m.merchant_id: m for m in merchants},
    )
