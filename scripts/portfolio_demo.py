"""
End-to-end check of the merchant and portfolio diagnosis (levels 2 and 3).

The "portfolio" split has planted process problems (see generate_dataset.py):
mch_04 rarely gets the cardholder's signature, mch_06 does not offer refunds,
and every merchant's evidence pipeline drops account history. This script
runs the real HTTP API in-process on a throwaway SQLite database:

    POST /v1/disputes/evaluate  for every case
    POST /v1/disputes/{id}/outcome  with the recorded outcome
    GET  /v1/reports/portfolio and /v1/reports/merchants/{id}

then checks the reports recovered exactly the planted problems and nothing
else, and writes eval/PORTFOLIO_REPORT.md.

Usage: python scripts/portfolio_demo.py
"""
from __future__ import annotations

import dataclasses
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from scripts._dataset import load_split  # noqa: E402
from scripts.generate_dataset import PLANTED_MERCHANT_GAPS, PLANTED_SYSTEMIC_GAP  # noqa: E402


def main() -> int:
    rows = load_split("portfolio")
    with tempfile.TemporaryDirectory() as tmp:
        settings = dataclasses.replace(Settings(), database_url=f"sqlite:///{Path(tmp) / 'demo.db'}",
                                       llm_api_key=None, api_key=None, log_level="WARNING")
        with TestClient(create_app(settings, create_schema=True)) as client:
            for r in rows:
                body = r["case"].model_dump(mode="json")
                assert client.post("/v1/disputes/evaluate", json=body).status_code == 201
                assert client.post(f"/v1/disputes/{r['case_id']}/outcome",
                                   json={"outcome": r["outcome"]}).status_code == 200
            portfolio = client.get("/v1/reports/portfolio").json()
            merchant_reports = {m: client.get(f"/v1/reports/merchants/{m}").json() for m in PLANTED_MERCHANT_GAPS}

    weak = portfolio["weaknesses"]
    found_specific = {m: [x["fact"] for x in v["merchant_specific_weaknesses"]] for m, v in weak["merchants"].items()}
    false_alarms = {m: fs for m, fs in found_specific.items() if fs and m not in PLANTED_MERCHANT_GAPS}
    checks = [
        ("systemic gap recovered", weak["systemic"] == [PLANTED_SYSTEMIC_GAP], weak["systemic"]),
        *[(f"{m}: planted `{f}` flagged as merchant-specific", found_specific.get(m) == [f], found_specific.get(m))
          for m, f in PLANTED_MERCHANT_GAPS.items()],
        ("no merchant-specific false alarms elsewhere", not false_alarms, false_alarms or "none"),
    ]

    lines = ["# Portfolio diagnosis: planted-problem check", "",
             f"{len(rows)} disputes from the `portfolio` split were evaluated and resolved through the HTTP API "
             "(`scripts/portfolio_demo.py`). The split has planted process problems; the reports must find them "
             "and nothing else.", "",
             "| check | result | detected |", "|---|---|---|"]
    lines += [f"| {name} | {'PASS' if ok else 'FAIL'} | `{detail}` |" for name, ok, detail in checks]
    lines += ["", "---", "", portfolio["markdown"], ""]
    for rep in merchant_reports.values():
        lines += ["---", "", rep["markdown"], ""]
    out = ROOT / "eval" / "PORTFOLIO_REPORT.md"
    out.write_text("\n".join(lines), encoding="utf-8", newline="\n")

    for name, ok, detail in checks:
        print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    print(f"wrote {out}")
    return 0 if all(ok for _, ok, _ in checks) else 1


if __name__ == "__main__":
    sys.exit(main())
