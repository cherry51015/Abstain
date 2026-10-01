"""
Load a demo portfolio into a running deployment's database so the Insights
page has something to show.

Evaluates disputes from the "portfolio" split (which has planted process
problems, see generate_dataset.py) through the real API code in-process, in
rules-only mode so it spends no LLM quota, and records each dispute's final
outcome. Safe to re-run: every request carries an Idempotency-Key, so cases
already loaded are replayed instead of duplicated.

Usage:
  python scripts/seed_demo.py                                  # uses DATABASE_URL (migrate it first)
  python scripts/seed_demo.py --database-url postgresql://...  # e.g. a hosted demo database
"""
from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from fastapi.testclient import TestClient  # noqa: E402

from app.config import Settings  # noqa: E402
from app.main import create_app  # noqa: E402
from scripts._dataset import load_split  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--database-url", default=None)
    ap.add_argument("--limit", type=int, default=600)
    args = ap.parse_args()

    base = Settings()
    settings = dataclasses.replace(base, database_url=args.database_url or base.database_url,
                                   llm_api_key=None, api_key=None, log_level="WARNING")
    rows = load_split("portfolio")[: args.limit]
    with TestClient(create_app(settings)) as client:
        for i, r in enumerate(rows, 1):
            resp = client.post("/v1/disputes/evaluate", json=r["case"].model_dump(mode="json"),
                               headers={"Idempotency-Key": f"seed:{r['case_id']}"})
            resp.raise_for_status()
            client.post(f"/v1/disputes/{r['case_id']}/outcome", json={"outcome": r["outcome"]}).raise_for_status()
            if i % 100 == 0:
                print(f"{i}/{len(rows)} disputes loaded", flush=True)
    print(f"done: {len(rows)} resolved disputes in {settings.database_url}")


if __name__ == "__main__":
    main()
