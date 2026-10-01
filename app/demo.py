"""
Demo data for a hosted instance.

When DEMO_SEED is on, the server loads a sample portfolio of resolved disputes
(the synthetic "portfolio" split, with planted process problems) the first time
it starts against an empty database, so the Insights page has something to show
to visitors who cannot run scripts. It uses rules-only extraction (no LLM
quota), runs in a background thread so startup and health checks are not
delayed, and is skipped whenever outcomes already exist.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import sessionmaker

from app import repository as repo
from app.catalog import DATA_DIR, Catalog
from app.db import OutcomeEnum, OutcomeRow, session_scope
from app.domain import DisputeCase
from app.engine.decision_engine import DecisionEngine
from app.scoring.win_model import WinModel
from app.service import DisputeService

logger = logging.getLogger("abstain.demo")
DEMO_FILE = DATA_DIR / "disputes_portfolio.jsonl"
BATCH = 50


def has_outcomes(sessions: sessionmaker) -> bool:
    with session_scope(sessions) as s:
        return bool(s.scalar(select(func.count()).select_from(OutcomeRow)))


def seed_demo_portfolio(sessions: sessionmaker, catalog: Catalog, model: WinModel, engine: DecisionEngine,
                        limit: int = 600, path: Path = DEMO_FILE) -> int:
    """Load up to `limit` resolved demo disputes. Returns how many were loaded (0 if data already exists)."""
    if has_outcomes(sessions):
        return 0
    service = DisputeService(catalog=catalog, model=model, engine=engine, llm=None, mode="rules")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()[:limit]]

    async def evaluate_all():
        return [await service.evaluate(DisputeCase(**{k: r[k] for k in DisputeCase.model_fields if k in r}))
                for r in rows]

    evaluations = asyncio.run(evaluate_all())
    for start in range(0, len(rows), BATCH):
        with session_scope(sessions) as s:
            for r, ev in zip(rows[start:start + BATCH], evaluations[start:start + BATCH], strict=True):
                repo.save_evaluation(s, ev, req_hash=repo.request_hash(ev.case), latency_ms=0,
                                     idempotency_key=f"demo-seed:{r['case_id']}")
                repo.record_outcome(s, r["case_id"], OutcomeEnum(r["outcome"]))
    logger.info("demo portfolio loaded", extra={"disputes": len(rows)})
    return len(rows)


def seed_in_background(sessions: sessionmaker, catalog: Catalog, model: WinModel, engine: DecisionEngine,
                       limit: int) -> None:
    def run():
        try:
            seed_demo_portfolio(sessions, catalog, model, engine, limit)
        except Exception:  # a failed demo load must never take the service down
            logger.exception("demo portfolio load failed")

    import threading
    threading.Thread(target=run, name="demo-seed", daemon=True).start()
