"""
Evaluation harness. Measures each stage separately so a bad number can be
traced to its cause:

  1. Extraction   per-fact accuracy against ground-truth facts
  2. Probability  Brier / log loss / ECE / AUC against outcomes, and mean
                  absolute error against the generator's true P(win)
  3. Policy       net rupees versus baselines, escalation rate, and a paired
                  bootstrap 95% CI for the difference versus contest-all

Policies are scored two ways. "Expected" uses the generator's true P(win)
(sum over contested cases of p*amount - cost), which removes coin-flip noise
from the comparison. "Realized" uses the sampled outcomes, like a real
backtest. Escalated cases are charged the review cost and then handled by a
reviewer who acts on the true P(win), an explicit best-case assumption for
the human, stated in the report.

Usage:
  python eval/run_eval.py                       # oracle + rules (no API calls)
  python eval/run_eval.py --llm --limit 120     # + LLM and cascade on the first 120 cases per split
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.catalog import load_catalog  # noqa: E402
from app.config import Settings  # noqa: E402
from app.domain import FACT_NAMES, Action, Tri  # noqa: E402
from app.engine.decision_engine import DecisionEngine  # noqa: E402
from app.extraction.cache import ResponseCache  # noqa: E402
from app.extraction.llm_client import OpenAICompatibleClient  # noqa: E402
from app.extraction.llm_extractor import ExtractionResult, LLMExtractor  # noqa: E402
from app.observability import configure_logging  # noqa: E402
from app.scoring.metrics import auc, brier, ece, log_loss  # noqa: E402
from app.scoring.win_model import WinModel  # noqa: E402
from app.service import DisputeService  # noqa: E402
from scripts._dataset import load_split  # noqa: E402

SPLITS = ("test", "test_shifted")


# ---------------------------------------------------------------- running

async def run_system(service: DisputeService, rows: list[dict], system: str, case_concurrency: int = 2) -> list[dict]:
    # Bounded case-level concurrency: launching every case at once would make
    # later cases queue behind the client's rate pacing for longer than any
    # sensible latency budget, and they would be scored as timeouts.
    gate = asyncio.Semaphore(case_concurrency)

    async def one(row):
        async with gate:
            case = row["case"]
            started = time.perf_counter()
            if system == "oracle":
                ev = service.assess(case, ExtractionResult(samples=[row["facts"]], source="oracle"))
            else:
                ev = await service.evaluate(case)
            return {"row": row, "ev": ev, "latency_s": time.perf_counter() - started}
    return await asyncio.gather(*(one(r) for r in rows))


def build_service(system: str, catalog, model, engine, settings: Settings, cache: ResponseCache | None):
    if system in ("oracle", "rules"):
        return DisputeService(catalog=catalog, model=model, engine=engine, llm=None, mode="rules"), None
    client = OpenAICompatibleClient(api_key=settings.llm_api_key, model=settings.llm_model,
                                    base_url=settings.llm_base_url, max_concurrency=settings.llm_max_concurrency,
                                    max_requests_per_minute=settings.llm_requests_per_minute,
                                    extra_body=settings.llm_extra_body,
                                    max_retries=8, backoff_cap_s=60, timeout_s=90, cache=cache)
    extractor = LLMExtractor(client, n_samples=settings.llm_samples, max_tokens=settings.llm_max_tokens)
    service = DisputeService(catalog=catalog, model=model, engine=engine, llm=extractor, mode=system,
                             llm_budget_s=None)  # offline benchmark: measure quality, not latency
    return service, client


# ---------------------------------------------------------------- metrics

def extraction_metrics(results: list[dict]) -> dict:
    ok, tot = defaultdict(int), defaultdict(int)
    known_ok = known_tot = all_ok = all_tot = 0
    for res in results:
        truth, pred = res["row"]["facts"], res["ev"].facts
        for f in FACT_NAMES:
            hit = truth.get(f) == pred.get(f)
            all_ok += hit
            all_tot += 1
            if truth.get(f) != Tri.UNKNOWN:
                ok[f] += hit
                tot[f] += 1
                known_ok += hit
                known_tot += 1
    ext = [r["ev"].extraction for r in results]
    return {
        "accuracy_known_facts": round(known_ok / known_tot, 4),
        "accuracy_all_facts": round(all_ok / all_tot, 4),
        "per_fact_known": {f: round(ok[f] / tot[f], 3) for f in FACT_NAMES if tot[f]},
        "llm_calls_per_case": round(sum(e.llm_calls for e in ext) / len(ext), 3),
        "cases_sent_to_llm": sum(e.llm_calls > 0 for e in ext),
        "failed_samples": sum(e.failed_samples for e in ext),
        "ungrounded_answers": sum(e.ungrounded_answers for e in ext),
        "tokens": sum(e.prompt_tokens + e.completion_tokens for e in ext),
        "degraded_runs": sum(r["ev"].degraded_reason is not None for r in results),
    }


def probability_metrics(results: list[dict]) -> dict:
    p = np.array([r["ev"].estimate.mean for r in results])
    y = np.array([1.0 if r["row"]["outcome"] == "won" else 0.0 for r in results])
    true_p = np.array([r["row"]["true_p_win"] for r in results])
    sd = np.array([r["ev"].estimate.std for r in results])
    # Is the reported uncertainty honest? A 90% interval should contain the true P(win) ~90% of the time.
    coverage = float(np.mean(np.abs(true_p - p) <= 1.645 * sd))
    return {"brier": round(brier(p, y), 4), "log_loss": round(log_loss(p, y), 4), "ece": round(ece(p, y), 4),
            "auc": round(auc(p, y), 4), "mae_vs_true_p": round(float(np.mean(np.abs(p - true_p))), 4),
            "mean_std": round(float(sd.mean()), 4), "coverage_90": round(coverage, 3)}


def case_values(action: Action, row: dict, cost: float, review_cost: float) -> tuple[float, float]:
    """(expected, realized) net rupees of taking `action` on this case."""
    amount, p = float(row["amount_inr"]), row["true_p_win"]
    won = row["outcome"] == "won"
    if action == Action.CONTEST:
        return p * amount - cost, (amount if won else 0.0) - cost
    if action == Action.ESCALATE:  # reviewer pays review cost, then acts on the true P(win)
        if p * amount - cost > 0:
            return p * amount - cost - review_cost, (amount if won else 0.0) - cost - review_cost
        return -review_cost, -review_cost
    return 0.0, 0.0


def policy_rows(results: list[dict], engine: DecisionEngine, catalog) -> dict[str, list[tuple[Action, float, float]]]:
    """Per policy: list of (action, expected, realized) per case."""
    cfg = engine.config
    out = defaultdict(list)
    for res in results:
        row, ev = res["row"], res["ev"]
        case, merchant = row["case"], catalog.merchant(row["merchant_id"])
        cost = engine.contest_cost(case, merchant)
        review = cfg.human_review_cost_inr * cfg.review_cost_multiplier[merchant.risk_tolerance]
        feasible = case.response_deadline_days_left >= cfg.min_days_to_respond
        naive = {
            "concede_all": Action.CONCEDE,
            "contest_all": Action.CONTEST if feasible else Action.CONCEDE,
            "abstain": ev.decision.action,
            "abstain_no_escalation": (ev.decision.action if ev.decision.action != Action.ESCALATE
                                      else (Action.CONTEST if ev.decision.ev_contest_inr > 0 else Action.CONCEDE)),
        }
        for name, action in naive.items():
            e, r = case_values(action, row, cost, review)
            out[name].append((action, e, r))
    return out


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, n: int = 2000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), (n, len(a)))
    diffs = (a[idx] - b[idx]).sum(axis=1)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def policy_metrics(results: list[dict], engine: DecisionEngine, catalog) -> dict:
    rows = policy_rows(results, engine, catalog)
    base_e = np.array([e for _, e, _ in rows["contest_all"]])
    base_r = np.array([r for _, _, r in rows["contest_all"]])
    out = {}
    for name, items in rows.items():
        acts = [a for a, _, _ in items]
        e = np.array([x for _, x, _ in items])
        r = np.array([x for _, _, x in items])
        lo, hi = bootstrap_diff_ci(e, base_e)
        rlo, rhi = bootstrap_diff_ci(r, base_r)
        out[name] = {
            "expected_net_inr": round(float(e.sum())), "realized_net_inr": round(float(r.sum())),
            "expected_vs_contest_all": round(float(e.sum() - base_e.sum())),
            "expected_vs_contest_all_ci95": [round(lo), round(hi)],
            "realized_vs_contest_all": round(float(r.sum() - base_r.sum())),
            "realized_vs_contest_all_ci95": [round(rlo), round(rhi)],
            "contest": acts.count(Action.CONTEST), "concede": acts.count(Action.CONCEDE),
            "escalate": acts.count(Action.ESCALATE),
        }
    # Oracle-optimal expected value: contest exactly when true EV > 0.
    best = 0.0
    for res in results:
        row = res["row"]
        m = catalog.merchant(row["merchant_id"])
        if row["case"].response_deadline_days_left >= engine.config.min_days_to_respond:
            best += max(0.0, row["true_p_win"] * row["amount_inr"] - engine.contest_cost(row["case"], m))
    out["_oracle_optimum_expected_net_inr"] = round(best)
    return out


def classification_metrics(actions: list[Action], results: list[dict], engine: DecisionEngine, catalog) -> dict:
    """The three-way decision scored as a binary classifier, as a risk team would read it:
    positive = CONTEST, ground truth = the dispute was won. ESCALATE is excluded from
    precision/recall (it is abstention, not a prediction) and reported as a rate.
    False-positive cost = contest cost spent on disputes that were lost; false-negative
    cost = amount left on the table by conceding disputes that were won."""
    tp = fp = tn = fn = esc = 0
    fp_cost = fn_cost = 0.0
    for action, res in zip(actions, results, strict=True):
        row = res["row"]
        won = row["outcome"] == "won"
        if action == Action.ESCALATE:
            esc += 1
        elif action == Action.CONTEST:
            if won:
                tp += 1
            else:
                fp += 1
                fp_cost += engine.contest_cost(row["case"], catalog.merchant(row["merchant_id"]))
        elif won:
            fn += 1
            fn_cost += float(row["amount_inr"])
        else:
            tn += 1
    return {"precision": round(tp / (tp + fp), 3) if tp + fp else None,
            "recall": round(tp / (tp + fn), 3) if tp + fn else None,
            "tp": tp, "fp": fp, "tn": tn, "fn": fn,
            "fp_cost_inr": round(fp_cost), "fn_cost_inr": round(fn_cost),
            "escalation_rate": round(esc / len(actions), 3)}


def boundary_sweep(results: list[dict], engine: DecisionEngine, catalog,
                   review_costs=(0, 100, 300, 600, 1000, 1e9)) -> list[dict]:
    """Decision-boundary experiment: the review cost is the knob that sets how readily the
    engine abstains. Re-decide every case (same estimates, no re-extraction) at each cost."""
    from dataclasses import replace

    out = []
    for cost in review_costs:
        e = DecisionEngine(replace(engine.config, human_review_cost_inr=cost))
        actions, expected = [], 0.0
        for res in results:
            ev, row = res["ev"], res["row"]
            m = catalog.merchant(row["merchant_id"])
            d = e.decide(row["case"], m, ev.estimate, ev.conflicts)
            actions.append(d.action)
            review = cost * e.config.review_cost_multiplier[m.risk_tolerance]
            expected += case_values(d.action, row, e.contest_cost(row["case"], m), review)[0]
        out.append({"review_cost_inr": "∞ (never escalate)" if cost >= 1e9 else int(cost),
                    **classification_metrics(actions, results, e, catalog), "expected_net_inr": round(expected)})
    return out


# ---------------------------------------------------------------- report

def render(report: dict) -> str:
    systems_run = {s for split in report["splits"].values() for s in split}
    llm = (f"LLM `{report['llm_model']}` × {report['llm_samples']} samples"
           if systems_run & {"llm", "cascade"} else "offline run: oracle and rules only, no LLM calls")
    L = ["# Abstain evaluation report", "",
         f"Win model `{report['model_version']}` · {llm} · generated by `eval/run_eval.py`.", "",
         "Splits: **test** uses the same phrasing family the rules were written against; "
         "**test_shifted** uses a held-out phrasing bank written after the rules were frozen.", ""]
    for split, systems in report["splits"].items():
        L += [f"## {split} ({next(iter(systems.values()))['n']} cases)", "",
              "### Extraction", "",
              "| system | acc (known facts) | acc (all facts) | cases → LLM | LLM calls/case | ungrounded | failed samples | degraded |",
              "|---|---|---|---|---|---|---|---|"]
        for s, m in systems.items():
            x = m["extraction"]
            L.append(f"| {s} | {x['accuracy_known_facts']:.3f} | {x['accuracy_all_facts']:.3f} | {x['cases_sent_to_llm']} | "
                     f"{x['llm_calls_per_case']} | {x['ungrounded_answers']} | {x['failed_samples']} | {x['degraded_runs']} |")
        L += ["", "### P(win) quality", "", "| system | Brier ↓ | log loss ↓ | ECE ↓ | AUC ↑ | MAE vs true P ↓ | mean σ | 90% interval coverage (target 0.90) |",
              "|---|---|---|---|---|---|---|---|"]
        for s, m in systems.items():
            p = m["probability"]
            L.append(f"| {s} | {p['brier']} | {p['log_loss']} | {p['ece']} | {p['auc']} | {p['mae_vs_true_p']} | "
                     f"{p['mean_std']} | {p['coverage_90']} |")
        L += ["", "### Policy economics (₹, vs contest-all; 95% paired bootstrap CI)", "",
              "| system · policy | expected net | Δ expected [CI] | realized net | Δ realized [CI] | contest / concede / escalate |",
              "|---|---|---|---|---|---|"]
        first = True
        for s, m in systems.items():
            for pname, v in m["policy"].items():
                if pname.startswith("_") or (pname in ("concede_all", "contest_all") and not first):
                    continue
                label = pname if pname in ("concede_all", "contest_all") else f"{s} · {pname}"
                L.append(f"| {label} | {v['expected_net_inr']:,} | {v['expected_vs_contest_all']:+,} "
                         f"[{v['expected_vs_contest_all_ci95'][0]:+,}, {v['expected_vs_contest_all_ci95'][1]:+,}] | "
                         f"{v['realized_net_inr']:,} | {v['realized_vs_contest_all']:+,} "
                         f"[{v['realized_vs_contest_all_ci95'][0]:+,}, {v['realized_vs_contest_all_ci95'][1]:+,}] | "
                         f"{v['contest']} / {v['concede']} / {v['escalate']} |")
            first = False
        opt = next(iter(systems.values()))["policy"]["_oracle_optimum_expected_net_inr"]
        L += ["", f"Oracle optimum (contest exactly when true EV > 0): expected net ₹{opt:,}.", "",
              "### Decisions as a classifier (positive = CONTEST, truth = dispute won; ESCALATE excluded)", "",
              "| system | precision | recall | TP / FP / TN / FN | FP cost | FN cost | escalation rate |",
              "|---|---|---|---|---|---|---|"]
        for s, m in systems.items():
            c = m["classification"]
            L.append(f"| {s} | {c['precision']} | {c['recall']} | {c['tp']} / {c['fp']} / {c['tn']} / {c['fn']} | "
                     f"₹{c['fp_cost_inr']:,} | ₹{c['fn_cost_inr']:,} | {c['escalation_rate']:.0%} |")
        for s, m in systems.items():
            L += ["", f"### Decision-boundary sweep: {s} (review cost controls how readily the engine abstains)", "",
                  "| review cost | escalation rate | precision | recall | FP cost | FN cost | expected net |",
                  "|---|---|---|---|---|---|---|"]
            for b in m["boundary_sweep"]:
                rc = b["review_cost_inr"] if isinstance(b["review_cost_inr"], str) else f"₹{b['review_cost_inr']:,}"
                L.append(f"| {rc} | {b['escalation_rate']:.0%} | {b['precision']} | {b['recall']} | "
                         f"₹{b['fp_cost_inr']:,} | ₹{b['fn_cost_inr']:,} | ₹{b['expected_net_inr']:,} |")
        L.append("")
    L += ["## Assumptions", "",
          "- Escalated cases pay the review cost; the reviewer then acts on the true P(win). This is a best-case "
          "human, so `abstain_no_escalation` (same system, escalations replaced by the EV-sign decision) is "
          "reported alongside to show what escalation itself contributes.",
          "- Cases with fewer days left than the feasibility minimum are conceded by every policy, including contest-all.",
          "- The win model is trained on ground-truth facts from the train split; extraction errors at test time "
          "propagate into P(win), which is what the MAE-vs-true-P column isolates.", ""]
    return "\n".join(L)


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm", action="store_true", help="also evaluate llm and cascade systems (uses API quota)")
    ap.add_argument("--limit", type=int, default=None, help="cases per split")
    ap.add_argument("--systems", default=None, help="comma list overriding the default system set")
    ap.add_argument("--report", default="REPORT", help="output name: eval/<report>.md and eval/<report>.json")
    args = ap.parse_args()

    configure_logging("WARNING")
    settings = Settings()
    catalog, model, engine = load_catalog(), WinModel.load(), DecisionEngine()
    systems = ["oracle", "rules"] + (["cascade", "llm"] if args.llm else [])
    if args.systems:
        systems = args.systems.split(",")
    if any(s in ("llm", "cascade") for s in systems) and not settings.llm_api_key:
        sys.exit("GROQ_API_KEY is required for llm/cascade systems")
    cache = ResponseCache(ROOT / ".cache" / "llm_eval.sqlite")

    report = {"model_version": model.version, "llm_model": settings.llm_model,
              "llm_samples": settings.llm_samples, "splits": {}}
    for split in SPLITS:
        rows = load_split(split)[: args.limit]
        report["splits"][split] = {}
        for system in systems:
            service, client = build_service(system, catalog, model, engine, settings, cache)
            started = time.perf_counter()
            results = await run_system(service, rows, system)
            if client:
                await client.aclose()
            report["splits"][split][system] = {
                "n": len(rows),
                "wall_s": round(time.perf_counter() - started, 1),
                "extraction": extraction_metrics(results),
                "probability": probability_metrics(results),
                "policy": policy_metrics(results, engine, catalog),
                "classification": classification_metrics([r["ev"].decision.action for r in results], results,
                                                         engine, catalog),
                "boundary_sweep": boundary_sweep(results, engine, catalog),
            }
            print(f"[{split}] {system}: done in {report['splits'][split][system]['wall_s']}s", flush=True)

    out = ROOT / "eval"
    (out / f"{args.report}.json").write_text(json.dumps(report, indent=1), encoding="utf-8", newline="\n")
    (out / f"{args.report}.md").write_text(render(report), encoding="utf-8", newline="\n")
    print(f"wrote {out / (args.report + '.md')}")


if __name__ == "__main__":
    asyncio.run(main())
