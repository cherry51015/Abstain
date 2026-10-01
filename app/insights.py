"""
Aggregations over resolved disputes (decision + recorded network outcome).
Three levels of diagnosis:

  1. dispute    the decision and its counterfactual (engine/counterfactual.py)
  2. merchant   "why does THIS merchant keep losing?" Its primary weakness,
                benchmarked against the rest of the portfolio, because a raw
                rate cannot tell a merchant problem from a portfolio one
  3. portfolio  "why do we keep losing?" A weakness shared by several
                separate merchants is systemic: fix the shared evidence
                pipeline once instead of chasing merchants one by one

Also:
  - root causes: which relevant facts were missing (no document) versus
    adverse (document says 'no') among lost disputes. Missing evidence is a
    collection problem; adverse evidence is an operations problem.
  - calibration monitoring: the live Brier score / ECE of P(win) against
    outcomes, so model drift is visible without re-running offline evals.
  - realized economics of the decisions actually taken.
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict

import numpy as np

from app.catalog import Catalog
from app.db import ActionEnum, DisputeRow, EvaluationRow, OutcomeEnum, OutcomeRow
from app.domain import fact_applicable
from app.scoring.metrics import brier, ece, reliability_bins

Resolved = tuple[EvaluationRow, OutcomeRow, DisputeRow]
ECE_ALERT = 0.10
MIN_CASES_FOR_ALERT = 50
MIN_CASES = 8               # disputes on a fact before a merchant rate is trusted
MIN_GAP_DIFF = 0.15         # merchant-specific: this much above the rest of the portfolio ...
FDR = 0.05                  # ... and significant after Benjamini-Hochberg across every merchant x fact test
MIN_CASES_SYSTEMIC = 5
SYSTEMIC_MERCHANTS = 3      # systemic: missing in most disputes of at least this many merchants ...
MIN_LOSS_LIFT = 0.15        # ... and missing it raises the loss rate by at least this much


def _final_action(ev: EvaluationRow) -> ActionEnum:
    return ev.review.action if ev.review is not None else ev.action


def root_causes(rows: list[Resolved], catalog: Catalog) -> dict:
    missing, adverse = Counter(), Counter()
    per_merchant: dict[str, dict] = defaultdict(lambda: {"resolved": 0, "lost": 0, "amount_lost_inr": 0.0,
                                                          "missing": Counter(), "adverse": Counter()})
    for ev, out, dispute in rows:
        m = per_merchant[dispute.merchant_id]
        m["resolved"] += 1
        if out.outcome != OutcomeEnum.lost:
            continue
        m["lost"] += 1
        m["amount_lost_inr"] += float(dispute.amount_inr)
        for fact in catalog.reason_code(dispute.reason_code).relevant_facts:
            value = ev.facts.get(fact, "unknown")
            if value == "unknown":
                missing[fact] += 1
                m["missing"][fact] += 1
            elif value == "no":
                adverse[fact] += 1
                m["adverse"][fact] += 1

    merchants = {}
    for mid, m in sorted(per_merchant.items()):
        profile = catalog.merchants.get(mid)
        merchants[mid] = {
            "resolved": m["resolved"], "lost": m["lost"],
            "loss_rate": round(m["lost"] / m["resolved"], 3) if m["resolved"] else None,
            "amount_lost_inr": round(m["amount_lost_inr"], 2),
            "top_missing": m["missing"].most_common(3), "top_adverse": m["adverse"].most_common(3),
            # Winning a representment does not remove a dispute from the network's
            # ratio, so a merchant near its threshold needs prevention, not more contests.
            "threshold_proximity": round(profile.threshold_proximity, 3) if profile else None,
        }
    return {"resolved": len(rows), "lost": sum(o.outcome == OutcomeEnum.lost for _, o, _ in rows),
            "top_missing": missing.most_common(5), "top_adverse": adverse.most_common(5), "merchants": merchants}


def _two_proportion_z(k1: int, n1: int, k2: int, n2: int) -> float:
    if n1 == 0 or n2 == 0:
        return 0.0
    pooled = (k1 + k2) / (n1 + n2)
    se = math.sqrt(pooled * (1 - pooled) * (1 / n1 + 1 / n2))
    return 0.0 if se == 0 else (k1 / n1 - k2 / n2) / se


def weaknesses(rows: list[Resolved], catalog: Catalog) -> dict:
    """Level 2 and 3 of the diagnosis.

    Rates are measured over *all* resolved disputes where a fact is relevant,
    not only lost ones: conditioning on losing makes every fact look weak,
    because losing cases are weak cases. For each merchant and fact:

      merchant-specific  the fact is weak (missing or adverse) in a clearly
                         larger share of this merchant's disputes than of the
                         rest of the portfolio's: difference >= MIN_GAP_DIFF,
                         n >= MIN_CASES, and a one-sided two-proportion z-test
                         that survives Benjamini-Hochberg at FDR across *all*
                         merchant x fact tests. (8 merchants x 9 facts is ~70
                         tests; at a plain 5% threshold a few would be flagged
                         by chance alone, which is exactly what happened before
                         the correction was added.)
      worth watching     the same direction and size of gap, but not yet enough
                         evidence (too few disputes or not significant after
                         correction). Shown, never acted on automatically.
      systemic           the fact is *missing* in most disputes of at least
                         SYSTEMIC_MERCHANTS separate merchants AND missing it
                         measurably costs wins (loss rate when missing exceeds
                         loss rate when present by >= MIN_LOSS_LIFT). That is
                         the shared evidence pipeline, not one merchant's process.

    "k of n losses" (how concentrated a weakness is among a merchant's losses)
    is reported for context but is not used to classify.
    """
    stats: dict[str, dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))
    outcome_by_value: dict[str, Counter] = defaultdict(Counter)   # fact -> (value, lost) counts
    losses_by_merchant: Counter = Counter()
    for ev, out, dispute in rows:
        lost = out.outcome == OutcomeEnum.lost
        losses_by_merchant[dispute.merchant_id] += lost
        relevant = catalog.reason_code(dispute.reason_code).relevant_facts
        for fact in relevant:
            if not fact_applicable(fact, ev.facts, relevant):
                continue  # e.g. no signature is possible on a parcel that was never delivered
            value = ev.facts.get(fact, "unknown")
            c = stats[fact][dispute.merchant_id]
            c["n"] += 1
            c["weak"] += value != "yes"
            c["missing"] += value == "unknown"
            c["losses"] += lost
            c["weak_in_losses"] += lost and value != "yes"
            outcome_by_value[fact][(value, lost)] += 1

    def loss_rate(fact: str, value: str) -> tuple[float | None, int]:
        o = outcome_by_value[fact]
        n = o[(value, True)] + o[(value, False)]
        return (o[(value, True)] / n if n else None), n

    facts, per_merchant, tests = {}, defaultdict(list), []
    for fact, by_m in stats.items():
        W = sum(c["weak"] for c in by_m.values())
        N = sum(c["n"] for c in by_m.values())
        miss_lr, n_miss = loss_rate(fact, "unknown")
        yes_lr, n_yes = loss_rate(fact, "yes")
        lift = (miss_lr - yes_lr) if (miss_lr is not None and yes_lr is not None
                                      and min(n_miss, n_yes) >= MIN_CASES) else None
        affected = sorted(m for m, c in by_m.items() if c["n"] >= MIN_CASES_SYSTEMIC and c["missing"] / c["n"] >= 0.5)
        systemic = len(affected) >= SYSTEMIC_MERCHANTS and lift is not None and lift >= MIN_LOSS_LIFT
        facts[fact] = {"weak_rate": round(W / N, 3), "missing_rate": round(sum(c["missing"] for c in by_m.values()) / N, 3),
                       "loss_rate_when_missing": None if miss_lr is None else round(miss_lr, 3),
                       "loss_rate_when_present": None if yes_lr is None else round(yes_lr, 3),
                       "merchants_affected": affected, "systemic": systemic}
        for m, c in by_m.items():
            rest_w, rest_n = W - c["weak"], N - c["n"]
            rate, rest_rate = c["weak"] / c["n"], (rest_w / rest_n if rest_n else None)
            z = _two_proportion_z(c["weak"], c["n"], rest_w, rest_n)
            big_gap = rest_rate is not None and rate - rest_rate >= MIN_GAP_DIFF
            item = {
                "fact": fact, "disputes": c["n"], "weak_rate": round(rate, 3),
                "rest_of_portfolio_rate": None if rest_rate is None else round(rest_rate, 3), "z": round(z, 2),
                "p_value": round(0.5 * math.erfc(z / math.sqrt(2)), 5),   # one-sided: merchant worse than the rest
                "weak_in_losses": c["weak_in_losses"], "relevant_losses": c["losses"],
                "merchant_specific": False, "watch": big_gap, "systemic": systemic,
            }
            per_merchant[m].append(item)
            if c["n"] >= MIN_CASES and rest_rate is not None:
                tests.append(item)

    # Benjamini-Hochberg: the largest k with p_(k) <= k/m * FDR; the k smallest p-values are discoveries.
    tests.sort(key=lambda x: x["p_value"])
    cutoff = max((k for k, t in enumerate(tests, 1) if t["p_value"] <= k / len(tests) * FDR), default=0)
    for t in tests[:cutoff]:
        if t["watch"]:
            t["merchant_specific"], t["watch"] = True, False

    merchants = {}
    for m, items in per_merchant.items():
        specific = sorted((x for x in items if x["merchant_specific"]),
                          key=lambda x: -(x["weak_rate"] - x["rest_of_portfolio_rate"]))
        systemic_items = [x for x in items if x["systemic"]]
        watch = sorted((x for x in items if x["watch"] and not x["systemic"]),
                       key=lambda x: -(x["weak_rate"] - x["rest_of_portfolio_rate"]))
        primary = (specific or systemic_items or watch
                   or sorted(items, key=lambda x: (-x["weak_in_losses"], -x["weak_rate"])))[0]
        merchants[m] = {
            "losses": losses_by_merchant[m],
            "primary_weakness": primary,
            "classification": ("merchant-specific" if primary["merchant_specific"]
                               else "systemic" if primary["systemic"]
                               else "worth watching" if primary["watch"] else "in line with portfolio"),
            "merchant_specific_weaknesses": specific,
            "worth_watching": watch,
        }
    return {"facts": facts, "systemic": sorted(f for f, v in facts.items() if v["systemic"]),
            "merchants": dict(sorted(merchants.items()))}


def calibration(rows: list[Resolved]) -> dict:
    if not rows:
        return {"n": 0, "brier": None, "ece": None, "bins": [], "alert": False}
    p = np.array([ev.p_win for ev, _, _ in rows])
    y = np.array([1.0 if o.outcome == OutcomeEnum.won else 0.0 for _, o, _ in rows])
    e = ece(p, y)
    return {"n": len(rows), "brier": round(brier(p, y), 4), "ece": round(e, 4),
            "bins": reliability_bins(p, y), "alert": len(rows) >= MIN_CASES_FOR_ALERT and e > ECE_ALERT}


def realized_economics(rows: list[Resolved]) -> dict:
    net, contested, won_contested = 0.0, 0, 0
    for ev, out, dispute in rows:
        if _final_action(ev) != ActionEnum.CONTEST:
            continue
        contested += 1
        won = out.outcome == OutcomeEnum.won
        won_contested += won
        net += (float(dispute.amount_inr) if won else 0.0) - float(ev.contest_cost_inr)
    return {"contested": contested, "won": won_contested,
            "win_rate_when_contested": round(won_contested / contested, 3) if contested else None,
            "net_recovered_inr": round(net, 2)}


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{x:.0%}"


def _render_weaknesses(weak: dict, merchant_id: str | None) -> list[str]:
    if merchant_id:
        m = weak["merchants"].get(merchant_id)
        if not m:
            return ["## Primary weakness", "", "- No lost disputes for this merchant yet.", ""]
        p = m["primary_weakness"]
        L = ["## Primary weakness", "",
             f"- **`{p['fact']}`** was weak (missing or adverse) in **{p['weak_in_losses']} of {p['relevant_losses']}** "
             f"losses where it mattered.",
             f"- Weak in {_pct(p['weak_rate'])} of this merchant's disputes vs {_pct(p['rest_of_portfolio_rate'])} "
             f"for the rest of the portfolio (z = {p['z']}): **{m['classification']}**.", "",
             "| fact | this merchant | rest of portfolio | z | verdict |", "|---|---|---|---|---|"]
        facts = [p] + [x for x in m["merchant_specific_weaknesses"] if x["fact"] != p["fact"]]
        for x in facts:
            verdict = "merchant-specific" if x["merchant_specific"] else "systemic" if x["systemic"] else "in line"
            L.append(f"| `{x['fact']}` | {_pct(x['weak_rate'])} of {x['disputes']} | "
                     f"{_pct(x['rest_of_portfolio_rate'])} | {x['z']} | {verdict} |")
        return L + [""]

    L = ["## Systemic gaps", ""]
    if weak["systemic"]:
        for f in weak["systemic"]:
            v = weak["facts"][f]
            L.append(f"- **`{f}`** is missing in {_pct(v['missing_rate'])} of relevant disputes, in most disputes of "
                     f"{len(v['merchants_affected'])} separate merchants ({', '.join(v['merchants_affected'])}), and "
                     f"disputes without it lose {_pct(v['loss_rate_when_missing'])} of the time vs "
                     f"{_pct(v['loss_rate_when_present'])} with it: fix the shared evidence pipeline, not individual merchants.")
    else:
        L.append("- None: no weakness is shared by enough separate merchants to be systemic.")
    L += ["", "## Merchant primary weaknesses (benchmarked against the rest of the portfolio)", "",
          "| merchant | losses | primary weakness | weak in (this merchant) | rest of portfolio | in losses | verdict |",
          "|---|---|---|---|---|---|---|"]
    for mid, m in weak["merchants"].items():
        if not m["losses"]:
            continue  # nothing lost yet, so nothing to diagnose
        p = m["primary_weakness"]
        L.append(f"| {mid} | {m['losses']} | `{p['fact']}` | {_pct(p['weak_rate'])} of {p['disputes']} | "
                 f"{_pct(p['rest_of_portfolio_rate'])} | {p['weak_in_losses']}/{p['relevant_losses']} | {m['classification']} |")
    return L + [""]


def render_markdown(title: str, causes: dict, calib: dict, econ: dict, weak: dict | None = None,
                    merchant_id: str | None = None) -> str:
    L = [f"# {title}", "", f"Resolved disputes: **{causes['resolved']}** · lost: **{causes['lost']}**", ""]
    if weak is not None:
        L += _render_weaknesses(weak, merchant_id)
    L += ["## Why disputes are lost", "",
          "*Missing* = no document covered the fact (collect it). *Adverse* = the document says no (fix operations).", "",
          "| fact | missing in losses | adverse in losses |", "|---|---|---|"]
    facts = {f for f, _ in causes["top_missing"]} | {f for f, _ in causes["top_adverse"]}
    if not facts:
        L[-2:] = ["- No lost disputes yet."]
    miss, adv = dict(causes["top_missing"]), dict(causes["top_adverse"])
    for f in sorted(facts, key=lambda f: -(miss.get(f, 0) + adv.get(f, 0))):
        L.append(f"| `{f}` | {miss.get(f, 0)} | {adv.get(f, 0)} |")
    if len(causes["merchants"]) > 1:
        L += ["", "## Merchants", "", "| merchant | resolved | loss rate | ₹ lost | network-ratio proximity |",
              "|---|---|---|---|---|"]
        for mid, m in causes["merchants"].items():
            prox = f"{m['threshold_proximity']:.0%}" if m["threshold_proximity"] is not None else "-"
            L.append(f"| {mid} | {m['resolved']} | {m['loss_rate'] if m['loss_rate'] is not None else '-'} | "
                     f"{m['amount_lost_inr']:,.0f} | {prox} |")
    L += ["", "## Decisions that were contested", "",
          f"- Contested: {econ['contested']}, won: {econ['won']} "
          f"(win rate {econ['win_rate_when_contested'] if econ['win_rate_when_contested'] is not None else '-'})",
          f"- Net recovered after contest costs: **₹{econ['net_recovered_inr']:,.0f}**", "",
          "## P(win) calibration on resolved cases", ""]
    if calib["n"]:
        L.append(f"- n = {calib['n']}, Brier = {calib['brier']}, ECE = {calib['ece']}"
                 + (" — **ALERT: calibration drift, retrain or recalibrate**" if calib["alert"] else ""))
    else:
        L.append("- No resolved cases yet.")
    return "\n".join(L)
