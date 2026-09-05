"""
portfolio_intelligence.py

Aggregates resolved dispute outcomes into a root-cause report — not "we
lost this dispute" but "we keep losing disputes because X is missing."
This is the intelligence loop most reason-code-level dispute tooling skips:
systems report per-case outcomes and stop there.

TWO SCOPES, same underlying aggregation:
  - analyze_portfolio(records)              -> "why are WE losing overall?"
  - analyze_merchant(records, merchant_id)  -> "why is THIS merchant losing?"

This distinction exists because they answer different questions for
different audiences. A demo reviewer picking one merchant and asking "what
are their biggest problems" should not get a cross-merchant answer back —
that's a real product gap, not a stylistic one. Both scopes reuse the same
`aggregate_root_causes` core so the numbers are guaranteed consistent
between views (a merchant's figures in the portfolio table and in their own
report come from the exact same pass over the exact same records).

Deliberately operates on POST-HOC resolved records, not live Decision
objects — a Decision at contest-time doesn't yet know whether contesting
actually worked; that's only knowable after the card network resolves the
dispute, typically days or weeks later. In a real deployment this data
would arrive via a reconciliation webhook well after the original
decision, which is why it's modeled as its own input type here rather than
bolted onto DecisionORM as if it were known at decision time.

Consistency with the rest of the codebase, stated explicitly:
  - Same "unknown" outcome handling as eval/run_eval.py: excluded from
    loss-rate math, reported as its own count, never silently dropped or
    guessed at. Every breakdown row now tracks won/lost/unknown
    explicitly, so total == won + lost + unknown always holds and never
    has to be inferred — the earlier version of this report left that
    relationship implicit, which is exactly what made the merchant-table
    row sums look inconsistent with the top-level resolved count.
  - Same fail-loudly-on-bad-input philosophy as decision_engine.py and
    portfolio.py: an unrecognized true_outcome value raises immediately
    rather than being coerced into "lost" or ignored. Same treatment for
    analyze_merchant() on a merchant_id with zero records — an empty
    report would silently look like "no losses," which is a different
    claim than "this merchant isn't in the record set."
  - Same "explicit named category for a genuine gap" pattern as
    decision_engine.py's gates: a loss with zero recorded evidence gaps
    (e.g. a subjective reason code, or a loss driven by conflicting
    evidence rather than a missing document) is tracked under its own
    named bucket, not dropped from the evidence-frequency table.
"""
from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from pydantic import BaseModel, Field

_NON_EVIDENCE_LOSS_LABEL = "(no missing evidence recorded — non-evidence loss)"
_VALID_OUTCOMES = {"won", "lost", "unknown"}
_RECOMMENDATION_TOP_N = 3


class ResolvedCaseRecord(BaseModel):
    """One dispute with a known-after-the-fact outcome. `missing_evidence`
    must be the gap list AS IT STOOD AT DECISION TIME — reconstructing it
    after the fact would let hindsight quietly rewrite what was actually
    known when the call was made, which defeats the point of an honest
    root-cause report."""
    case_id: str
    merchant_id: str
    reason_code: str
    true_outcome: str  # "won" / "lost" / "unknown"
    missing_evidence: list[str] = Field(default_factory=list)
    dispute_amount_inr: float = 0.0


@dataclass
class ReasonCodeBreakdown:
    reason_code: str
    total: int = 0
    won: int = 0
    lost: int = 0
    unknown: int = 0

    @property
    def loss_rate(self) -> float | None:
        resolved = self.won + self.lost
        return round(self.lost / resolved, 3) if resolved else None


@dataclass
class MerchantBreakdown:
    merchant_id: str
    total: int = 0
    won: int = 0
    lost: int = 0
    unknown: int = 0
    amount_lost_inr: float = 0.0

    @property
    def loss_rate(self) -> float | None:
        resolved = self.won + self.lost
        return round(self.lost / resolved, 3) if resolved else None


@dataclass
class RootCauseReport:
    scope: str = "portfolio"          # "portfolio" or "merchant"
    merchant_id: str | None = None    # set only when scope == "merchant"

    total_records: int = 0
    excluded_unknown: int = 0
    resolved: int = 0
    lost: int = 0
    won: int = 0

    missing_evidence_frequency_among_losses: dict[str, int] = field(default_factory=dict)
    reason_code_breakdown: dict[str, ReasonCodeBreakdown] = field(default_factory=dict)
    merchant_breakdown: dict[str, MerchantBreakdown] = field(default_factory=dict)

    @property
    def loss_rate(self) -> float | None:
        return round(self.lost / self.resolved, 3) if self.resolved else None

    @property
    def amount_lost_inr(self) -> float:
        """Total ₹ lost across every record in this report's scope. For a
        merchant-scoped report there's exactly one entry in
        merchant_breakdown, so this doubles as "this merchant's loss"."""
        return sum(m.amount_lost_inr for m in self.merchant_breakdown.values())

    @property
    def top_operational_weakness(self) -> tuple[str, int] | None:
        if not self.missing_evidence_frequency_among_losses:
            return None
        return max(self.missing_evidence_frequency_among_losses.items(), key=lambda kv: kv[1])

    @property
    def top_weaknesses(self) -> list[tuple[str, int]]:
        """Highest-count gaps first, ties broken alphabetically so output
        is deterministic run to run."""
        return sorted(
            self.missing_evidence_frequency_among_losses.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )

    # ---- portfolio-only highlights (meaningless with a single merchant) ----

    @property
    def merchant_with_highest_loss_rate(self) -> MerchantBreakdown | None:
        candidates = [m for m in self.merchant_breakdown.values() if m.loss_rate is not None]
        return max(candidates, key=lambda m: m.loss_rate) if candidates else None

    @property
    def merchant_with_highest_exposure(self) -> MerchantBreakdown | None:
        candidates = [m for m in self.merchant_breakdown.values() if m.amount_lost_inr > 0]
        return max(candidates, key=lambda m: m.amount_lost_inr) if candidates else None


def aggregate_root_causes(
    records: list[ResolvedCaseRecord],
    *,
    scope: str = "portfolio",
    merchant_id: str | None = None,
) -> RootCauseReport:
    """Core aggregation, scope-agnostic — it has no idea whether `records`
    is the whole portfolio or one merchant's slice of it. `scope` and
    `merchant_id` are carried through purely for the renderer to know which
    view it's building; they never affect the numbers."""
    if not records:
        raise ValueError("Cannot build a root-cause report from an empty record set.")

    report = RootCauseReport(scope=scope, merchant_id=merchant_id, total_records=len(records))
    evidence_counter: Counter[str] = Counter()
    reason_codes: dict[str, ReasonCodeBreakdown] = {}
    merchants: dict[str, MerchantBreakdown] = {}

    for rec in records:
        if rec.true_outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"Case {rec.case_id!r} has an unrecognized true_outcome "
                f"{rec.true_outcome!r} — expected one of {sorted(_VALID_OUTCOMES)}."
            )

        rc = reason_codes.setdefault(rec.reason_code, ReasonCodeBreakdown(reason_code=rec.reason_code))
        rc.total += 1
        m = merchants.setdefault(rec.merchant_id, MerchantBreakdown(merchant_id=rec.merchant_id))
        m.total += 1

        if rec.true_outcome == "unknown":
            report.excluded_unknown += 1
            rc.unknown += 1
            m.unknown += 1
            continue

        report.resolved += 1
        if rec.true_outcome == "won":
            report.won += 1
            rc.won += 1
            m.won += 1
            continue

        # lost
        report.lost += 1
        rc.lost += 1
        m.lost += 1
        m.amount_lost_inr += rec.dispute_amount_inr

        if rec.missing_evidence:
            for gap in rec.missing_evidence:
                evidence_counter[gap] += 1
        else:
            evidence_counter[_NON_EVIDENCE_LOSS_LABEL] += 1

    report.missing_evidence_frequency_among_losses = dict(evidence_counter)
    report.reason_code_breakdown = reason_codes
    report.merchant_breakdown = merchants
    return report


def filter_records_for_merchant(
    records: list[ResolvedCaseRecord], merchant_id: str
) -> list[ResolvedCaseRecord]:
    """Fails loudly on an unknown/empty merchant_id rather than returning an
    empty list — an empty-but-valid report and 'this merchant isn't in the
    record set' are different claims, and aggregate_root_causes() already
    refuses empty input, so this is where that distinction needs a clear
    message instead of a confusing downstream ValueError."""
    matched = [r for r in records if r.merchant_id == merchant_id]
    if not matched:
        raise ValueError(f"No records found for merchant_id={merchant_id!r}.")
    return matched


def analyze_portfolio(records: list[ResolvedCaseRecord]) -> RootCauseReport:
    """'Why are we losing overall?' — every merchant, aggregated."""
    return aggregate_root_causes(records, scope="portfolio")


def analyze_merchant(records: list[ResolvedCaseRecord], merchant_id: str) -> RootCauseReport:
    """'Why is THIS merchant losing?' — same aggregation, pre-filtered to
    one merchant's records. Raises if merchant_id matches nothing."""
    merchant_records = filter_records_for_merchant(records, merchant_id)
    return aggregate_root_causes(merchant_records, scope="merchant", merchant_id=merchant_id)


def load_resolved_records(dataset_dir: str | Path) -> list[ResolvedCaseRecord]:
    """Loads ResolvedCaseRecord objects straight from the synthetic
    dataset's disputes.json. This is the only place in the app that treats
    true_outcome as known — everywhere else (decision_engine.py, the live
    /disputes/evaluate endpoint) it's genuinely unknown at decision time.
    True_outcome only exists here because it's synthetic ground truth for
    the demo dataset; a production deployment would source these records
    from the reconciliation webhook described in the module docstring
    instead of a JSON file, but ResolvedCaseRecord's shape stays the same
    either way."""
    dataset_dir = Path(dataset_dir)
    with open(dataset_dir / "disputes.json") as f:
        raw = json.load(f)
    return [
        ResolvedCaseRecord(
            case_id=d["case_id"],
            merchant_id=d["merchant_id"],
            reason_code=d["reason_code"],
            true_outcome=d["true_outcome"],
            missing_evidence=d.get("missing_evidence", []),
            dispute_amount_inr=d.get("dispute_amount_inr", 0.0),
        )
        for d in raw
    ]


# ---------------------------- rendering ----------------------------

def _render_weakness_table(report: RootCauseReport, denom_label: str) -> list[str]:
    lines = [f"| evidence gap | count | % of {denom_label} |", "|---|---|---|"]
    for gap, count in report.top_weaknesses:
        pct = round(count / report.lost, 3) if report.lost else 0.0
        lines.append(f"| {gap} | {count} | {pct:.0%} |")
    return lines


def render_portfolio_markdown(report: RootCauseReport) -> str:
    if report.scope != "portfolio":
        raise ValueError(f"render_portfolio_markdown() requires scope='portfolio', got {report.scope!r}.")

    lines = [
        "# Portfolio Intelligence — Root Cause Report",
        "",
        f"Total records: {report.total_records}  |  Resolved: {report.resolved}  |  "
        f"Excluded (unknown outcome): {report.excluded_unknown}",
        f"Won: {report.won}  |  Lost: {report.lost}  |  "
        f"Overall loss rate: {report.loss_rate if report.loss_rate is not None else 'N/A'}",
        "",
        "*Note: `total` in the tables below counts every record for that reason "
        "code / merchant, including unknown-outcome ones — it is not the same as "
        "`won + lost`. `total == won + lost + unknown` always holds.*",
        "",
        "## Highlights",
        "",
    ]

    top = report.top_operational_weakness
    if top is None:
        lines.append("- No losses recorded yet — nothing to attribute.")
    else:
        gap, count = top
        pct = round(count / report.lost, 3) if report.lost else 0.0
        lines.append(
            f"- 📊 **Top systemic weakness:** {gap} — present in {count} of "
            f"{report.lost} lost disputes ({pct:.0%})."
        )

    worst_rate = report.merchant_with_highest_loss_rate
    if worst_rate is not None:
        lines.append(f"- 🚨 **Highest loss rate:** {worst_rate.merchant_id} ({worst_rate.loss_rate:.0%})")

    worst_exposure = report.merchant_with_highest_exposure
    if worst_exposure is not None:
        lines.append(
            f"- 💰 **Highest financial exposure:** {worst_exposure.merchant_id} "
            f"(₹{worst_exposure.amount_lost_inr:,.0f} lost)"
        )

    lines += ["", "## Missing-evidence frequency among losses", ""]
    lines += _render_weakness_table(report, "losses")

    lines += [
        "", "## By reason code", "",
        "| reason_code | total | won | lost | unknown | loss_rate |", "|---|---|---|---|---|---|",
    ]
    for rc in sorted(report.reason_code_breakdown.values(), key=lambda x: -x.lost):
        lines.append(f"| {rc.reason_code} | {rc.total} | {rc.won} | {rc.lost} | {rc.unknown} | "
                      f"{rc.loss_rate if rc.loss_rate is not None else 'N/A'} |")

    lines += [
        "", "## By merchant", "",
        "| merchant_id | total | won | lost | unknown | loss_rate | amount_lost_inr |",
        "|---|---|---|---|---|---|---|",
    ]
    for m in sorted(report.merchant_breakdown.values(), key=lambda x: -x.amount_lost_inr):
        lines.append(f"| {m.merchant_id} | {m.total} | {m.won} | {m.lost} | {m.unknown} | "
                      f"{m.loss_rate if m.loss_rate is not None else 'N/A'} | ₹{m.amount_lost_inr:,.0f} |")

    return "\n".join(lines)


def render_merchant_markdown(report: RootCauseReport) -> str:
    if report.scope != "merchant" or report.merchant_id is None:
        raise ValueError("render_merchant_markdown() requires a report built by analyze_merchant().")

    lines = [
        f"# Merchant Root Cause Report — {report.merchant_id}",
        "",
        f"Disputes: {report.total_records}  |  Won: {report.won}  |  Lost: {report.lost}  |  "
        f"Unknown outcome: {report.excluded_unknown}",
        f"Loss rate: {f'{report.loss_rate:.1%}' if report.loss_rate is not None else 'N/A'}  |  "
        f"Amount lost: ₹{report.amount_lost_inr:,.0f}",
        "",
        "## Top weaknesses",
        "",
    ]
    lines += _render_weakness_table(report, "this merchant's losses")

    lines += ["", "## Recommendation", ""]
    # The non-evidence-loss bucket isn't an evidence gap — nothing to
    # "capture" there — so it's excluded from the recommendation even
    # though it stays in the frequency table above for transparency.
    actionable = [(gap, count) for gap, count in report.top_weaknesses if gap != _NON_EVIDENCE_LOSS_LABEL]
    top_n = actionable[:_RECOMMENDATION_TOP_N]
    if not top_n:
        lines.append("No actionable evidence gaps recorded for this merchant's losses.")
    else:
        gaps = ", ".join(gap for gap, _ in top_n)
        lines.append(
            f"Prioritize capturing: **{gaps}** — these account for the most "
            f"losses in this merchant's history."
        )

    return "\n".join(lines)


def render_markdown(report: RootCauseReport) -> str:
    """Dispatches on report.scope. Kept as the single stable entry point so
    callers that already do `render_markdown(analyze_portfolio(records))`
    don't need to know the scope split exists."""
    if report.scope == "portfolio":
        return render_portfolio_markdown(report)
    if report.scope == "merchant":
        return render_merchant_markdown(report)
    raise ValueError(f"Unrecognized report scope {report.scope!r}.")


if __name__ == "__main__":
    """Local CLI, no FastAPI/DB/Docker required — mirrors eval/run_eval.py's
    __main__ pattern. Writes the portfolio report plus one file per
    merchant, all under reporting_output/ next to this script.

    Usage:
        python portfolio_intelligence.py                 # auto-locate dataset/
        python portfolio_intelligence.py path/to/dataset  # explicit dataset dir
    """
    import sys

    def _default_dataset_dir() -> Path:
        here = Path(__file__).resolve().parent
        for candidate in (here.parent / "dataset", here.parent.parent / "dataset"):
            if (candidate / "disputes.json").exists():
                return candidate
        raise FileNotFoundError(
            "Could not auto-locate dataset/disputes.json near this script "
            "(checked one and two levels up). Pass the dataset directory "
            "explicitly: python portfolio_intelligence.py path/to/dataset"
        )

    dataset_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else _default_dataset_dir()
    out_dir = Path(__file__).resolve().parent / "reporting_output"
    out_dir.mkdir(exist_ok=True)

    records = load_resolved_records(dataset_dir)

    portfolio = analyze_portfolio(records)
    portfolio_md = render_markdown(portfolio)
    (out_dir / "portfolio_report.md").write_text(portfolio_md, encoding="utf-8")
    print(portfolio_md)
    print(f"\nWritten to {out_dir / 'portfolio_report.md'}")

    merchant_ids = sorted(portfolio.merchant_breakdown.keys())
    for mid in merchant_ids:
        merchant_md = render_markdown(analyze_merchant(records, mid))
        (out_dir / f"merchant_{mid}_report.md").write_text(merchant_md, encoding="utf-8")

    print(f"\nWrote {len(merchant_ids)} merchant reports to {out_dir}/merchant_<id>_report.md")
    print(f"Merchants: {', '.join(merchant_ids)}")


