"""
run_eval.py

Evaluates the full pipeline against the synthetic dataset and produces
eval_report.md — the artifact meant to actually prove the "honest metrics"
bar, not a number quoted in a slide.

WHAT "PRECISION/RECALL" MEANS HERE — stated explicitly because it is not
obvious by default. The engine's output is a three-way action
(CONTEST/CONCEDE/ESCALATE), not a raw win/lose prediction, so evaluating
it as a binary classifier requires a defined mapping, not an assumed one:

  - Ground truth "should contest"  = true_outcome == "won"
    (contesting would have recovered the money)
  - Ground truth "should concede"  = true_outcome == "lost"
    (contesting would have wasted ops cost for nothing)
  - "unknown" outcomes are held OUT of precision/recall entirely and
    reported separately. Scoring against a label that doesn't exist would
    be fabricating ground truth, not evaluating against it.

  - ESCALATE decisions are EXCLUDED from precision/recall on purpose. They
    are the abstention mechanism working as designed, not a missed
    prediction — folding them into "wrong" would penalize the exact
    behavior this project is built around. They're reported separately as
    the escalation rate, which is itself a headline number.

  - Precision = TP / (TP + FP), computed only over auto-decided cases
    (CONTEST vs CONCEDE)
  - Recall    = TP / (TP + FN), same restricted set

COST ACCOUNTING, not just accuracy:
  - False-positive cost: ops cost spent contesting cases that were
    actually lost.
  - False-negative cost: recoverable amount left on the table by
    conceding cases that were actually winnable.
  - Baseline comparison: net outcome of a naive "contest everything"
    policy, so the engine's selectivity is measured against something,
    not reported in a vacuum.

KNOWN LIMITATION, reported here rather than hidden: with no LLM client
configured, the evidence_scorer fallback heuristic pins uncertainty at a
fixed value that sits at the LOW-confidence boundary, so the engine
escalates nearly every non-trivial case rather than ever auto-deciding.
That makes precision/recall computable on very few (or zero) cases in
fallback mode. This is correct, conservative behavior — but it means a
credible eval run for the actual submission needs a live LLM client, not
this fallback path. Run this file with llm_client=None first to confirm
the harness itself works end-to-end (it will, on ~0 auto-decided cases),
then re-run with Groq configured for the numbers that go in the pitch.
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.engine.decision_engine import DecisionEngine
from app.engine.evidence_scorer import EvidenceScorer
from app.engine.models import Merchant, DisputeCase, Action
from app.llm_client import build_llm_client


@dataclass
class EvalCaseResult:
    case_id: str
    merchant_id: str
    true_outcome: str  # "won" / "lost" / "unknown"
    action: Action
    dispute_amount_inr: float
    ops_cost_inr: float
    confidence_label: str
    case_type: str


@dataclass
class EvalReport:
    total_cases: int = 0
    scored_cases: int = 0       # true_outcome != "unknown"
    excluded_unknown: int = 0
    escalated: int = 0
    auto_decided: int = 0       # scored_cases - escalated

    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    fp_cost_inr: float = 0.0    # ops cost wasted contesting cases that were lost
    fn_cost_inr: float = 0.0    # recoverable amount left on the table

    engine_total_cost_inr: float = 0.0        # ops cost the engine actually spent (CONTEST only)
    engine_total_recovered_inr: float = 0.0

    baseline_total_cost_inr: float = 0.0       # naive "contest everything" policy
    baseline_total_recovered_inr: float = 0.0

    results: list[EvalCaseResult] = field(default_factory=list)

    @property
    def precision(self) -> float | None:
        denom = self.tp + self.fp
        return round(self.tp / denom, 3) if denom > 0 else None

    @property
    def recall(self) -> float | None:
        denom = self.tp + self.fn
        return round(self.tp / denom, 3) if denom > 0 else None

    @property
    def escalation_rate(self) -> float:
        return round(self.escalated / self.scored_cases, 3) if self.scored_cases else 0.0

    @property
    def net_saved_vs_baseline_inr(self) -> float:
        engine_net = self.engine_total_recovered_inr - self.engine_total_cost_inr
        baseline_net = self.baseline_total_recovered_inr - self.baseline_total_cost_inr
        return round(engine_net - baseline_net, 2)


def run_eval(dataset_dir: str | Path, llm_client=None) -> EvalReport:
    dataset_dir = Path(dataset_dir)
    merchants = {m["merchant_id"]: m for m in json.load(open(dataset_dir / "merchants.json"))}
    reason_codes = {r["code"]: r for r in json.load(open(dataset_dir / "reason_codes.json"))}
    disputes = json.load(open(dataset_dir / "disputes.json"))

    scorer = EvidenceScorer(llm_client=llm_client)
    engine = DecisionEngine()
    report = EvalReport(total_cases=len(disputes))

    for d in disputes:
        m = Merchant(**merchants[d["merchant_id"]])
        case = DisputeCase(
            case_id=d["case_id"], merchant_id=d["merchant_id"], reason_code=d["reason_code"],
            dispute_amount_inr=d["dispute_amount_inr"],
            response_deadline_days_left=d["response_deadline_days_left"],
            is_repeat_dispute=d["is_repeat_dispute"], conflicting_evidence=d["conflicting_evidence"],
        )
        rc = reason_codes[d["reason_code"]]
        assessment = scorer.assess(
            reason_code=d["reason_code"], reason_label=rc["label"],
            required_evidence=rc["required_evidence"],
            present_evidence=d["evidence_present"], missing_evidence=d["missing_evidence"],
        )
        decision = engine.decide(case, m, assessment)
        true_outcome = d["true_outcome"]

        # --- naive "contest everything" baseline, computed for every case ---
        report.baseline_total_cost_inr += decision.ops_cost_inr
        if true_outcome == "won":
            report.baseline_total_recovered_inr += d["dispute_amount_inr"]

        report.results.append(EvalCaseResult(
            case_id=d["case_id"], merchant_id=d["merchant_id"], true_outcome=true_outcome,
            action=decision.action, dispute_amount_inr=d["dispute_amount_inr"],
            ops_cost_inr=decision.ops_cost_inr, confidence_label=decision.confidence_label,
            case_type=d["case_type"],
        ))

        # --- engine's actual cost/recovery, restricted to what it really contested ---
        if decision.action == Action.CONTEST:
            report.engine_total_cost_inr += decision.ops_cost_inr
            if true_outcome == "won":
                report.engine_total_recovered_inr += d["dispute_amount_inr"]

        if true_outcome == "unknown":
            report.excluded_unknown += 1
            continue
        report.scored_cases += 1

        if decision.action == Action.ESCALATE:
            report.escalated += 1
            continue  # excluded from precision/recall by design — see module docstring

        report.auto_decided += 1
        predicted_contest = decision.action == Action.CONTEST
        actually_winnable = true_outcome == "won"

        if predicted_contest and actually_winnable:
            report.tp += 1
        elif predicted_contest and not actually_winnable:
            report.fp += 1
            report.fp_cost_inr += decision.ops_cost_inr
        elif not predicted_contest and not actually_winnable:
            report.tn += 1
        else:  # conceded a case that was actually winnable
            report.fn += 1
            report.fn_cost_inr += d["dispute_amount_inr"]

    return report


def render_markdown(report: EvalReport) -> str:
    lines = [
        "# Abstain — Evaluation Report",
        "",
        f"Total cases: {report.total_cases}  |  Scored (has ground truth): {report.scored_cases}  |  "
        f"Excluded (unknown outcome): {report.excluded_unknown}",
        "",
        "## Precision / Recall",
        "",
        "Computed only over auto-decided cases (CONTEST vs CONCEDE). ESCALATE is "
        "excluded by design — it's the abstention mechanism working, not a missed call.",
        "",
    ]

    if report.auto_decided == 0:
        lines += [
            "**No cases were auto-decided — precision/recall cannot be computed from this run.** "
            "Expected if no LLM client was configured: the fallback heuristic's fixed "
            "uncertainty sits at the LOW-confidence boundary by design, so it never trusts "
            "itself enough to auto-contest or auto-concede on evidence-driven cases. "
            "Re-run with a live LLM client for meaningful numbers.",
            "",
        ]
    else:
        lines += [
            f"- Precision: {report.precision if report.precision is not None else 'N/A (no auto-CONTEST decisions)'}",
            f"- Recall: {report.recall if report.recall is not None else 'N/A (no winnable cases auto-decided)'}",
            f"- TP={report.tp}  FP={report.fp}  TN={report.tn}  FN={report.fn}",
            "",
        ]

    lines += [
        "## Abstention",
        "",
        f"- Escalation rate: {report.escalation_rate:.0%} of scored cases "
        f"({report.escalated} of {report.scored_cases})",
        "",
        "## Cost of errors",
        "",
        f"- False-positive cost (ops cost wasted contesting cases that were lost): ₹{report.fp_cost_inr:,.0f}",
        f"- False-negative cost (recoverable amount left on the table by conceding winnable cases): "
        f"₹{report.fn_cost_inr:,.0f}",
        "",
        "## Engine vs. naive 'contest everything' baseline",
        "",
        f"- Baseline net (recovered − ops cost): "
        f"₹{report.baseline_total_recovered_inr - report.baseline_total_cost_inr:,.0f}",
        f"- Engine net (recovered − ops cost, CONTEST cases only): "
        f"₹{report.engine_total_recovered_inr - report.engine_total_cost_inr:,.0f}",
        f"- **Net improvement over baseline: ₹{report.net_saved_vs_baseline_inr:,.0f}**",
        "",
        "Note: escalated cases are assumed to cost/recover nothing in this comparison, "
        "since their real outcome depends on a human decision this eval doesn't model. "
        "That's a conservative assumption — it likely understates the engine's true "
        "advantage, since a competent human reviewer wouldn't blindly contest either.",
        "",
        "## Case-by-case breakdown",
        "",
        "| case_type | action | true_outcome | confidence |",
        "|---|---|---|---|",
    ]
    for res in report.results:
        lines.append(f"| {res.case_type} | {res.action.value} | {res.true_outcome} | {res.confidence_label} |")

    return "\n".join(lines)


if __name__ == "__main__":
    dataset_dir = Path(__file__).resolve().parents[1] / "dataset"
    report = run_eval(dataset_dir, llm_client=build_llm_client())
    md = render_markdown(report)
    out_path = Path(__file__).resolve().parent / "eval_report.md"
    out_path.write_text(md, encoding="utf-8")
    print(md)
    print(f"\nWritten to {out_path}")