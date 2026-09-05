# Abstain — Evaluation Report

Total cases: 57  |  Scored (has ground truth): 56  |  Excluded (unknown outcome): 1

## Precision / Recall

Computed only over auto-decided cases (CONTEST vs CONCEDE). ESCALATE is excluded by design — it's the abstention mechanism working, not a missed call.

- Precision: 0.565
- Recall: 0.464
- TP=13  FP=10  TN=11  FN=15

## Abstention

- Escalation rate: 12% of scored cases (7 of 56)

## Cost of errors

- False-positive cost (ops cost wasted contesting cases that were lost): ₹7,320
- False-negative cost (recoverable amount left on the table by conceding winnable cases): ₹18,730

## Engine vs. naive 'contest everything' baseline

- Baseline net (recovered − ops cost): ₹260,110
- Engine net (recovered − ops cost, CONTEST cases only): ₹210,280
- **Net improvement over baseline: ₹-49,830**

Note: escalated cases are assumed to cost/recover nothing in this comparison, since their real outcome depends on a human decision this eval doesn't model. That's a conservative assumption — it likely understates the engine's true advantage, since a competent human reviewer wouldn't blindly contest either.

## Case-by-case breakdown

| case_type | action | true_outcome | confidence |
|---|---|---|---|
| standard | CONCEDE | lost | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONTEST | lost | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | lost | MEDIUM |
| standard | CONTEST | lost | MEDIUM |
| standard | CONCEDE | lost | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONTEST | lost | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | ESCALATE | lost | LOW |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONCEDE | lost | LOW |
| standard | CONCEDE | lost | MEDIUM |
| standard | CONTEST | lost | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | ESCALATE | won | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | ESCALATE | won | LOW |
| standard | CONTEST | lost | MEDIUM |
| standard | CONTEST | won | MEDIUM |
| standard | CONCEDE | won | MEDIUM |
| standard | CONTEST | lost | MEDIUM |
| standard | ESCALATE | won | MEDIUM |
| standard | ESCALATE | won | LOW |
| standard | CONCEDE | lost | LOW |
| standard | CONTEST | lost | MEDIUM |
| standard | CONCEDE | won | LOW |
| standard | CONTEST | lost | MEDIUM |
| portfolio_pair_A | ESCALATE | lost | LOW |
| portfolio_pair_B | CONCEDE | lost | MEDIUM |
| counterfactual_baseline | CONCEDE | lost | MEDIUM |
| counterfactual_with_evidence | ESCALATE | won | MEDIUM |
| trivial_amount | CONCEDE | won | MEDIUM |
| high_value_uncertain | CONTEST | lost | MEDIUM |
| late_response_window | CONCEDE | lost | MEDIUM |
| repeat_arbitration | CONTEST | lost | MEDIUM |
| friendly_fraud_suspected | CONCEDE | lost | MEDIUM |
| conflicting_evidence | CONCEDE | lost | LOW |
| partial_refund_mismatch | CONCEDE | won | MEDIUM |
| unresolved_ground_truth | CONTEST | unknown | MEDIUM |