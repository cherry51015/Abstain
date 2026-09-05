"""
Abstain — synthetic dataset generator
Generates merchants.json and disputes.json for the chargeback decision engine.
Deterministic (seeded) so results are reproducible for the eval harness.
"""
import json, random

random.seed(42)

with open("data/reason_codes.json") as f:
    REASON_CODES = json.load(f)

ALL_EVIDENCE = [
    "delivery_proof", "tracking_number", "customer_communication_log",
    "signed_receipt", "terms_acceptance", "auth_data", "ip_geolocation",
    "device_fingerprint", "product_description_match", "return_policy_acceptance"
]

# ---------------- Merchants ----------------
MERCHANTS = [
    {"merchant_id": "mch_01", "name": "Apex Electronics", "industry": "Electronics",
     "tier": "Enterprise", "historical_win_rate": 0.62, "dispute_volume_90d": 340,
     "current_chargeback_rate_pct": 0.35, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 650, "risk_tolerance": "aggressive"},
    {"merchant_id": "mch_02", "name": "Lumina Boutique", "industry": "Fashion",
     "tier": "SMB", "historical_win_rate": 0.41, "dispute_volume_90d": 210,
     "current_chargeback_rate_pct": 0.82, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 900, "risk_tolerance": "conservative"},
    {"merchant_id": "mch_03", "name": "Northbridge SaaS", "industry": "Software/Subscription",
     "tier": "Mid-Market", "historical_win_rate": 0.71, "dispute_volume_90d": 95,
     "current_chargeback_rate_pct": 0.22, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 500, "risk_tolerance": "aggressive"},
    {"merchant_id": "mch_04", "name": "Curio Home Decor", "industry": "Home Goods",
     "tier": "SMB", "historical_win_rate": 0.38, "dispute_volume_90d": 180,
     "current_chargeback_rate_pct": 0.88, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 850, "risk_tolerance": "conservative"},
    {"merchant_id": "mch_05", "name": "Vantage Fitness Gear", "industry": "Sports/Fitness",
     "tier": "Mid-Market", "historical_win_rate": 0.55, "dispute_volume_90d": 150,
     "current_chargeback_rate_pct": 0.55, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 700, "risk_tolerance": "moderate"},
    {"merchant_id": "mch_06", "name": "Zenith Digital Courses", "industry": "EdTech/Digital Goods",
     "tier": "SMB", "historical_win_rate": 0.29, "dispute_volume_90d": 260,
     "current_chargeback_rate_pct": 0.95, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 400, "risk_tolerance": "conservative"},
    {"merchant_id": "mch_07", "name": "Ferro Industrial Supplies", "industry": "B2B/Industrial",
     "tier": "Enterprise", "historical_win_rate": 0.78, "dispute_volume_90d": 40,
     "current_chargeback_rate_pct": 0.10, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 1200, "risk_tolerance": "aggressive"},
    {"merchant_id": "mch_08", "name": "Pinnacle Mobile Accessories", "industry": "Electronics",
     "tier": "SMB", "historical_win_rate": 0.47, "dispute_volume_90d": 300,
     "current_chargeback_rate_pct": 0.70, "network_threshold_pct": 0.90,
     "ops_cost_per_contest_inr": 600, "risk_tolerance": "moderate"},
]

with open("data/merchants.json", "w") as f:
    json.dump(MERCHANTS, f, indent=2)


def evidence_subset(required, completeness_target):
    """Pick a realistic subset of required evidence to hit roughly a target completeness."""
    n = max(0, round(len(required) * completeness_target))
    n = min(n, len(required))
    return sorted(random.sample(required, n))


def simulate_outcome(base_win_rate_range, completeness, noise=0.12):
    lo, hi = base_win_rate_range
    base = lo + (hi - lo) * completeness
    p_win = max(0.02, min(0.98, base + random.uniform(-noise, noise)))
    return "won" if random.random() < p_win else "lost", round(p_win, 3)


disputes = []
case_n = 1

def add_case(merchant_id, reason_code_obj, amount, completeness_target, case_type,
             days_left=random.randint(5, 21), is_repeat=False, forced_outcome=None,
             conflicting=False, note=""):
    global case_n
    required = reason_code_obj["required_evidence"]
    present = evidence_subset(required, completeness_target)
    missing = [e for e in required if e not in present]
    completeness = round(len(present) / len(required), 2) if required else 0.0
    if forced_outcome:
        outcome, p_win = forced_outcome, None
    else:
        outcome, p_win = simulate_outcome(reason_code_obj["base_win_rate_range"], completeness)
    disputes.append({
        "case_id": f"case_{case_n:04d}",
        "merchant_id": merchant_id,
        "reason_code": reason_code_obj["code"],
        "network": reason_code_obj["network"],
        "dispute_amount_inr": amount,
        "evidence_present": present,
        "missing_evidence": missing,
        "evidence_completeness_score": completeness,
        "conflicting_evidence": conflicting,
        "response_deadline_days_left": days_left,
        "is_repeat_dispute": is_repeat,
        "true_outcome": outcome,
        "simulated_win_probability": p_win,
        "case_type": case_type,
        "notes": note
    })
    case_n += 1

# ---- bulk realistic random cases (the statistical backbone, ~45 cases) ----
for _ in range(45):
    rc = random.choice(REASON_CODES)
    merchant = random.choice(MERCHANTS)["merchant_id"]
    amount = random.choice([250, 500, 900, 1500, 2200, 3800, 6000, 9500, 15000, 22000, 45000])
    completeness_target = random.uniform(0.1, 1.0)
    add_case(merchant, rc, amount, completeness_target, "standard")

# ---- deliberate edge cases (the ones that make the demo interesting) ----

# 1) Portfolio-aware pair: identical dispute content, two different merchants
shared_rc = next(r for r in REASON_CODES if r["code"] == "4853")
add_case("mch_01", shared_rc, 3200, 0.55, "portfolio_pair_A",
         note="Same dispute as portfolio_pair_B, different merchant risk profile — engine should diverge.")
add_case("mch_04", shared_rc, 3200, 0.55, "portfolio_pair_B",
         note="Same dispute as portfolio_pair_A — mch_04 is near network threshold, should concede/escalate where mch_01 might contest.")

# 2) Counterfactual pair: same case, one missing evidence item added
weak_rc = next(r for r in REASON_CODES if r["code"] == "13.1")
add_case("mch_02", weak_rc, 4200, 0.33, "counterfactual_baseline",
         note="Missing delivery_proof — baseline weak case for counterfactual demo.")
add_case("mch_02", weak_rc, 4200, 1.0, "counterfactual_with_evidence",
         note="Same case as counterfactual_baseline but with delivery_proof added — should flip decision.")

# 3) Trivial amount — ops cost exceeds amount at stake regardless of evidence
add_case("mch_08", random.choice(REASON_CODES), 180, 0.9, "trivial_amount",
         note="Amount below typical ops cost of contesting — should concede even with strong evidence.")

# 4) Very high amount, moderate/uncertain evidence — should escalate, not auto-decide
add_case("mch_07", next(r for r in REASON_CODES if r["code"] == "10.4"), 95000, 0.5, "high_value_uncertain",
         note="High stakes + partial evidence in a hard fraud category — textbook escalate case.")

# 5) Late response window — strong evidence but almost no time to act
add_case("mch_03", next(r for r in REASON_CODES if r["code"] == "13.1"), 5200, 0.9, "late_response_window",
         days_left=1, note="Only 1 day left to respond despite strong evidence — operational feasibility matters, not just win probability.")

# 6) Repeat / arbitration dispute — second chargeback on same transaction, different economics
add_case("mch_05", next(r for r in REASON_CODES if r["code"] == "4837"), 7600, 0.6, "repeat_arbitration",
         is_repeat=True, note="Second-stage arbitration after an initial loss — higher fee, different EV calculus.")

# 7) Friendly fraud suspected — ambiguous, genuinely uncertain ground truth
add_case("mch_06", next(r for r in REASON_CODES if r["code"] == "F29"), 2100, 0.45, "friendly_fraud_suspected",
         forced_outcome="lost", conflicting=True,
         note="Auth data matches normal usage pattern, but customer insists non-recognition — ambiguous by nature, included to test abstention rather than false confidence.")

# 8) Conflicting evidence — delivery proof exists but signature doesn't match cardholder name
add_case("mch_04", next(r for r in REASON_CODES if r["code"] == "13.3"), 3400, 0.6, "conflicting_evidence",
         conflicting=True, note="Delivery confirmed but signed for by a different name than the cardholder — evidence pulls in two directions.")

# 9) Partial refund already issued, dispute amount doesn't match remaining balance
add_case("mch_02", next(r for r in REASON_CODES if r["code"] == "12.5"), 1250, 0.7, "partial_refund_mismatch",
         note="Merchant already refunded 40% before the dispute was filed; disputed amount doesn't reconcile with remaining charge.")

# 10) Unknown/unresolved ground truth — simulates real held-out uncertainty, not every case can be labeled cleanly
add_case("mch_08", next(r for r in REASON_CODES if r["code"] == "10.4"), 5400, 0.5, "unresolved_ground_truth",
         forced_outcome="unknown", note="Case outcome not yet determined at data collection time — included to keep the eval harness honest about what it can and can't score.")

with open("data/disputes.json", "w") as f:
    json.dump(disputes, f, indent=2)

print(f"Generated {len(MERCHANTS)} merchants and {len(disputes)} disputes.")