"""Root-cause and calibration aggregations (app/insights.py) on hand-built rows."""
from __future__ import annotations

from decimal import Decimal
from types import SimpleNamespace

from app.db import ActionEnum, OutcomeEnum
from app.insights import calibration, realized_economics, render_markdown, root_causes


def row(case_id, merchant, reason, amount, outcome, facts, p_win=0.5, action=ActionEnum.CONTEST, review=None):
    ev = SimpleNamespace(facts=facts, p_win=p_win, action=action, review=review, contest_cost_inr=Decimal("500"))
    return ev, SimpleNamespace(outcome=OutcomeEnum(outcome)), SimpleNamespace(
        case_id=case_id, merchant_id=merchant, reason_code=reason, amount_inr=Decimal(amount))


ROWS = [
    row("1", "mch_01", "13.1", 4000, "lost", {"delivery_confirmed": "unknown", "signed_by_cardholder": "no"}),
    row("2", "mch_01", "13.1", 2000, "lost", {"delivery_confirmed": "unknown", "signed_by_cardholder": "yes"}),
    row("3", "mch_02", "13.1", 9000, "won", {"delivery_confirmed": "yes"}, p_win=0.9),
    row("4", "mch_02", "12.5", 1000, "lost", {"amount_matches_agreement": "no"}, action=ActionEnum.CONCEDE),
]


def test_missing_and_adverse_are_counted_separately(catalog):
    causes = root_causes(ROWS, catalog)
    assert dict(causes["top_missing"])["delivery_confirmed"] == 2
    assert dict(causes["top_adverse"])["signed_by_cardholder"] == 1
    assert dict(causes["top_adverse"])["amount_matches_agreement"] == 1
    assert causes["resolved"] == 4 and causes["lost"] == 3


def test_per_merchant_breakdown(catalog):
    m = root_causes(ROWS, catalog)["merchants"]
    assert m["mch_01"]["loss_rate"] == 1.0 and m["mch_01"]["amount_lost_inr"] == 6000
    assert m["mch_02"]["loss_rate"] == 0.5 and m["mch_02"]["threshold_proximity"] is not None


def test_realized_economics_counts_only_contested_and_honours_reviews():
    econ = realized_economics(ROWS)
    assert econ["contested"] == 3 and econ["won"] == 1 and econ["net_recovered_inr"] == 9000 - 3 * 500
    reviewed = row("5", "mch_03", "13.1", 5000, "won", {}, action=ActionEnum.ESCALATE,
                   review=SimpleNamespace(action=ActionEnum.CONTEST))
    assert realized_economics([reviewed])["contested"] == 1


def test_calibration_alert_needs_enough_cases():
    few = calibration(ROWS)
    assert few["n"] == 4 and not few["alert"]
    many = calibration([row(str(i), "mch_01", "13.1", 1000, "lost", {}, p_win=0.95) for i in range(60)])
    assert many["alert"] and many["ece"] > 0.9


def test_markdown_renders(catalog):
    md = render_markdown("Portfolio", root_causes(ROWS, catalog), calibration(ROWS), realized_economics(ROWS))
    assert "delivery_confirmed" in md and "Merchants" in md


def planted_rows():
    """Planted: mch_01 has its own signature problem; every merchant's pipeline drops the customer's
    acknowledgement of receipt, and disputes without it are lost while the others are won."""
    rows = []
    for m in ["mch_01", "mch_02", "mch_03", "mch_05", "mch_07"]:
        for i in range(20):
            ack_missing = i < 16                                                   # 80% missing everywhere
            facts = {
                "delivery_confirmed": "yes",
                "signed_by_cardholder": "no" if (m == "mch_01" and i < 18) or i % 5 == 0 else "yes",  # 90% vs 20%
                "customer_acknowledged_receipt": "unknown" if ack_missing else "yes",
            }
            rows.append(row(f"{m}-{i}", m, "13.1", 1000, "lost" if ack_missing else "won", facts))
    return rows


def test_merchant_specific_weakness_is_benchmarked_and_flagged(catalog):
    from app.insights import weaknesses
    w = weaknesses(planted_rows(), catalog)
    m1 = {x["fact"]: x for x in w["merchants"]["mch_01"]["merchant_specific_weaknesses"]}
    assert set(m1) == {"signed_by_cardholder"}
    assert m1["signed_by_cardholder"]["weak_rate"] == 0.9 and m1["signed_by_cardholder"]["rest_of_portfolio_rate"] == 0.2
    assert m1["signed_by_cardholder"]["z"] > 1.96
    assert w["merchants"]["mch_01"]["classification"] == "merchant-specific"
    for other in ["mch_02", "mch_03", "mch_05", "mch_07"]:
        assert w["merchants"][other]["merchant_specific_weaknesses"] == []


def test_shared_gap_that_costs_wins_is_systemic(catalog):
    from app.insights import weaknesses
    w = weaknesses(planted_rows(), catalog)
    assert w["systemic"] == ["customer_acknowledged_receipt"]
    assert len(w["facts"]["customer_acknowledged_receipt"]["merchants_affected"]) == 5
    assert w["merchants"]["mch_02"]["classification"] == "systemic"


def test_a_common_gap_that_does_not_affect_outcomes_is_not_systemic(catalog):
    from app.insights import weaknesses
    rows = [row(f"{m}-{i}", m, "13.1", 1000, "lost" if i % 2 else "won",
                {"delivery_confirmed": "unknown" if i < 16 else "yes"})
            for m in ["mch_01", "mch_02", "mch_03"] for i in range(20)]
    assert weaknesses(rows, catalog)["systemic"] == []


def test_small_samples_are_not_flagged(catalog):
    from app.insights import weaknesses
    rows = [row(f"x{i}", "mch_01", "13.1", 1000, "lost", {"signed_by_cardholder": "no"}) for i in range(3)]
    rows += [row(f"y{i}", "mch_02", "13.1", 1000, "lost", {"signed_by_cardholder": "yes"}) for i in range(3)]
    w = weaknesses(rows, catalog)
    assert w["merchants"]["mch_01"]["merchant_specific_weaknesses"] == []  # 3 disputes is not evidence


def test_signature_is_not_blamed_when_nothing_was_delivered(catalog):
    """No delivery -> no signature is possible; counting it would confound the two facts."""
    from app.insights import weaknesses
    rows = [row(f"{m}-{i}", m, "13.1", 1000, "lost" if i < 12 else "won",
                {"delivery_confirmed": "no" if i < 12 else "yes",
                 "signed_by_cardholder": "unknown" if i < 12 else "yes"})
            for m in ["mch_01", "mch_02", "mch_03"] for i in range(20)]
    w = weaknesses(rows, catalog)
    assert "signed_by_cardholder" not in w["systemic"]
    assert w["facts"]["signed_by_cardholder"]["missing_rate"] == 0.0   # only delivered parcels counted


def test_chance_gap_among_many_tests_is_watched_not_flagged(catalog):
    """8 merchants x 5 fraud facts = 40 tests. A 7-of-8 vs 45% gap (z~2.3, p~0.01) would pass a
    plain 5% test, but not Benjamini-Hochberg across 40 tests: it is shown as worth watching."""
    from app.insights import weaknesses
    facts5 = ["avs_cvv_match", "ip_consistent_with_cardholder", "prior_undisputed_orders",
              "delivery_confirmed", "signed_by_cardholder"]
    rows = []
    for m in ["mch_01", "mch_02", "mch_03", "mch_04", "mch_05", "mch_06", "mch_08"]:
        for i in range(20):
            rows.append(row(f"{m}-{i}", m, "10.4", 1000, "lost" if i % 2 else "won",
                            {f: ("no" if i < 9 else "yes") for f in facts5}))
    for i in range(8):
        f = {x: ("no" if i < 4 else "yes") for x in facts5}
        f["avs_cvv_match"] = "no" if i < 7 else "yes"
        rows.append(row(f"mch_07-{i}", "mch_07", "10.4", 1000, "lost", f))
    m7 = weaknesses(rows, catalog)["merchants"]["mch_07"]
    assert m7["merchant_specific_weaknesses"] == []
    assert [x["fact"] for x in m7["worth_watching"]] == ["avs_cvv_match"]
    assert m7["classification"] == "worth watching"
