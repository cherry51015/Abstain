"""
test_portfolio_intelligence.py

Covers the merchant/portfolio scope split: the two views must be
independently correct AND mutually consistent — a merchant's numbers in
the portfolio table have to match what analyze_merchant() produces for
that same merchant from the same record set.
"""
from __future__ import annotations

import pytest

from app.reporting.portfolio_intelligence import (
    ResolvedCaseRecord, aggregate_root_causes, analyze_merchant,
    analyze_portfolio, filter_records_for_merchant, render_merchant_markdown,
    render_portfolio_markdown, render_markdown,
)


def _records() -> list[ResolvedCaseRecord]:
    return [
        # mch_a: 2 lost, 1 won -> loss_rate 2/3
        ResolvedCaseRecord(case_id="c1", merchant_id="mch_a", reason_code="13.1",
                            true_outcome="lost", missing_evidence=["tracking_number"],
                            dispute_amount_inr=1000),
        ResolvedCaseRecord(case_id="c2", merchant_id="mch_a", reason_code="13.1",
                            true_outcome="lost", missing_evidence=["tracking_number", "delivery_proof"],
                            dispute_amount_inr=500),
        ResolvedCaseRecord(case_id="c3", merchant_id="mch_a", reason_code="4853",
                            true_outcome="won", dispute_amount_inr=200),
        # mch_b: 1 lost, 1 unknown -> loss_rate 1/1 (unknown excluded from rate)
        ResolvedCaseRecord(case_id="c4", merchant_id="mch_b", reason_code="13.1",
                            true_outcome="lost", missing_evidence=["delivery_proof"],
                            dispute_amount_inr=5000),
        ResolvedCaseRecord(case_id="c5", merchant_id="mch_b", reason_code="13.1",
                            true_outcome="unknown"),
    ]


class TestAggregateRootCauses:
    def test_empty_records_raises(self):
        with pytest.raises(ValueError):
            aggregate_root_causes([])

    def test_unrecognized_outcome_raises(self):
        bad = [ResolvedCaseRecord(case_id="c1", merchant_id="m", reason_code="x", true_outcome="pending")]
        with pytest.raises(ValueError):
            aggregate_root_causes(bad)

    def test_totals_include_unknown(self):
        """total == won + lost + unknown must hold for every breakdown row —
        this is the exact relationship that was implicit (and therefore easy
        to misread) before merchant/reason-code unknown counts existed."""
        report = analyze_portfolio(_records())
        for m in report.merchant_breakdown.values():
            assert m.total == m.won + m.lost + m.unknown
        for rc in report.reason_code_breakdown.values():
            assert rc.total == rc.won + rc.lost + rc.unknown

    def test_portfolio_totals(self):
        report = analyze_portfolio(_records())
        assert report.total_records == 5
        assert report.excluded_unknown == 1
        assert report.resolved == 4
        assert report.won == 1
        assert report.lost == 3

    def test_non_evidence_loss_bucket(self):
        recs = _records() + [
            ResolvedCaseRecord(case_id="c6", merchant_id="mch_a", reason_code="4853",
                                true_outcome="lost", missing_evidence=[], dispute_amount_inr=100),
        ]
        report = analyze_portfolio(recs)
        assert "(no missing evidence recorded — non-evidence loss)" in report.missing_evidence_frequency_among_losses


class TestMerchantScope:
    def test_filter_isolates_one_merchant(self):
        filtered = filter_records_for_merchant(_records(), "mch_a")
        assert {r.case_id for r in filtered} == {"c1", "c2", "c3"}

    def test_filter_unknown_merchant_raises(self):
        with pytest.raises(ValueError):
            filter_records_for_merchant(_records(), "mch_nonexistent")

    def test_analyze_merchant_unknown_id_raises(self):
        with pytest.raises(ValueError):
            analyze_merchant(_records(), "mch_nonexistent")

    def test_analyze_merchant_matches_portfolio_breakdown(self):
        """The load-bearing consistency check: a merchant's own report and
        that same merchant's row in the portfolio report must agree,
        because both are produced by the same aggregation over the same
        underlying records."""
        records = _records()
        portfolio = analyze_portfolio(records)
        merchant_report = analyze_merchant(records, "mch_a")

        portfolio_row = portfolio.merchant_breakdown["mch_a"]
        assert merchant_report.won == portfolio_row.won
        assert merchant_report.lost == portfolio_row.lost
        assert merchant_report.amount_lost_inr == portfolio_row.amount_lost_inr
        assert merchant_report.loss_rate == portfolio_row.loss_rate

    def test_merchant_report_excludes_other_merchants(self):
        report = analyze_merchant(_records(), "mch_a")
        assert set(report.merchant_breakdown.keys()) == {"mch_a"}
        assert report.total_records == 3

    def test_merchant_weaknesses_scoped_correctly(self):
        report = analyze_merchant(_records(), "mch_a")
        assert report.missing_evidence_frequency_among_losses["tracking_number"] == 2
        assert report.missing_evidence_frequency_among_losses["delivery_proof"] == 1
        # mch_b's delivery_proof gap must not leak into mch_a's report
        report_b = analyze_merchant(_records(), "mch_b")
        assert report_b.missing_evidence_frequency_among_losses["delivery_proof"] == 1


class TestHighlights:
    def test_highest_loss_rate_and_exposure(self):
        report = analyze_portfolio(_records())
        # mch_b: 1/1 lost = 100% loss rate, mch_a: 2/3 = 66.7%
        assert report.merchant_with_highest_loss_rate.merchant_id == "mch_b"
        # mch_b lost ₹5000 in one case vs mch_a's ₹1500 total
        assert report.merchant_with_highest_exposure.merchant_id == "mch_b"

    def test_top_weaknesses_sorted_and_deterministic(self):
        report = analyze_portfolio(_records())
        counts = [c for _, c in report.top_weaknesses]
        assert counts == sorted(counts, reverse=True)


class TestRendering:
    def test_portfolio_render_wrong_scope_raises(self):
        report = analyze_merchant(_records(), "mch_a")
        with pytest.raises(ValueError):
            render_portfolio_markdown(report)

    def test_merchant_render_wrong_scope_raises(self):
        report = analyze_portfolio(_records())
        with pytest.raises(ValueError):
            render_merchant_markdown(report)

    def test_dispatch_render_markdown(self):
        portfolio_md = render_markdown(analyze_portfolio(_records()))
        merchant_md = render_markdown(analyze_merchant(_records(), "mch_a"))
        assert portfolio_md.startswith("# Portfolio Intelligence")
        assert merchant_md.startswith("# Merchant Root Cause Report — mch_a")

    def test_merchant_markdown_has_recommendation(self):
        md = render_markdown(analyze_merchant(_records(), "mch_a"))
        assert "## Recommendation" in md
        assert "tracking_number" in md
