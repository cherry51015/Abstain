from __future__ import annotations

from app.domain import Action, EvidenceFacts, Tri
from app.engine.counterfactual import find_flip
from app.scoring.win_model import WinEstimate
from tests.conftest import make_case

RELEVANT = ["delivery_confirmed", "signed_by_cardholder", "customer_acknowledged_receipt"]


def estimator(weights: dict[str, float], base: float = 0.05):
    """P(win) = base + sum of weights of facts that are YES (mean over samples)."""
    def estimate(samples):
        ps = [min(1.0, base + sum(w for f, w in weights.items() if s.get(f) == Tri.YES)) for s in samples]
        return WinEstimate(mean=sum(ps) / len(ps), std=0.0, n_draws=len(ps))
    return estimate


def run(engine, catalog, estimate, facts):
    case, m = make_case(), catalog.merchant("mch_05")
    baseline = engine.decide(case, m, estimate([facts]))
    return baseline, find_flip(engine=engine, estimate=estimate, case=case, merchant=m,
                               relevant_facts=RELEVANT, fact_samples=[facts], baseline=baseline)


def test_single_fact_flip_is_found(engine, catalog):
    baseline, cf = run(engine, catalog, estimator({"delivery_confirmed": 0.7}), EvidenceFacts())
    assert baseline.action == Action.CONCEDE
    assert cf.flips and cf.evidence_needed == ["delivery_confirmed"] and cf.resulting_action == Action.CONTEST


def test_pair_is_found_when_no_single_fact_suffices(engine, catalog):
    # Contest cost is 700 on a 5000 dispute: one fact (P=0.08) is not enough, two (P=0.16) are.
    weights = {"delivery_confirmed": 0.08, "customer_acknowledged_receipt": 0.08}
    _, cf = run(engine, catalog, estimator(weights, base=0.0), EvidenceFacts())
    assert cf.flips and sorted(cf.evidence_needed) == ["customer_acknowledged_receipt", "delivery_confirmed"]


def test_no_flip_reported_honestly(engine, catalog):
    _, cf = run(engine, catalog, estimator({}), EvidenceFacts())
    assert not cf.flips and "No set" in cf.note


def test_contest_needs_no_counterfactual(engine, catalog):
    facts = EvidenceFacts(delivery_confirmed=Tri.YES)
    baseline, cf = run(engine, catalog, estimator({"delivery_confirmed": 0.9}), facts)
    assert baseline.action == Action.CONTEST and not cf.flips


def test_note_distinguishes_missing_adverse_and_unread_facts(engine, catalog):
    case, m = make_case(), catalog.merchant("mch_05")
    estimate = estimator({"delivery_confirmed": 0.7})
    for facts, unread, phrase in [
        (EvidenceFacts(), (), "obtaining evidence for delivery_confirmed"),
        (EvidenceFacts(delivery_confirmed=Tri.NO), (), "being yes instead of no"),
        (EvidenceFacts(), ("delivery_confirmed",), "confirming delivery_confirmed"),
    ]:
        baseline = engine.decide(case, m, estimate([facts]))
        cf = find_flip(engine=engine, estimate=estimate, case=case, merchant=m, relevant_facts=RELEVANT,
                       fact_samples=[facts], baseline=baseline, unread=unread)
        assert phrase in cf.note


def test_lists_every_minimal_option_ranked_and_typed(engine, catalog):
    # Either delivery or receipt alone flips; signature alone does not, and pairs containing a
    # single that already flips are not minimal. Signature is adverse ('no') in the documents.
    weights = {"delivery_confirmed": 0.5, "customer_acknowledged_receipt": 0.3, "signed_by_cardholder": 0.05}
    facts = EvidenceFacts(signed_by_cardholder=Tri.NO)
    _, cf = run(engine, catalog, estimator(weights), facts)
    evidence = [o.evidence for o in cf.options]
    assert evidence[:2] == [["delivery_confirmed"], ["customer_acknowledged_receipt"]]
    assert all(len(e) == 1 or not ({"delivery_confirmed", "customer_acknowledged_receipt"} & set(e)) for e in evidence)
    assert cf.evidence_needed == ["delivery_confirmed"]           # largest EV gain
    assert cf.options[0].kinds == {"delivery_confirmed": "missing"}
    assert "other option" in cf.note
