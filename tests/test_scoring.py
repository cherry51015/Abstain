from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, roc_auc_score

from app.domain import Category, EvidenceFacts, Tri
from app.scoring.metrics import auc, brier, ece
from app.scoring.win_model import featurize


def test_metrics_match_sklearn():
    rng = np.random.default_rng(1)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)
    assert brier(p, y) == pytest.approx(brier_score_loss(y, p))
    assert auc(p, y) == pytest.approx(roc_auc_score(y, p))


def test_perfectly_calibrated_forecast_has_low_ece():
    rng = np.random.default_rng(2)
    p = rng.random(20_000)
    y = (rng.random(20_000) < p).astype(float)
    assert ece(p, y) < 0.02


def test_featurize_encodes_tri_state():
    x = featurize(EvidenceFacts(delivery_confirmed=Tri.YES, signed_by_cardholder=Tri.NO),
                  ["delivery_confirmed", "signed_by_cardholder"], merchant_win_rate=0.6)
    assert list(x) == [1.0, 0.0, 0.0, 1.0, pytest.approx(0.1)]


def test_model_artifact_learned_sensible_directions(model):
    m = model.categories[Category.NOT_RECEIVED]
    coef = dict(zip(m.feature_names, m.coefs.mean(axis=0), strict=False))
    assert coef["delivery_confirmed=yes"] > 0 > coef["customer_acknowledged_receipt=no"]


def test_favourable_evidence_raises_p_win(model, catalog):
    merchant = catalog.merchant("mch_05")
    weak = model.estimate(Category.NOT_RECEIVED, [EvidenceFacts()], merchant)
    strong = model.estimate(Category.NOT_RECEIVED, [EvidenceFacts(delivery_confirmed=Tri.YES,
                                                                  customer_acknowledged_receipt=Tri.YES)], merchant)
    assert strong.mean > weak.mean + 0.3


def test_imputation_widens_spread_and_is_seeded(model, catalog):
    merchant = catalog.merchant("mch_05")
    base = [EvidenceFacts()]
    imputed = model.impute(base, ["delivery_confirmed", "customer_acknowledged_receipt"], seed=7)
    assert len(imputed) == 16
    assert model.estimate(Category.NOT_RECEIVED, imputed, merchant).std > \
        model.estimate(Category.NOT_RECEIVED, base, merchant).std
    assert imputed == model.impute(base, ["delivery_confirmed", "customer_acknowledged_receipt"], seed=7)
