"""
Train the win-probability model and write models/win_model.json.

Training labels are dispute outcomes; training features are the ground-truth
facts, standing in for analyst-reviewed historical cases. At inference the
facts come from an extractor, so extraction errors propagate into P(win)
and are measured separately by the eval harness.

Usage: python scripts/train_win_model.py [--members 30] [--C 1.0] [--out models/win_model.json]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.catalog import load_catalog  # noqa: E402
from app.domain import FACT_NAMES, FACT_SOURCES, Tri  # noqa: E402
from app.scoring.metrics import auc, brier, ece, log_loss  # noqa: E402
from app.scoring.win_model import WinModel, feature_names, featurize  # noqa: E402
from scripts._dataset import load_split  # noqa: E402


def design(rows, catalog):
    by_cat = defaultdict(lambda: ([], []))
    for r in rows:
        rc = catalog.reason_code(r["reason_code"])
        merchant = catalog.merchant(r["merchant_id"])
        X, y = by_cat[rc.category]
        X.append(featurize(r["facts"], rc.relevant_facts, merchant.historical_win_rate))
        y.append(1.0 if r["outcome"] == "won" else 0.0)
    return {c: (np.array(X), np.array(y)) for c, (X, y) in by_cat.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", type=int, default=30)
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=ROOT / "models" / "win_model.json")
    args = ap.parse_args()

    catalog = load_catalog()
    relevant = {rc.category: rc.relevant_facts for rc in catalog.reason_codes.values()}
    train = design(load_split("train"), catalog)
    rng = np.random.default_rng(args.seed)

    categories = {}
    for cat, (X, y) in train.items():
        intercepts, coefs = [], []
        for _ in range(args.members):
            idx = rng.integers(0, len(y), len(y))
            clf = LogisticRegression(C=args.C, max_iter=1000).fit(X[idx], y[idx])
            intercepts.append(float(clf.intercept_[0]))
            coefs.append([round(float(c), 5) for c in clf.coef_[0]])
        categories[cat.value] = {
            "relevant_facts": relevant[cat],
            "feature_names": feature_names(relevant[cat]),
            "n_train": int(len(y)),
            "intercepts": [round(v, 5) for v in intercepts],
            "coefs": coefs,
        }

    # Priors for facts a present document may or may not address, used to
    # impute facts the rules could not read (see app/service.py).
    priors = {}
    for f in FACT_NAMES:
        covered = [r["facts"].get(f) for r in load_split("train")
                   if any(d.doc_type in FACT_SOURCES[f] for d in r["case"].documents)]
        known = [v for v in covered if v != Tri.UNKNOWN]
        priors[f] = {"p_addressed": round(len(known) / max(1, len(covered)), 4),
                     "p_yes": round(sum(v == Tri.YES for v in known) / max(1, len(known)), 4)}

    body = {"categories": categories, "fact_priors": priors}
    version = "wm-" + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()[:10]
    artifact = {"version": version, **body, "metadata": {"members": args.members, "C": args.C}}

    # Validation metrics are computed from the artifact itself, through the
    # same inference path the API uses.
    tmp = args.out
    tmp.parent.mkdir(exist_ok=True)
    tmp.write_text(json.dumps(artifact, indent=1), encoding="utf-8", newline="\n")
    model = WinModel.load(tmp)
    p, y = [], []
    for r in load_split("val"):
        rc = catalog.reason_code(r["reason_code"])
        est = model.estimate(rc.category, [r["facts"]], catalog.merchant(r["merchant_id"]))
        p.append(est.mean)
        y.append(1.0 if r["outcome"] == "won" else 0.0)
    p, y = np.array(p), np.array(y)
    val = {"n": len(y), "brier": round(brier(p, y), 4), "log_loss": round(log_loss(p, y), 4),
           "ece": round(ece(p, y), 4), "auc": round(auc(p, y), 4)}
    artifact["metadata"]["val_metrics"] = val
    tmp.write_text(json.dumps(artifact, indent=1), encoding="utf-8", newline="\n")
    print(f"wrote {tmp} version={version} val={val}")


if __name__ == "__main__":
    main()
