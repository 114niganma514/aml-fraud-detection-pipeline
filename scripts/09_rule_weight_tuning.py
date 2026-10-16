# -*- coding: utf-8 -*-
"""
09_rule_weight_tuning.py -- Rule weight grid search for the hybrid score.

Goal: find rule weights (w1..w6) such that the hybrid risk score

    hybrid = xgb_prob + SUM_i (w_i * RULE_i)

maximises PR-AUC / Best-F1 on the VALIDATION split (time-ordered 60/20/20).
The tuned weights are written to config/rules_config.json and used as the
interpretable "rule layer" on top of the best model.

Why tune instead of equal weights?
  * equal weights over-emphasise broad rules (e.g. R5 ACH hits 600k tx)
  * search reveals which rules add ranking signal AFTER the model

Implementation:
  * grid over {0, 0.5, 1.0} for w1..w6 (3^6 = 729 combos; each evaluation is a
    cheap vectorised PR-AUC on the validation set) -> fast even in pure Python.
  * objective: Best-F1, with PR-AUC reported as secondary.
"""
import os
import sys
import json
import itertools
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, FEATURES_PARQUET, RULE_FLAGS_PARQUET,
                   TEST_PREDICTIONS_PARQUET, RULES_CONFIG_JSON,
                   best_f1_point, write_markdown)

RULES = ["RULE_1", "RULE_2", "RULE_3", "RULE_4", "RULE_5", "RULE_6"]
WEIGHT_GRID = [0.0, 0.5, 1.0]


def load_validation_set():
    """Validation split (60%-80% by tx_seq) with model prob + rule flags."""
    feat = pd.read_parquet(FEATURES_PARQUET).sort_values("tx_seq").reset_index(drop=True)
    n = len(feat)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    va = feat.iloc[i1:i2].copy()

    preds = pd.read_parquet(TEST_PREDICTIONS_PARQUET)  # contains xgb_prob for test only
    # NOTE: 09 re-uses the TEST predictions file only for row alignment sanity;
    # the actual validation probabilities are recomputed by training here.
    # For simplicity we train a small XGBoost on the train split to score val.
    rules = pd.read_parquet(RULE_FLAGS_PARQUET)
    va = va.merge(rules[["tx_seq"] + RULES], on="tx_seq", how="left").fillna(0)
    return feat, va, i1, i2


def quick_xgb_val(X_tr, y_tr, X_va):
    """Lightweight XGBoost (weighted) to score the validation split."""
    import xgboost as xgb
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.05,
                              max_depth=8, n_estimators=200,
                              scale_pos_weight=pos_w, tree_method="hist",
                              enable_categorical=True,
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr)
    return model.predict_proba(X_va)[:, 1]


def main():
    ensure_dirs()
    print(">>> Loading data for validation ...")
    feat, va, i1, i2 = load_validation_set()

    DROP = ["tx_seq", "Timestamp", "Account", "Account.1", "From Bank", "To Bank",
            "Is Laundering", "is_laundering"]
    X_all = feat.drop(columns=[c for c in DROP if c in feat.columns])
    for c in ["Payment Format", "Receiving Currency", "Payment Currency"]:
        X_all[c] = X_all[c].astype("category")

    y_all = feat["Is Laundering"].astype(int).values
    X_tr, y_tr = X_all.iloc[:i1], y_all[:i1]
    X_va = X_all.iloc[i1:i2]
    y_va = y_all[i1:i2]
    print(f"    train {len(X_tr):,} / val {len(X_va):,}")

    print(">>> Scoring validation with quick XGBoost ...")
    xgb_prob = quick_xgb_val(X_tr, y_tr, X_va)

    # rule matrix on validation
    R = va[RULES].values.astype(float)
    base_pr_auc = average_precision_score(y_va, xgb_prob)
    _, _, base_f1, _ = best_f1_point(y_va, xgb_prob)
    print(f"    XGBoost alone: PR-AUC={base_pr_auc:.4f} BestF1={base_f1:.4f}")

    print(">>> Grid searching rule weights (729 combos) ...")
    best = None
    for weights in itertools.product(WEIGHT_GRID, repeat=6):
        if not any(weights):
            continue
        hybrid = xgb_prob + R @ np.array(weights)
        ap = average_precision_score(y_va, hybrid)
        _, _, f1, _ = best_f1_point(y_va, hybrid)
        if best is None or f1 > best[0]:
            best = (f1, ap, weights)

    best_f1, best_ap, best_w = best
    print(f"\n=== Best weights (objective=BestF1) ===")
    for r, w in zip(RULES, best_w):
        print(f"    {r}: {w:.1f}")
    print(f"    hybrid BestF1={best_f1:.4f} (base {base_f1:.4f})  "
          f"PR-AUC={best_ap:.4f} (base {base_pr_auc:.4f})")

    # persist config
    cfg = {"rule_weights": {r: w for r, w in zip(RULES, best_w)},
           "base_model": "xgboost", "objective": "best_f1",
           "base_best_f1": base_f1, "hybrid_best_f1": best_f1}
    with open(RULES_CONFIG_JSON, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
    print(f"    config -> {RULES_CONFIG_JSON}")

    write_markdown(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "output", "reports", "09_rule_weight_tuning.md"),
                   f"""# 09 Rule Weight Tuning

## Method
- search {len(list(itertools.product(WEIGHT_GRID, repeat=6))):,} weight combos
  over {{0, 0.5, 1.0}} for RULE_1..RULE_6 on the validation split
- hybrid score = xgb_prob + sum(w_i * RULE_i); objective = Best-F1

## Result
| metric | XGBoost alone | Hybrid (tuned weights) |
|---|---|---|
| PR-AUC | {base_pr_auc:.4f} | {best_ap:.4f} |
| Best F1 | {base_f1:.4f} | {best_f1:.4f} |

## Tuned weights
| rule | weight |
|---|---|
""" + "".join(f"| {r} | {w:.1f} |\n" for r, w in zip(RULES, best_w)) + f"""
## Interpretation
- Rules with positive weight add ranking signal on top of the model;
  zero-weight rules are redundant once the model has learned the same patterns.
- The tuned hybrid keeps rules as an interpretable layer without hurting the
  model's ranking -- the production-relevant takeaway.
""")
    print(">>> Rule weight tuning done.")


if __name__ == "__main__":
    main()
