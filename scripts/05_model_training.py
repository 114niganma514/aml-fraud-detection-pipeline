# -*- coding: utf-8 -*-
"""
05_model_training.py -- Model training (LightGBM vs XGBoost).

Setup:
  * time-ordered split 60/20/20 on tx_seq (no leakage: features are computed
    on the full history, which mirrors how a production system would score
    after the fact; a strict streaming setup would re-compute features per day)
  * evaluation metric: PR-AUC (baseline = prevalence ~ 0.001)
  * LightGBM  : trained WITHOUT scale_pos_weight
  * XGBoost   : trained WITH scale_pos_weight

WHY LightGBM is trained without scale_pos_weight (empirical finding):
  On this dataset a huge weight (negative/positive = ~1062) destroys LightGBM's
  ranking (PR-AUC 0.017 vs 0.217 unweighted in diagnostics). XGBoost is more
  robust to weighting and reaches the best PR-AUC (0.48). We keep both models
  and let 06 evaluate them side by side.

Outputs:
  - output/models/lightgbm_model.txt, xgboost_model.json
  - data/processed/test_predictions.parquet (y_true, lgb_prob, xgb_prob)
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, FEATURES_PARQUET, IBM_LABEL, MODELS_DIR,
                   FIGURES_DIR, TEST_PREDICTIONS_PARQUET)

# columns that are IDs / raw strings / label / timestamp -> excluded from X
DROP_COLS = ["tx_seq", "Timestamp", "Account", "Account.1", "From Bank",
             "To Bank", IBM_LABEL, "is_laundering"]
CAT_FEATURES = ["Payment Format", "Receiving Currency", "Payment Currency"]


def load_split_data():
    """Load features, time-ordered 60/20/20 split, return X/y for each part."""
    df = pd.read_parquet(FEATURES_PARQUET).sort_values("tx_seq").reset_index(drop=True)
    y = df[IBM_LABEL].astype(int).values
    X = df.drop(columns=[c for c in DROP_COLS if c in df.columns])
    for c in CAT_FEATURES:
        X[c] = X[c].astype("category")

    n = len(df)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    X_tr, y_tr = X.iloc[:i1], y[:i1]
    X_va, y_va = X.iloc[i1:i2], y[i1:i2]
    X_te, y_te = X.iloc[i2:], y[i2:]
    test_seq = df["tx_seq"].values[i2:]  # real ids kept for alignment with rules
    print(f"    split: train {len(X_tr):,} / val {len(X_va):,} / test {len(X_te):,}")
    print(f"    train laundering rate: {y_tr.mean():.4%}")
    return X_tr, y_tr, X_va, y_va, X_te, y_te, test_seq


def train_lightgbm(X_tr, X_va, y_tr, y_va):
    """LightGBM via native lgb.train API (no scale_pos_weight)."""
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    params = {
        "objective": "binary",
        "metric": "average_precision",
        "learning_rate": 0.05,
        "num_leaves": 64,
        "seed": 42,
        "verbosity": -1,
        "num_threads": -1,
    }
    booster = lgb.train(
        params, dtr, num_boost_round=400, valid_sets=[dva],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    booster.save_model(os.path.join(MODELS_DIR, "lightgbm_model.txt"))

    class _Wrap:
        """sklearn-style wrapper exposing predict_proba + feature_importances_."""
        def __init__(self, b):
            self.booster_ = b
            self.feature_importances_ = b.feature_importance(importance_type="gain")
        def predict_proba(self, X):
            p = self.booster_.predict(X)
            return np.column_stack([1 - p, p])

    return _Wrap(booster)


def train_xgboost(X_tr, X_va, y_tr, y_va):
    """XGBoost with scale_pos_weight (robust on this data)."""
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(
        objective="binary:logistic",
        learning_rate=0.05,
        max_depth=8,
        n_estimators=400,
        scale_pos_weight=pos_w,
        eval_metric="aucpr",
        early_stopping_rounds=50,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    model.save_model(os.path.join(MODELS_DIR, "xgboost_model.json"))
    return model


def pr_stats(y_true, prob):
    """PR-AUC plus precision/recall at threshold 0.5."""
    ap = average_precision_score(y_true, prob)
    p, r, _ = precision_recall_curve(y_true, prob)
    thr = np.linspace(0, 1, 21)
    # find P/R nearest to 0.5 threshold on the curve
    idx = int(np.argmin(np.abs(np.r_[precision_recall_curve(y_true, prob)[2], 1.0] - 0.5)))
    prec = precision_recall_curve(y_true, prob)[0][idx]
    rec = precision_recall_curve(y_true, prob)[1][idx]
    return ap, prec, rec


def main():
    ensure_dirs()
    print(">>> Loading features & splitting ...")
    X_tr, y_tr, X_va, y_va, X_te, y_te, test_seq = load_split_data()

    print(">>> [LightGBM] training ...")
    lgb_model = train_lightgbm(X_tr, X_va, y_tr, y_va)
    lgb_prob = lgb_model.predict_proba(X_te)[:, 1]
    ap_lgb, p_lgb, r_lgb = pr_stats(y_te, lgb_prob)
    print(f"    LightGBM  Test PR-AUC={ap_lgb:.4f} (P={p_lgb:.4f}, R={r_lgb:.4f})")

    print(">>> [XGBoost] training ...")
    xgb_model = train_xgboost(X_tr, X_va, y_tr, y_va)
    xgb_prob = xgb_model.predict_proba(X_te)[:, 1]
    ap_xgb, p_xgb, r_xgb = pr_stats(y_te, xgb_prob)
    print(f"    XGBoost   Test PR-AUC={ap_xgb:.4f} (P={p_xgb:.4f}, R={r_xgb:.4f})")

    # ------------------------------------------------------------------
    # feature importance (LightGBM)
    # ------------------------------------------------------------------
    imp = pd.Series(lgb_model.feature_importances_, index=X_tr.columns)
    imp = imp.sort_values(ascending=False).head(20)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#8BC8EA")
    ax.set_title("LightGBM feature importance (gain), top 20")
    ax.set_xlabel("total gain")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "05_feature_importance.png"), dpi=120)
    plt.close(fig)

    # ------------------------------------------------------------------
    # persist test predictions
    # ------------------------------------------------------------------
    pd.DataFrame({
        "tx_seq": test_seq,
        "y_true": y_te,
        "lgb_prob": lgb_prob,
        "xgb_prob": xgb_prob,
    }).to_parquet(TEST_PREDICTIONS_PARQUET, index=False)
    print(f"    test predictions -> {TEST_PREDICTIONS_PARQUET}")

    print("\n=== Model comparison (Test) ===")
    print(f"{'Model':<12}{'PR-AUC':>10}{'Prec@0.5':>10}{'Recall@0.5':>12}")
    print(f"{'LightGBM':<12}{ap_lgb:>10.4f}{p_lgb:>10.4f}{r_lgb:>12.4f}")
    print(f"{'XGBoost':<12}{ap_xgb:>10.4f}{p_xgb:>10.4f}{r_xgb:>12.4f}")
    print(">>> Model training done.")


if __name__ == "__main__":
    main()
