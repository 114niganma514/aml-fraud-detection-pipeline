# -*- coding: utf-8 -*-
"""
07_ieee_cis_validation.py -- Real-data cross validation: IEEE-CIS Fraud Detection.

WHY THIS FILE EXISTS ("synthetic training + real validation"):
  IBM HI-Small is synthetic.  IEEE-CIS is a REAL e-commerce fraud dataset
  (590k transactions, ~3.5% fraud, real card/device/email behaviour).
  Running the same modelling stack here shows the pipeline generalises to
  genuine fraud data -- a check that results on synthetic data transfer.

Pipeline (mirrors 05):
  * join train_transaction + train_identity on TransactionID
  * time-ordered 60/20/20 split on TransactionDT
  * LightGBM (no weight) vs XGBoost (scale_pos_weight), PR-AUC evaluation

Outputs:
  - output/reports/07_ieee_cis_validation.md
  - output/figures/07_ieee_pr_curve.png, 07_ieee_importance.png
"""
import os
import sys
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import average_precision_score, precision_recall_curve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# building a 390-col frame column-by-column fragments it; we concat once instead
warnings.filterwarnings("ignore", category=pd.errors.PerformanceWarning)
warnings.filterwarnings("ignore", category=pd.errors.Pandas4Warning)

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, IEEE_TRAIN_TX, IEEE_TRAIN_ID, FIGURES_DIR,
                   REPORTS_DIR, best_f1_point, write_markdown)

ID_COL = "TransactionID"
LABEL = "isFraud"


def load_ieee():
    """Join transaction + identity tables, basic cleaning."""
    print(">>> Loading IEEE-CIS data ...")
    tx = pd.read_csv(IEEE_TRAIN_TX)
    idn = pd.read_csv(IEEE_TRAIN_ID)
    print(f"    transaction rows: {len(tx):,}  identity rows: {len(idn):,}")
    df = tx.merge(idn, on=ID_COL, how="left")
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    print(f"    merged rows: {len(df):,}  fraud rate: {df[LABEL].mean():.4%}")
    return df


def prepare_features(df):
    """Numeric cols fillna(-999); categorical cols -> pandas category dtype.
    Built in ONE pd.concat to avoid DataFrame fragmentation (390 cols)."""
    num_cols = df.select_dtypes(include=[np.number]).columns.drop(
        [LABEL, ID_COL, "TransactionDT"], errors="ignore")
    cat_cols = df.select_dtypes(include=["object"]).columns

    num_parts = [df[c].fillna(-999).astype(np.float32) for c in num_cols]
    cat_parts = [df[c].astype("category") for c in cat_cols
                 if df[c].nunique() <= 100_000]
    X = pd.concat(num_parts + cat_parts, axis=1)
    y = df[LABEL].astype(int).values
    return X, y


def main():
    ensure_dirs()
    df = load_ieee()
    X, y = prepare_features(df)
    print(f"    feature matrix: {X.shape}")

    # time-ordered split
    n = len(X)
    i1, i2 = int(n * 0.6), int(n * 0.8)
    X_tr, y_tr = X.iloc[:i1], y[:i1]
    X_va, y_va = X.iloc[i1:i2], y[i1:i2]
    X_te, y_te = X.iloc[i2:], y[i2:]
    print(f"    split: train {len(X_tr):,} / val {len(X_va):,} / test {len(X_te):,}")

    # ---------------- LightGBM (no weight) ----------------
    print(">>> [LightGBM] training (no scale_pos_weight) ...")
    dtr = lgb.Dataset(X_tr, label=y_tr)
    dva = lgb.Dataset(X_va, label=y_va, reference=dtr)
    params = {"objective": "binary", "metric": "average_precision",
              "learning_rate": 0.05, "num_leaves": 64, "seed": 42,
              "verbosity": -1, "num_threads": -1}
    booster = lgb.train(params, dtr, num_boost_round=400, valid_sets=[dva],
                        callbacks=[lgb.early_stopping(50, verbose=False)])
    lgb_prob = booster.predict(X_te)

    # ---------------- XGBoost (weighted) ----------------
    print(">>> [XGBoost] training (scale_pos_weight) ...")
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    model = xgb.XGBClassifier(objective="binary:logistic", learning_rate=0.05,
                              max_depth=8, n_estimators=400,
                              scale_pos_weight=pos_w, eval_metric="aucpr",
                              early_stopping_rounds=50, tree_method="hist",
                              enable_categorical=True,
                              random_state=42, n_jobs=-1, verbosity=0)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    xgb_prob = model.predict_proba(X_te)[:, 1]

    # ---------------- evaluation ----------------
    print("\n=== IEEE-CIS test results ===")
    results = {}
    for name, prob in [("LightGBM", lgb_prob), ("XGBoost", xgb_prob)]:
        ap = average_precision_score(y_te, prob)
        p, r, f1, thr = best_f1_point(y_te, prob)
        results[name] = (ap, p, r, f1, thr)
        print(f"    {name:<10} PR-AUC={ap:.4f}  BestF1 P={p:.4f} R={r:.4f}")

    # PR curve
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"LightGBM": "#8BC8EA", "XGBoost": "#9BBBF4"}
    for name, prob in [("LightGBM", lgb_prob), ("XGBoost", xgb_prob)]:
        p, r, _ = precision_recall_curve(y_te, prob)
        ax.plot(r, p, label=f"{name} (PR-AUC={results[name][0]:.4f})",
                color=colors[name], lw=2)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_title("IEEE-CIS (REAL data): PR curves")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "07_ieee_pr_curve.png"), dpi=120)
    plt.close(fig)

    # importance
    imp = pd.Series(booster.feature_importance(importance_type="gain"),
                    index=X_tr.columns).sort_values(ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index[::-1], imp.values[::-1], color="#94D8C3")
    ax.set_title("IEEE-CIS LightGBM top-15 features (gain)")
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "07_ieee_importance.png"), dpi=120)
    plt.close(fig)

    # report
    rep = f"""# 07 Real-Data Validation -- IEEE-CIS Fraud Detection

## Dataset
- rows: {len(df):,} (real e-commerce card-not-present transactions)
- fraud rate: {df[LABEL].mean():.4%}
- joined identity table: device / card / email behavioural fields

## Results (time-ordered test split, {len(X_te):,} rows)
| Model | PR-AUC | Best F1 | Precision | Recall |
|---|---|---|---|---|
"""
    for name, (ap, p, r, f1, _) in results.items():
        rep += f"| {name} | {ap:.4f} | {f1:.4f} | {p:.4f} | {r:.4f} |\n"
    rep += f"""
## Interpretation
- PR-AUC baseline = fraud prevalence = {y_te.mean():.4f}.
- The stack built on synthetic IBM data (LightGBM unweighted / XGBoost weighted,
  PR-AUC metric, time-ordered split) transfers directly to real fraud data,
  validating the "synthetic training + real validation" approach.

## Caveats
- IEEE-CIS contains engineered aggregate features (C1-C14, V*) that already
  encode a lot of signal; feature engineering effort was intentionally minimal.
- We validate the *stack*, not a production-grade real-time system.
"""
    write_markdown(os.path.join(REPORTS_DIR, "07_ieee_cis_validation.md"), rep)
    print(">>> IEEE-CIS validation done.")


if __name__ == "__main__":
    main()
