# -*- coding: utf-8 -*-
"""
06_evaluation_report.py -- Final evaluation: rule vs model vs hybrid.

Evaluates on the time-ordered test split (last 20%):
  1. Rule-only        : rule_score normalised to [0,1]
  2. LightGBM         : unweighted LightGBM probability
  3. XGBoost          : weighted XGBoost probability
  4. Hybrid           : XGBoost + strong-rule override
                        (linear blending HURTS here -- see note below)

EMPIRICAL NOTE (why not linear blending):
  A naive 0.7*model + 0.3*rule_norm drops PR-AUC from 0.48 to 0.34, because
  the rule score adds a near-constant offset to a large share of transactions,
  destroying the model's ranking. The hybrid used here therefore treats rules
  as an interpretable hard-override: transactions hitting >=3 rules get their
  risk score floored at 0.6. This preserves model ranking while keeping rules
  as a compliance/explainability layer.

Outputs:
  - output/figures/06_pr_curve.png, 06_confusion.png
  - output/reports/06_evaluation_report.md
"""
import os
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, FEATURES_PARQUET, RULE_FLAGS_PARQUET, IBM_LABEL,
                   TEST_PREDICTIONS_PARQUET, FIGURES_DIR, REPORTS_DIR,
                   best_f1_point, write_markdown)


def load_test_set():
    """Rebuild the test split (last 20% by tx_seq) and attach rules + model probs."""
    df = pd.read_parquet(FEATURES_PARQUET).sort_values("tx_seq").reset_index(drop=True)
    n = len(df)
    i2 = int(n * 0.8)
    test_df = df.iloc[i2:].copy()

    preds = pd.read_parquet(TEST_PREDICTIONS_PARQUET)
    test_df["lgb_prob"] = preds["lgb_prob"].values
    test_df["xgb_prob"] = preds["xgb_prob"].values

    rules = pd.read_parquet(RULE_FLAGS_PARQUET)
    test_df = test_df.merge(rules[["tx_seq", "rule_score", "rule_alert"]],
                            on="tx_seq", how="left")
    test_df["rule_score"] = test_df["rule_score"].fillna(0)
    test_df["rule_alert"] = test_df["rule_alert"].fillna(0)
    return test_df


def main():
    ensure_dirs()
    print(">>> Rebuilding test set (rules + model predictions aligned) ...")
    test = load_test_set()
    y = test[IBM_LABEL].astype(int).values
    print(f"    test rows={len(test):,}  laundering rate={y.mean():.4%}")

    # --- scoring schemes ---
    rule_norm = (test["rule_score"] / 6.0).values
    lgb_prob = test["lgb_prob"].values
    xgb_prob = test["xgb_prob"].values

    hybrid_prob = xgb_prob.copy()
    strong = (test["rule_score"] >= 3).values
    hybrid_prob[strong] = np.maximum(hybrid_prob[strong], 0.6)

    schemes = {
        "Rule-only": rule_norm,
        "LightGBM": lgb_prob,
        "XGBoost": xgb_prob,
        "Hybrid(XGB+Rule)": hybrid_prob,
    }

    # --- PR curves ---
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = {"Rule-only": "#E4D48F", "LightGBM": "#8BC8EA",
              "XGBoost": "#9BBBF4", "Hybrid(XGB+Rule)": "#52C41A"}
    results = {}
    for name, score in schemes.items():
        p, r, _ = precision_recall_curve(y, score)
        ap = average_precision_score(y, score)
        results[name] = (ap, *best_f1_point(y, score))
        ax.plot(r, p, label=f"{name} (PR-AUC={ap:.4f})", color=colors[name], lw=2)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall: Rule vs LightGBM vs XGBoost vs Hybrid")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "06_pr_curve.png"), dpi=120)
    plt.close(fig)

    # --- confusion matrix for the best scheme (XGBoost) at best-F1 threshold ---
    ap_h, p_h, r_h, f1_h, thr_h = results["XGBoost"]
    pred = (xgb_prob >= thr_h).astype(int)
    cm = confusion_matrix(y, pred)
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Launder"])
    ax.set_yticklabels(["Legit", "Launder"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", fontsize=12)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"XGBoost confusion matrix (thr={thr_h:.3f})")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "06_confusion.png"), dpi=120)
    plt.close(fig)

    # --- report ---
    rep = f"""# 06 Final Evaluation Report

- test rows: {len(test):,}
- test laundering rate: {y.mean():.4%}  (baseline PR-AUC = {y.mean():.4f})

## 1. Scheme comparison (test set)
| Scheme | PR-AUC | Best F1 | Precision@BestF1 | Recall@BestF1 |
|---|---|---|---|---|
"""
    for name, (ap, p, r, f1, _) in results.items():
        rep += f"| {name} | {ap:.4f} | {f1:.4f} | {p:.4f} | {r:.4f} |\n"

    tp, fp, fn, tn = int(cm[1, 1]), int(cm[0, 1]), int(cm[1, 0]), int(cm[0, 0])
    rep += f"""
## 2. Best model business metrics (XGBoost @ best-F1 threshold {thr_h:.3f})
- TP (correctly flagged launderers): {tp:,}
- FP (false positives): {fp:,}
- FN (missed launderers): {fn:,}
- TN: {tn:,}
- Precision {p_h:.4f} | Recall {r_h:.4f} | F1 {f1_h:.4f}

## 3. Key insights
1. Extreme imbalance (0.1%) makes accuracy meaningless; PR-AUC / Recall are the metrics.
2. XGBoost with scale_pos_weight is robust and reaches PR-AUC {results['XGBoost'][0]:.4f}.
3. LightGBM must NOT use scale_pos_weight on this data (ranking collapses).
4. Linear rule blending hurts the model; rules serve as an explainability/override layer.
"""
    write_markdown(os.path.join(REPORTS_DIR, "06_evaluation_report.md"), rep)

    print("\n=== Final comparison (test set) ===")
    print(f"{'Scheme':<18}{'PR-AUC':>10}{'BestF1':>10}{'P':>8}{'R':>8}")
    for name, (ap, p, r, f1, _) in results.items():
        print(f"{name:<18}{ap:>10.4f}{f1:>10.4f}{p:>8.4f}{r:>8.4f}")
    print(">>> Evaluation done.")


if __name__ == "__main__":
    main()
