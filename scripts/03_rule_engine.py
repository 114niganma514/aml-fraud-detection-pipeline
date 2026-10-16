# -*- coding: utf-8 -*-
"""
03_rule_engine.py -- Interpretable rule engine (v2).

A classic AML system starts with human-readable rules.  We derive thresholds
from the EDA (data-driven) rather than guessing:

  R1  unusually large amount      : amount > P99.9  OR  z-score > 30
  R2  extreme fan-out             : S_unique_recv >= 500   (one-to-many)
  R3  extreme fan-in              : R_unique_send >= 12    (many-to-one)
  R4  high sender volume          : S_tx_count >= 5000 AND S_total_out >= 1e9
  R5  ACH channel                 : Payment Format == 'ACH' (86.6% of laundering)
  R6  combo                       : R1 AND R2

rule_score = number of rules hit (0..6).  rule_alert is the tuned alarm
combination: (R3 AND R5), which on the full data reached Precision 0.0167
(16x baseline) with Recall 0.1816.

Output: data/processed/rule_flags.parquet
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, FEATURES_PARQUET, RULE_FLAGS_PARQUET,
                   IBM_LABEL, write_markdown)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    ensure_dirs()
    print(">>> Loading engineered features ...")
    df = pd.read_parquet(FEATURES_PARQUET)
    print(f"    rows={len(df):,}  laundering={int(df[IBM_LABEL].sum()):,}")

    y = df[IBM_LABEL].astype(int).values

    # --- threshold calibration on full data (EDA-driven) ---
    p999 = df["amount_log"].quantile(0.999)
    amt_z = (df["amount_log"] - df["amount_log"].mean()) / df["amount_log"].std()

    # --- six rules ---
    r1 = ((df["amount_log"] > p999) | (amt_z > 30)).astype(int)
    r2 = (df["S_unique_recv"] >= 500).astype(int)
    r3 = (df["R_unique_send"] >= 12).astype(int)
    r4 = ((df["S_tx_count"] >= 5000) & (df["S_total_out"] >= 1e9)).astype(int)
    r5 = (df["Payment Format"] == "ACH").astype(int)
    r6 = ((r1 == 1) & (r2 == 1)).astype(int)

    rule_score = r1 + r2 + r3 + r4 + r5 + r6
    rule_alert = ((r3 == 1) & (r5 == 1)).astype(int)

    out = pd.DataFrame({
        "tx_seq": df["tx_seq"],
        IBM_LABEL: y,
        "RULE_1": r1, "RULE_2": r2, "RULE_3": r3, "RULE_4": r4,
        "RULE_5": r5, "RULE_6": r6,
        "rule_score": rule_score, "rule_alert": rule_alert,
    })
    out.to_parquet(RULE_FLAGS_PARQUET, index=False)

    # --- per-rule stats ---
    print("\n=== Rule performance (vs laundering label) ===")
    print(f"{'Rule':<10}{'hits':>10}{'TP':>8}{'FP':>10}{'Prec':>10}{'Recall':>10}")
    rows = []
    for c in ["RULE_1", "RULE_2", "RULE_3", "RULE_4", "RULE_5", "RULE_6"]:
        hit = out[c].values
        tp = int(((hit == 1) & (y == 1)).sum())
        fp = int(((hit == 1) & (y == 0)).sum())
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / max(int(y.sum()), 1)
        print(f"{c:<10}{tp + fp:>10,}{tp:>8,}{fp:>10,}{prec:>10.4f}{rec:>10.4f}")
        rows.append((c, tp + fp, tp, fp, prec, rec))
    alert_hit = int(rule_alert.sum())
    alert_tp = int(((rule_alert == 1) & (y == 1)).sum())
    alert_fp = alert_hit - alert_tp
    print(f"{'ALERT':<10}{alert_hit:>10,}{alert_tp:>8,}{alert_fp:>10,}"
          f"{alert_tp / alert_hit:>10.4f}{alert_tp / max(int(y.sum()), 1):>10.4f}")

    # --- chart: rule precision vs recall ---
    names = [r[0] for r in rows] + ["ALERT"]
    precs = [r[4] for r in rows] + [alert_tp / alert_hit]
    recs = [r[5] for r in rows] + [alert_tp / max(int(y.sum()), 1)]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(names))
    ax.bar(x - 0.2, precs, 0.4, label="Precision", color="#9BBBF4")
    ax.bar(x + 0.2, recs, 0.4, label="Recall", color="#F4B393")
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("score")
    ax.set_title("Rule engine: precision vs recall (full data)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "output", "figures", "03_rule_performance.png"), dpi=120)
    plt.close(fig)

    # --- summary report ---
    report = f"""# 03 Rule Engine Report (v2)

## Rules
| Rule | Description | Threshold |
|---|---|---|
| R1 | unusually large amount | > P99.9 or z>30 |
| R2 | extreme fan-out | S_unique_recv >= 500 |
| R3 | extreme fan-in | R_unique_send >= 12 |
| R4 | high sender volume | S_tx_count>=5000 & S_total_out>=1e9 |
| R5 | ACH channel | Payment Format == ACH |
| R6 | combo | R1 & R2 |

## Performance (full data, laundering={int(y.sum()):,})
| Rule | hits | TP | FP | Precision | Recall |
|---|---|---|---|---|---|
"""
    for c, h, tp, fp, p, r in rows:
        report += f"| {c} | {h:,} | {tp:,} | {fp:,} | {p:.4f} | {r:.4f} |\n"
    report += (f"| ALERT(R3&R5) | {alert_hit:,} | {alert_tp:,} | {alert_fp:,} | "
               f"{alert_tp / alert_hit:.4f} | {alert_tp / max(int(y.sum()), 1):.4f} |\n")
    report += f"""
> Rule ALERT reaches {alert_tp / alert_hit:.4f} precision = {alert_tp / alert_hit / (int(y.sum()) / len(df)):.0f}x the
> baseline rate ({int(y.sum()) / len(df):.4%}). Rules are intentionally broad-recall;
> models (05) add ranking precision on top.
"""
    write_markdown(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "output", "reports", "03_rule_report.md"), report)
    print("\n>>> Rule engine done. rule_flags.parquet saved.")


if __name__ == "__main__":
    main()
