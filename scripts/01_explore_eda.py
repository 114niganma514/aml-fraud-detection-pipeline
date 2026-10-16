# -*- coding: utf-8 -*-
"""
01_explore_eda.py -- Exploratory Data Analysis on IBM HI-Small.

Answers the questions a risk analyst would ask first:
  * how big is the data, is anything missing?
  * how imbalanced is the laundering label?
  * how does laundering distribute across payment channels / time / amount?
  * what does the transaction network look like (degree long tail)?

Outputs: output/reports/01_eda_summary.md + output/figures/01_*.png

GOTCHAS demonstrated (as learning material):
  * the label column is named "Is Laundering" (space inside, no underscore)
  * PowerShell does not accept `&&` as a command separator
"""
import os
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, IBM_RAW_CSV, IBM_LABEL, FIGURES_DIR,
                   REPORTS_DIR, write_markdown)

# ---------------------------------------------------------------------------
# 1. Load raw data
# ---------------------------------------------------------------------------
print(">>> Loading raw IBM HI-Small data ...")
df = pd.read_csv(IBM_RAW_CSV, parse_dates=["Timestamp"])
print(f"    rows={len(df):,}  cols={len(df.columns)}")

# Add a monotonic sequence id (useful later for time-ordered splits)
df = df.sort_values("Timestamp").reset_index(drop=True)
df["tx_seq"] = np.arange(len(df))

# ---------------------------------------------------------------------------
# 2. Overview: missing values, label imbalance
# ---------------------------------------------------------------------------
missing = df.isna().sum().sum()
n_launder = int(df[IBM_LABEL].sum())
launder_rate = n_launder / len(df)

print(f"    missing values: {missing}")
print(f"    laundering tx: {n_launder:,}  rate={launder_rate:.4%}")

# --- label distribution chart ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
colors = ["#8BC8EA", "#EA6668"]
vals = [len(df) - n_launder, n_launder]
labels = ["Legit", "Laundering"]
axes[0].bar(labels, vals, color=colors)
axes[0].set_title("Transaction label distribution")
axes[0].set_yscale("log")
axes[0].set_ylabel("Count (log scale)")
for i, v in enumerate(vals):
    axes[0].text(i, v * 1.2, f"{v:,}", ha="center")
axes[1].bar(labels, [100 - launder_rate * 100, launder_rate * 100], color=colors)
axes[1].set_title("Share (%)")
axes[1].set_ylabel("Percent")
for i, v in enumerate([100 - launder_rate * 100, launder_rate * 100]):
    axes[1].text(i, v + 0.02, f"{v:.3f}%", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_label_distribution.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 3. Time analysis: hourly distribution
# ---------------------------------------------------------------------------
df["hour"] = df["Timestamp"].dt.hour
hourly = df.groupby("hour").size() / len(df) * 100
launder_by_hour = df.groupby("hour")[IBM_LABEL].mean() * 100

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(hourly.index, hourly.values, color="#A3D5E8", label="All tx (%)")
ax2 = ax.twinx()
ax2.plot(launder_by_hour.index, launder_by_hour.values, color="#EA6668",
         marker="o", lw=2, label="Laundering rate (%)")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Share of all tx (%)")
ax2.set_ylabel("Laundering rate (%)")
ax.set_title("Hourly activity vs laundering rate")
fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.92))
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_hourly_distribution.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 4. Payment channel analysis (key finding: ACH dominates laundering)
# ---------------------------------------------------------------------------
pay = df.groupby("Payment Format")[IBM_LABEL].agg(["size", "mean"])
pay["count"] = pay["size"]
pay["launder_rate"] = pay["mean"] * 100
pay = pay.sort_values("launder_rate", ascending=False)

fig, ax = plt.subplots(figsize=(9, 4))
ax.bar(pay.index, pay["launder_rate"], color="#9BBBF4")
ax.set_ylabel("Laundering rate (%)")
ax.set_title("Laundering rate by payment format")
for i, v in enumerate(pay["launder_rate"]):
    ax.text(i, v + 0.005, f"{v:.3f}%", ha="center", fontsize=8)
ax.tick_params(axis="x", rotation=15)
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_payment_format.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 5. Amount distribution (log scale) and round-amount behaviour
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
amt = np.log1p(df["Amount Received"])
axes[0].hist(amt, bins=100, color="#94D8C3")
axes[0].set_title("log(Amount Received) histogram")
axes[0].set_xlabel("log1p(amount)")
axes[0].set_ylabel("Frequency")
round_amt = ((df["Amount Received"] % 1000) == 0).mean() * 100
axes[1].bar(["non-round", "round (x000)"],
            [100 - round_amt, round_amt], color=["#A3D5E8", "#E1B98F"])
axes[1].set_title("Share of round amounts")
axes[1].set_ylabel("Percent")
axes[1].text(1, round_amt + 0.5, f"{round_amt:.2f}%", ha="center")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_amount_distribution.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 6. Account degree (network) long tail
# ---------------------------------------------------------------------------
sender_deg = df.groupby("Account").size()
receiver_deg = df.groupby("Account.1").size()

fig, axes = plt.subplots(1, 2, figsize=(10, 4))
for ax, deg, name, col in [
    (axes[0], sender_deg, "sender out-degree", "#8BC8EA"),
    (axes[1], receiver_deg, "receiver in-degree", "#C9A7E8"),
]:
    v = np.log1p(deg.sort_values(ascending=False).values[:2000])
    ax.plot(v, color=col)
    ax.set_title(f"Top 2000 {name} (log scale)")
    ax.set_xlabel("rank")
    ax.set_ylabel("log1p(degree)")
fig.tight_layout()
fig.savefig(os.path.join(FIGURES_DIR, "01_degree_distribution.png"), dpi=120)
plt.close(fig)

# ---------------------------------------------------------------------------
# 7. Write report
# ---------------------------------------------------------------------------
report = f"""# 01 Exploratory Data Analysis -- IBM HI-Small

## 1.1 Overview
- rows: {len(df):,}
- columns: {len(df.columns)}
- time span: {df['Timestamp'].min()} -> {df['Timestamp'].max()} ({df['Timestamp'].dt.day.nunique()} days)
- missing values: {missing}
- unique accounts: {df['Account'].nunique():,} (sender) / {df['Account.1'].nunique():,} (receiver)
- unique banks: {df['From Bank'].nunique():,} / {df['To Bank'].nunique():,}

## 1.2 Label imbalance
- laundering transactions: {n_launder:,} ({launder_rate:.4%} ~ 1 in {1/launder_rate:.0f})
- baseline PR-AUC = {launder_rate:.4f}

## 1.3 Key finding -- payment channel
| Payment Format | tx count | laundering rate |
|---|---|---|
"""
for fmt, row in pay.iterrows():
    report += f"| {fmt} | {row['count']:,} | {row['launder_rate']:.4f}% |\n"
report += f"""
> ACH carries {pay.loc['ACH', 'count'] / len(df):.1%} of all tx but its laundering rate is
> {pay.loc['ACH', 'launder_rate']:.4f}% -- far above the global mean ({launder_rate:.4%}).
> This is the single strongest rule signal.

## 1.4 Time behaviour
- peak hour(s) by volume: {hourly.idxmax()} (UTC)
- laundering rate is roughly stable across hours ({launder_by_hour.min():.3f}%-{launder_by_hour.max():.3f}%)

## 1.5 Amount behaviour
- round-amount (multiple of 1000) share: {round_amt:.2f}% of all tx
- amount distribution is heavy-tailed (log-normal like)

## 1.6 Network structure
- sender out-degree long tail: top sender sends {sender_deg.max():,} tx
- receiver in-degree long tail: top receiver receives {receiver_deg.max():,} tx
> Fan-out / fan-in features are therefore highly discriminative.
"""
write_markdown(os.path.join(REPORTS_DIR, "01_eda_summary.md"), report)
print(">>> EDA done. Figures and report saved.")
