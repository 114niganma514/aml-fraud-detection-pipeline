# -*- coding: utf-8 -*-
"""
02_feature_engineering.py -- Feature engineering for IBM HI-Small.

Builds two artifacts:
  1. transactions_features.parquet (5.08M rows x 44 cols)
     - transaction-level features (time, amount, currency, format)
     - account-level network features (sender side S_*, receiver side R_*)
     - deviation interactions (amount vs account baselines)
  2. account_profiles.parquet (496,995 accounts)
     - per-account aggregated profiles (used later by 04_customer_profiling.py)

Naming convention:
  S_*  = sender-side (Account) aggregated features  -- e.g. S_unique_recv = fan-out
  R_*  = receiver-side (Account.1) features        -- e.g. R_unique_send = fan-in

GOTCHA: the label is "Is Laundering" (with a space). All other code must use
IBM_LABEL from utils, never "Is_Laundering".
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (ensure_dirs, IBM_RAW_CSV, IBM_LABEL, FEATURES_PARQUET,
                   PROFILES_PARQUET)

# ---------------------------------------------------------------------------
# 1. Load and prepare raw data
# ---------------------------------------------------------------------------
print(">>> Loading raw data ...")
df = pd.read_csv(IBM_RAW_CSV, parse_dates=["Timestamp"])
print(f"    rows={len(df):,}")

df = df.sort_values("Timestamp").reset_index(drop=True)
df["tx_seq"] = np.arange(len(df))
df["is_laundering"] = df[IBM_LABEL].astype(int)

# ---------------------------------------------------------------------------
# 2. Transaction-level time & amount features
# ---------------------------------------------------------------------------
print(">>> Building time features ...")
df["hour"] = df["Timestamp"].dt.hour
df["day_of_week"] = df["Timestamp"].dt.dayofweek
df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
df["is_night"] = ((df["hour"] >= 23) | (df["hour"] <= 5)).astype(int)

print(">>> Building amount features ...")
df["amount_log"] = np.log1p(df["Amount Received"])
df["same_currency"] = (df["Receiving Currency"] == df["Payment Currency"]).astype(int)
df["is_round_amount"] = ((df["Amount Received"] % 1000) == 0).astype(int)

# ---------------------------------------------------------------------------
# 3. Account-level network features (fan-out / fan-in / volume)
# ---------------------------------------------------------------------------
print(">>> Building account-level network features ...")
S = df.groupby("Account").agg(
    S_tx_count=("tx_seq", "size"),
    S_total_out=("Amount Received", "sum"),
    S_avg_out=("Amount Received", "mean"),
    S_max_out=("Amount Received", "max"),
    S_std_out=("Amount Received", "std"),
    S_unique_recv=("Account.1", "nunique"),      # fan-out
    S_unique_bank_to=("To Bank", "nunique"),
    S_unique_currency=("Receiving Currency", "nunique"),
    S_active_days=("Timestamp", "nunique"),
).reset_index()

R = df.groupby("Account.1").agg(
    R_tx_count=("tx_seq", "size"),
    R_total_in=("Amount Received", "sum"),
    R_avg_in=("Amount Received", "mean"),
    R_max_in=("Amount Received", "max"),
    R_std_in=("Amount Received", "std"),
    R_unique_send=("Account", "nunique"),        # fan-in
    R_unique_bank_from=("From Bank", "nunique"),
    R_unique_currency=("Payment Currency", "nunique"),
    R_active_days=("Timestamp", "nunique"),
).reset_index()

df = df.merge(S, on="Account", how="left")
df = df.merge(R, on="Account.1", how="left")

# ---------------------------------------------------------------------------
# 4. Deviation interactions (amount vs account baselines)
# ---------------------------------------------------------------------------
print(">>> Building deviation features ...")
df["amt_vs_sender_avg"] = df["Amount Received"] / (df["S_avg_out"] + 1e-6)
df["amt_vs_receiver_avg"] = df["Amount Received"] / (df["R_avg_in"] + 1e-6)
df["S_avg_per_day"] = df["S_tx_count"] / (df["S_active_days"] + 1e-6)
df["R_avg_per_day"] = df["R_tx_count"] / (df["R_active_days"] + 1e-6)

# ---------------------------------------------------------------------------
# 5. Persist
# ---------------------------------------------------------------------------
print(">>> Saving feature matrix ...")
FEAT_COLS = (
    ["tx_seq", "Timestamp", "Account", "Account.1", "From Bank", "To Bank",
     "Payment Format", "Receiving Currency", "Payment Currency",
     "Amount Received", "is_laundering", IBM_LABEL,
     "hour", "day_of_week", "is_weekend", "is_night",
     "amount_log", "same_currency", "is_round_amount",
     "S_tx_count", "S_total_out", "S_avg_out", "S_max_out", "S_std_out",
     "S_unique_recv", "S_unique_bank_to", "S_unique_currency", "S_active_days",
     "R_tx_count", "R_total_in", "R_avg_in", "R_max_in", "R_std_in",
     "R_unique_send", "R_unique_bank_from", "R_unique_currency", "R_active_days",
     "amt_vs_sender_avg", "amt_vs_receiver_avg", "S_avg_per_day", "R_avg_per_day"]
)
feat = df[FEAT_COLS].copy()
feat.to_parquet(FEATURES_PARQUET, index=False)
print(f"    saved {len(feat):,} rows x {len(feat.columns)} cols -> {FEATURES_PARQUET}")

# Account profile summary (intermediate artifact reused by 04)
profiles = df.groupby("Account").agg(
    n_tx=("tx_seq", "size"),
    total_out=("Amount Received", "sum"),
    fan_out=("Account.1", "nunique"),
    fan_in=("Account", "count"),   # placeholder, refined in 04
    n_days=("Timestamp", "nunique"),
    n_currency=("Receiving Currency", "nunique"),
    n_bank_to=("To Bank", "nunique"),
).reset_index()
profiles.to_parquet(PROFILES_PARQUET, index=False)
print(f"    account profiles -> {PROFILES_PARQUET} ({len(profiles):,} accounts)")
print(">>> Feature engineering done.")
