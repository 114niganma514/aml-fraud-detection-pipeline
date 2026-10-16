# -*- coding: utf-8 -*-
"""
utils.py -- Shared utilities for the AML & Fraud Detection Pipeline.

Central place for:
  * canonical path constants (project layout: config/ data/ output/ scripts/)
  * IBM HI-Small raw data loading (label column contains a SPACE: "Is Laundering")
  * PR-based evaluation helpers for highly imbalanced data
  * markdown report writers

Environment note (Windows / PowerShell):
  * Always run scripts with `python scripts\\NN_name.py`
  * `&&` / `||` are NOT valid statement separators in PowerShell; use `;`
"""
import os
import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve, average_precision_score

# ---------------------------------------------------------------------------
# Canonical paths (D:\\Users\\jiayi\\aml-fraud-detection-pipeline)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_DIR = os.path.join(BASE_DIR, "config")
DATA_RAW = os.path.join(BASE_DIR, "data", "raw")
DATA_PROCESSED = os.path.join(BASE_DIR, "data", "processed")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
MODELS_DIR = os.path.join(OUTPUT_DIR, "models")

# --- raw data files ---------------------------------------------------------
IBM_RAW_CSV = os.path.join(DATA_RAW, "ibm", "HI-Small_Trans.csv")
IEEE_TRAIN_TX = os.path.join(DATA_RAW, "ieee", "train_transaction.csv")
IEEE_TRAIN_ID = os.path.join(DATA_RAW, "ieee", "train_identity.csv")
ELLIPTIC_DIR = os.path.join(DATA_RAW, "elliptic")

# --- processed artifacts ----------------------------------------------------
FEATURES_PARQUET = os.path.join(DATA_PROCESSED, "transactions_features.parquet")
RULE_FLAGS_PARQUET = os.path.join(DATA_PROCESSED, "rule_flags.parquet")
PROFILES_PARQUET = os.path.join(DATA_PROCESSED, "account_profiles.parquet")
TEST_PREDICTIONS_PARQUET = os.path.join(DATA_PROCESSED, "test_predictions.parquet")
RULES_CONFIG_JSON = os.path.join(CONFIG_DIR, "rules_config.json")

# --- IBM column names -------------------------------------------------------
# IMPORTANT GOTCHA: the label column is literally named "Is Laundering"
# (with a space). Using "Is_Laundering" raises KeyError.
IBM_LABEL = "Is Laundering"
IBM_ID_COLS = ["tx_seq"]

# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------
def ensure_dirs():
    """Create every output / config directory if it does not exist yet."""
    for d in (CONFIG_DIR, DATA_PROCESSED, FIGURES_DIR, REPORTS_DIR, MODELS_DIR):
        os.makedirs(d, exist_ok=True)


def best_f1_point(y_true, score):
    """
    Pick the operating threshold that maximises F1 on the PR curve.

    Why not threshold = 0.5?  With extreme class imbalance the raw model
    probabilities are not calibrated around 0.5; choosing the best F1 point
    on the validation curve is the standard "threshold tuning" practice.

    Returns (precision, recall, f1, threshold).
    """
    p, r, th = precision_recall_curve(y_true, score)
    f1 = 2 * p * r / (p + r + 1e-9)
    i = int(f1.argmax())
    thr = th[i] if i < len(th) else 0.0
    return p[i], r[i], f1[i], thr


def write_markdown(path, text):
    """Write a utf-8 markdown report and print the destination."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)
    print(f"[saved] {path}")


def head(df, n=5):
    """Print first n rows for quick inspection."""
    with pd.option_context("display.max_columns", None):
        print(df.head(n).to_string())
