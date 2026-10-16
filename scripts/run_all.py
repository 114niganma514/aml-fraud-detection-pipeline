# -*- coding: utf-8 -*-
"""
run_all.py -- End-to-end pipeline orchestrator.

Runs every stage in dependency order. Any failure stops the run.

Stages:
  00 download data (idempotent)
  01 EDA                  -> output/reports/01_eda_summary.md
  02 feature engineering  -> data/processed/transactions_features.parquet
  03 rule engine          -> data/processed/rule_flags.parquet
  04 customer profiling   -> output/reports/04_profiling_summary.md
  05 model training       -> output/models/*, data/processed/test_predictions.parquet
  06 evaluation report    -> output/reports/06_evaluation_report.md
  07 IEEE-CIS real-data validation   -> output/reports/07_ieee_cis_validation.md
  08 Elliptic real blockchain + GNN  -> output/reports/08_elliptic_gnn.md
  09 rule weight tuning   -> config/rules_config.json

Run:  python scripts\\run_all.py
"""
import os
import subprocess
import sys

SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.dirname(SCRIPTS_DIR)
PY = sys.executable

STAGES = [
    "00_download_data.py",
    "01_explore_eda.py",
    "02_feature_engineering.py",
    "03_rule_engine.py",
    "04_customer_profiling.py",
    "05_model_training.py",
    "06_evaluation_report.py",
    "07_ieee_cis_validation.py",
    "08_elliptic_gnn.py",
    "09_rule_weight_tuning.py",
]


def main():
    print(f"Project root: {BASE_DIR}")
    for script in STAGES:
        print("=" * 60)
        print(f">>> running: {script}")
        print("=" * 60)
        rc = subprocess.call([PY, os.path.join(SCRIPTS_DIR, script)], cwd=BASE_DIR)
        if rc != 0:
            print(f"[FAILED] {script} (exit code {rc})")
            sys.exit(rc)
    print("\n" + "=" * 60)
    print(">>> ALL PIPELINE STAGES COMPLETED")
    print("    reports: output/reports/ | figures: output/figures/")
    print("=" * 60)


if __name__ == "__main__":
    main()
