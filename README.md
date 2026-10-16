# AML & Fraud Detection Pipeline

End-to-end suspicious transaction detection for **anti-money laundering (AML)** and
**fraud**, combining **interpretable rules**, **tree-based ML models**, **customer
risk profiling**, **graph neural networks**, and **real-data cross validation**.

> One investment of effort covers two resume directions (anti-fraud + AML) with a
> "synthetic training + real validation" narrative.

---

## 1. Datasets

| Dataset | Type | Size | Role |
|---|---|---|---|
| [IBM Transactions for AML (HI-Small)](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | synthetic bank transactions | 5.08M tx / 515k accounts / 5,177 laundering (0.10%) | main modelling |
| [IEEE-CIS Fraud Detection](https://www.kaggle.com/competitions/ieee-fraud-detection) | **real** e-commerce fraud | 590k tx / 3.5% fraud | real-data validation (fraud) |
| [Elliptic Bitcoin](https://www.kaggle.com/datasets/ellipticco/elliptic-data-set) | **real** blockchain AML | 203k nodes / 234k edges | real-data validation + GNN |

**Licensing & compliance (important for interviews):**
- IBM HI-Small: CDLA-Sharing-1.0 (allows modification & use with attribution).
- IEEE-CIS / Elliptic: research/competition terms; used here strictly for
  personal learning and portfolio demonstration. Not for commercial deployment.

## 2. Architecture

```
data/raw/{ibm,ieee,elliptic}    raw downloads (00_download_data.py)
data/processed/                 engineered features, rule flags, profiles, predictions
config/rules_config.json        tuned rule weights (09)
output/figures/                 all charts
output/reports/                 all markdown reports
output/models/                  trained model files
scripts/                        numbered pipeline stages + utils
```

## 3. Pipeline stages

| # | Script | Deliverable | Key result (current run) |
|---|---|---|---|
| 01 | explore_eda | EDA report + figures | 0.10% laundering; ACH channel 86.6% of laundering |
| 02 | feature_engineering | 44-col feature matrix | time/amount/network/deviation features |
| 03 | rule_engine | rule flags (R1-R6) | interpretable rules, 16x baseline precision |
| 04 | customer_profiling | 515k account profiles | 0-100 risk score; launderers score higher |
| 05 | model_training | LightGBM + XGBoost | PR-AUC up to ~0.48 (baseline 0.001) |
| 06 | evaluation_report | PR curves + report | rule vs model vs hybrid comparison |
| 07 | ieee_cis_validation | real fraud validation | stack transfers to real data |
| 08 | elliptic_gnn | SGC-GCN vs XGBoost | graph structure adds AML signal |
| 09 | rule_weight_tuning | rules_config.json | tuned rule weights on validation |

## 4. Quick start

```powershell
cd D:\Users\jiayi\aml-fraud-detection-pipeline
pip install -r requirements.txt
python scripts\run_all.py        # or double-click run_all.bat
```

Each stage can also run alone: `python scripts\05_model_training.py`.

## 5. Key design decisions (interview talking points)

1. **PR-AUC not accuracy**: 0.1% prevalence makes accuracy meaningless; all
   evaluation uses precision-recall based metrics.
2. **LightGBM without scale_pos_weight** (empirically diagnosed): huge positive
   weighting collapses LightGBM ranking (PR-AUC 0.017 vs 0.217) on this data,
   while XGBoost tolerates it and reaches the best PR-AUC.
3. **Rules as interpretable layer, not linear blend**: naive blending
   (0.7*model+0.3*rule) hurts ranking (0.48 -> 0.34); rules are kept as a
   strong-signal override + compliance layer.
4. **Time-ordered splits**: 60/20/20 on tx_seq to respect temporal order.
5. **Synthetic + real validation**: same stack proven on synthetic IBM data is
   re-run on IEEE-CIS (real fraud) and Elliptic (real blockchain), plus a
   dependency-free SGC graph convolution implemented on scipy sparse matrices.

## 6. Known limitations (honest disclosure)

- IBM HI-Small is synthetic; real-world behaviour may differ.
- Elliptic public files lack the official time step, so a random labelled split
  is used instead of the official time-based protocol.
- IEEE-CIS validation reuses the same stack but with minimal feature
  engineering (the dataset ships engineered aggregate features).
