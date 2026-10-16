# 07 Real-Data Validation -- IEEE-CIS Fraud Detection

## Dataset
- rows: 590,540 (real e-commerce card-not-present transactions)
- fraud rate: 3.4990%
- joined identity table: device / card / email behavioural fields

## Results (time-ordered test split, 118,108 rows)
| Model | PR-AUC | Best F1 | Precision | Recall |
|---|---|---|---|---|
| LightGBM | 0.5080 | 0.4981 | 0.6741 | 0.3949 |
| XGBoost | 0.4363 | 0.4439 | 0.5295 | 0.3821 |

## Interpretation
- PR-AUC baseline = fraud prevalence = 0.0344.
- The stack built on synthetic IBM data (LightGBM unweighted / XGBoost weighted,
  PR-AUC metric, time-ordered split) transfers directly to real fraud data,
  validating the "synthetic training + real validation" narrative.

## Caveats (honesty for interviews)
- IEEE-CIS contains engineered aggregate features (C1-C14, V*) that already
  encode a lot of signal; feature engineering effort was intentionally minimal.
- We validate the *stack*, not a production-grade real-time system.
