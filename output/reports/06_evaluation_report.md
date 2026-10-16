# 06 Final Evaluation Report

- test rows: 1,015,669
- test laundering rate: 0.1769%  (baseline PR-AUC = 0.0018)

## 1. Scheme comparison (test set)
| Scheme | PR-AUC | Best F1 | Precision@BestF1 | Recall@BestF1 |
|---|---|---|---|---|
| Rule-only | 0.0066 | 0.0147 | 0.0074 | 0.9917 |
| LightGBM | 0.4206 | 0.5335 | 0.8047 | 0.3990 |
| XGBoost | 0.5226 | 0.5184 | 0.6812 | 0.4185 |
| Hybrid(XGB+Rule) | 0.5226 | 0.5184 | 0.6812 | 0.4185 |

## 2. Best model business metrics (XGBoost @ best-F1 threshold 0.987)
- TP (correctly flagged launderers): 752
- FP (false positives): 352
- FN (missed launderers): 1,045
- TN: 1,013,520
- Precision 0.6812 | Recall 0.4185 | F1 0.5184

## 3. Key insights
1. Extreme imbalance (0.1%) makes accuracy meaningless; PR-AUC / Recall are the metrics.
2. XGBoost with scale_pos_weight is robust and reaches PR-AUC 0.5226.
3. LightGBM must NOT use scale_pos_weight on this data (ranking collapses).
4. Linear rule blending hurts the model; rules serve as an explainability/override layer.
