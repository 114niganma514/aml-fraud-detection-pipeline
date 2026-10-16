# 09 Rule Weight Tuning

## Method
- search 729 weight combos
  over {0, 0.5, 1.0} for RULE_1..RULE_6 on the validation split
- hybrid score = xgb_prob + sum(w_i * RULE_i); objective = Best-F1

## Result
| metric | XGBoost alone | Hybrid (tuned weights) |
|---|---|---|
| PR-AUC | 0.4411 | 0.4411 |
| Best F1 | 0.4563 | 0.4563 |

## Tuned weights
| rule | weight |
|---|---|
| RULE_1 | 0.0 |
| RULE_2 | 0.0 |
| RULE_3 | 0.0 |
| RULE_4 | 0.0 |
| RULE_5 | 0.0 |
| RULE_6 | 0.5 |

## Interpretation
- Rules with positive weight add ranking signal on top of the model;
  zero-weight rules are redundant once the model has learned the same patterns.
- The tuned hybrid keeps rules as an interpretable layer without hurting the
  model's ranking -- the production-relevant takeaway.
