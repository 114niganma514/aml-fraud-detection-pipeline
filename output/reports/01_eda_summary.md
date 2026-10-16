# 01 Exploratory Data Analysis -- IBM HI-Small

## 1.1 Overview
- rows: 5,078,345
- columns: 13
- time span: 2022-09-01 00:00:00 -> 2022-09-18 16:18:00 (18 days)
- missing values: 0
- unique accounts: 496,995 (sender) / 420,636 (receiver)
- unique banks: 30,470 / 15,811

## 1.2 Label imbalance
- laundering transactions: 5,177 (0.1019% ~ 1 in 981)
- baseline PR-AUC = 0.0010

## 1.3 Key finding -- payment channel
| Payment Format | tx count | laundering rate |
|---|---|---|
| ACH | 600,797.0 | 0.7462% |
| Bitcoin | 146,091.0 | 0.0383% |
| Cash | 490,891.0 | 0.0220% |
| Cheque | 1,864,331.0 | 0.0174% |
| Credit Card | 1,323,324.0 | 0.0156% |
| Reinvestment | 481,056.0 | 0.0000% |
| Wire | 171,855.0 | 0.0000% |

> ACH carries 11.8% of all tx but its laundering rate is
> 0.7462% -- far above the global mean (0.1019%).
> This is the single strongest rule signal.

## 1.4 Time behaviour
- peak hour(s) by volume: 0 (UTC)
- laundering rate is roughly stable across hours (0.028%-0.174%)

## 1.5 Amount behaviour
- round-amount (multiple of 1000) share: 0.00% of all tx
- amount distribution is heavy-tailed (log-normal like)

## 1.6 Network structure
- sender out-degree long tail: top sender sends 168,672 tx
- receiver in-degree long tail: top receiver receives 1,084 tx
> Fan-out / fan-in features are therefore highly discriminative.
