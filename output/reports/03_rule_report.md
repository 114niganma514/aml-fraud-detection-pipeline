# 03 Rule Engine Report (v2)

## Rules
| Rule | Description | Threshold |
|---|---|---|
| R1 | unusually large amount | > P99.9 or z>30 |
| R2 | extreme fan-out | S_unique_recv >= 500 |
| R3 | extreme fan-in | R_unique_send >= 12 |
| R4 | high sender volume | S_tx_count>=5000 & S_total_out>=1e9 |
| R5 | ACH channel | Payment Format == ACH |
| R6 | combo | R1 & R2 |

## Performance (full data, laundering=5,177)
| Rule | hits | TP | FP | Precision | Recall |
|---|---|---|---|---|---|
| RULE_1 | 5,071 | 8 | 5,063 | 0.0016 | 0.0015 |
| RULE_2 | 449,859 | 633 | 449,226 | 0.0014 | 0.1223 |
| RULE_3 | 73,614 | 948 | 72,666 | 0.0129 | 0.1831 |
| RULE_4 | 436,786 | 615 | 436,171 | 0.0014 | 0.1188 |
| RULE_5 | 600,797 | 4,483 | 596,314 | 0.0075 | 0.8659 |
| RULE_6 | 260 | 0 | 260 | 0.0000 | 0.0000 |
| ALERT(R3&R5) | 56,402 | 940 | 55,462 | 0.0167 | 0.1816 |

> Rule ALERT reaches 0.0167 precision = 16x the
> baseline rate (0.1019%). Rules are intentionally broad-recall;
> models (05) add ranking precision on top.
