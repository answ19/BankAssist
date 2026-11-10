# DistilBERT results (PILOT (10 examples per intent))

Date: 2026-11-10. Training split: 770 examples. Validation split: 2000 queries. The official test set was not used.

**This is a partial run. Its numbers must not be reported as the final DistilBERT result.**

## Setup

| Item | Value |
|---|---|
| Model | distilbert-base-uncased (67,012,685 parameters) |
| Hardware | Tesla T4 |
| Epochs run / best epoch | 3 / 3 (max 3, patience 2) |
| Batch size, learning rate | 32, 5e-05 |
| Warmup ratio, weight decay | 0.1, 0.01 |
| Max sequence length | 64 (truncated: 0.52% of training queries; 99th pct length 51, longest 87) |
| Seed | 42 |
| Training time | 18 s |
| Peak GPU memory | 1441 MB |
| Software | Python 3.13.16, torch 2.14.1+cu130 |

Best epoch is chosen by validation macro-F1, so validation scores are slightly optimistic; the untouched test set gives the unbiased estimate.

## Training history

| Epoch | Train loss | Val accuracy | Val macro-F1 | Seconds |
|---:|---:|---:|---:|---:|
| 1 | 4.3423 | 7.05% | 3.96% | 6 |
| 2 | 4.1990 | 19.60% | 15.33% | 5 |
| 3 | 4.0139 | 25.80% | 22.14% | 5 |

## Validation results (best epoch)

| Measure | Result |
|---|---:|
| Accuracy | 25.80% |
| Macro-F1 | 22.14% |
| Weighted-F1 | 21.60% |

Lowest per-class F1:

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| age_limit | 0.00% | 0.00% | 0.00% | 22 |
| card_about_to_expire | 0.00% | 0.00% | 0.00% | 26 |
| card_delivery_estimate | 0.00% | 0.00% | 0.00% | 22 |
| card_linking | 0.00% | 0.00% | 0.00% | 28 |
| card_not_working | 0.00% | 0.00% | 0.00% | 22 |
| card_payment_not_recognised | 0.00% | 0.00% | 0.00% | 34 |
| cash_withdrawal_charge | 0.00% | 0.00% | 0.00% | 35 |
| change_pin | 0.00% | 0.00% | 0.00% | 24 |
| compromised_card | 0.00% | 0.00% | 0.00% | 17 |
| contactless_not_working | 0.00% | 0.00% | 0.00% | 7 |

Most frequent confusions:

| True intent | Predicted intent | Count |
|---|---|---:|
| card_payment_wrong_exchange_rate | exchange_charge | 27 |
| fiat_currency_support | exchange_charge | 24 |
| getting_spare_card | order_physical_card | 24 |
| top_up_reverted | pending_top_up | 24 |
| card_linking | order_physical_card | 23 |
| direct_debit_payment_not_recognised | cash_withdrawal_not_recognised | 23 |
| wrong_exchange_rate_for_cash_withdrawal | exchange_charge | 23 |
| change_pin | pin_blocked | 20 |
| get_disposable_virtual_card | disposable_card_limits | 19 |
| declined_card_payment | order_physical_card | 18 |

## Confidence threshold

| Threshold | Automatic rate | Accuracy when automatic | Wrong automatic answers | Escalated |
|---:|---:|---:|---:|---:|
| 0.30 | 0.00% | n/a | 0 | 2000 |
| 0.40 | 0.00% | n/a | 0 | 2000 |
| 0.50 | 0.00% | n/a | 0 | 2000 |
| 0.60 | 0.00% | n/a | 0 | 2000 |
| 0.70 | 0.00% | n/a | 0 | 2000 |
| 0.80 | 0.00% | n/a | 0 | 2000 |
| 0.85 | 0.00% | n/a | 0 | 2000 |
| 0.90 | 0.00% | n/a | 0 | 2000 |
| 0.95 | 0.00% | n/a | 0 | 2000 |

Calibration (ECE, 10 bins): 0.2362.

| Confidence bin | Queries | Mean confidence | Accuracy |
|---|---:|---:|---:|
| 0.0-0.1 | 2000 | 2.18% | 25.80% |

## Runtime

Single-query classifier time on this hardware (n=200, after one warm-up call): mean 5.44 ms, median 5.37 ms, max 9.44 ms.

## Reproduce

```
python train_distilbert.py --pilot-per-class 10
```
