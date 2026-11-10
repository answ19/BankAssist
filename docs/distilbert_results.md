# DistilBERT results (FULL RUN)

Date: 2026-11-10. Training split: 7999 examples. Validation split: 2000 queries. The official test set was not used.

## Setup

| Item | Value |
|---|---|
| Model | distilbert-base-uncased (67,012,685 parameters) |
| Hardware | Tesla T4 |
| Epochs run / best epoch | 8 / 8 (max 8, patience 2) |
| Batch size, learning rate | 32, 5e-05 |
| Warmup ratio, weight decay | 0.1, 0.01 |
| Max sequence length | 64 (truncated: 0.31% of training queries; 99th pct length 54, longest 93) |
| Seed | 42 |
| Training time | 310 s |
| Peak GPU memory | 1437 MB |
| Software | Python 3.13.16, torch 2.14.1+cu130 |

Best epoch is chosen by validation macro-F1, so validation scores are slightly optimistic; the untouched test set gives the unbiased estimate.

## Training history

| Epoch | Train loss | Val accuracy | Val macro-F1 | Seconds |
|---:|---:|---:|---:|---:|
| 1 | 3.6103 | 65.45% | 58.61% | 35 |
| 2 | 1.1588 | 88.80% | 87.99% | 36 |
| 3 | 0.3960 | 89.95% | 90.14% | 39 |
| 4 | 0.1854 | 90.45% | 90.64% | 40 |
| 5 | 0.0922 | 90.55% | 90.79% | 39 |
| 6 | 0.0475 | 90.85% | 91.15% | 39 |
| 7 | 0.0249 | 90.65% | 90.89% | 39 |
| 8 | 0.0182 | 90.95% | 91.25% | 40 |

## Validation results (best epoch)

| Measure | Result |
|---|---:|
| Accuracy | 90.95% |
| Macro-F1 | 91.25% |
| Weighted-F1 | 91.02% |

Lowest per-class F1:

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| lost_or_stolen_card | 70.00% | 87.50% | 77.78% | 16 |
| pending_top_up | 77.42% | 80.00% | 78.69% | 30 |
| top_up_failed | 75.00% | 82.76% | 78.69% | 29 |
| transfer_not_received_by_recipient | 79.41% | 79.41% | 79.41% | 34 |
| supported_cards_and_currencies | 69.44% | 96.15% | 80.65% | 26 |
| balance_not_updated_after_bank_transfer | 84.38% | 79.41% | 81.82% | 34 |
| failed_transfer | 79.31% | 85.19% | 82.14% | 27 |
| declined_transfer | 87.50% | 77.78% | 82.35% | 27 |
| pending_transfer | 85.71% | 80.00% | 82.76% | 30 |
| fiat_currency_support | 86.96% | 80.00% | 83.33% | 25 |

Most frequent confusions:

| True intent | Predicted intent | Count |
|---|---|---:|
| direct_debit_payment_not_recognised | card_payment_not_recognised | 6 |
| fiat_currency_support | supported_cards_and_currencies | 5 |
| pending_top_up | top_up_failed | 5 |
| why_verify_identity | verify_my_identity | 4 |
| declined_transfer | failed_transfer | 3 |
| top_up_failed | declined_card_payment | 3 |
| transfer_not_received_by_recipient | transfer_timing | 3 |
| Refund_not_showing_up | request_refund | 2 |
| activate_my_card | card_not_working | 2 |
| balance_not_updated_after_bank_transfer | pending_transfer | 2 |

## Confidence threshold

| Threshold | Automatic rate | Accuracy when automatic | Wrong automatic answers | Escalated |
|---:|---:|---:|---:|---:|
| 0.30 | 99.60% | 91.21% | 175 | 8 |
| 0.40 | 99.05% | 91.62% | 166 | 19 |
| 0.50 | 97.95% | 92.24% | 152 | 41 |
| 0.60 | 96.20% | 93.30% | 129 | 76 |
| 0.70 | 94.80% | 94.25% | 109 | 104 |
| 0.80 | 93.15% | 95.12% | 91 | 137 |
| 0.85 | 91.95% | 95.60% | 81 | 161 |
| 0.90 | 89.90% | 96.22% | 68 | 202 |
| 0.95 | 86.75% | 97.52% | 43 | 265 |

Calibration (ECE, 10 bins): 0.0478.

| Confidence bin | Queries | Mean confidence | Accuracy |
|---|---:|---:|---:|
| 0.1-0.2 | 4 | 17.78% | 25.00% |
| 0.2-0.3 | 4 | 24.75% | 25.00% |
| 0.3-0.4 | 11 | 34.89% | 18.18% |
| 0.4-0.5 | 22 | 45.62% | 36.36% |
| 0.5-0.6 | 35 | 55.46% | 34.29% |
| 0.6-0.7 | 28 | 65.73% | 28.57% |
| 0.7-0.8 | 33 | 75.93% | 45.45% |
| 0.8-0.9 | 65 | 85.83% | 64.62% |
| 0.9-1.0 | 1798 | 98.98% | 96.22% |

## Runtime

Single-query classifier time on this hardware (n=200, after one warm-up call): mean 5.18 ms, median 5.11 ms, max 9.27 ms.

## Reproduce

```
python train_distilbert.py --epochs 8
```
