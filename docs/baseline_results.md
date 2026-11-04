# Baseline results: TF-IDF + Logistic Regression

Date: 2026-11-03. Split: `data/processed/validation.csv` (2000 queries, 77 intents). The official BANKING77 test set was not used.

Model: `tfidf-logreg-baseline` trained on `data/processed/train.csv` only. Hyperparameters were chosen by 5-fold stratified cross-validation (macro-F1) inside the training split: ngram_range=(1, 1), C=10.0, sublinear_tf=True. Confidence score = largest predicted class probability.

Environment: Python 3.14.2, scikit-learn 1.9.1, Intel64 Family 6 Model 189 Stepping 1, GenuineIntel.

## Classification

| Measure | Result |
|---|---:|
| Accuracy | 88.25% |
| Macro-F1 | 88.30% |
| Weighted-F1 | 88.34% |

Lowest per-class F1 (all 77 are in the predictions CSV and can be recomputed):

| Intent | Precision | Recall | F1 | Support |
|---|---:|---:|---:|---:|
| top_up_failed | 62.50% | 68.97% | 65.57% | 29 |
| balance_not_updated_after_bank_transfer | 75.00% | 70.59% | 72.73% | 34 |
| why_verify_identity | 84.21% | 66.67% | 74.42% | 24 |
| pending_top_up | 78.57% | 73.33% | 75.86% | 30 |
| card_not_working | 67.86% | 86.36% | 76.00% | 22 |
| lost_or_stolen_card | 72.22% | 81.25% | 76.47% | 16 |
| pending_transfer | 76.67% | 76.67% | 76.67% | 30 |
| card_acceptance | 81.82% | 75.00% | 78.26% | 12 |
| declined_cash_withdrawal | 70.45% | 88.57% | 78.48% | 35 |
| declined_card_payment | 78.12% | 80.65% | 79.37% | 31 |

Most frequent confusions:

| True intent | Predicted intent | Count |
|---|---|---:|
| direct_debit_payment_not_recognised | card_payment_not_recognised | 6 |
| why_verify_identity | verify_my_identity | 6 |
| pending_cash_withdrawal | declined_cash_withdrawal | 5 |
| pending_top_up | top_up_failed | 5 |
| balance_not_updated_after_bank_transfer | pending_transfer | 4 |
| balance_not_updated_after_bank_transfer | balance_not_updated_after_cheque_or_cash_deposit | 3 |
| card_payment_wrong_exchange_rate | wrong_exchange_rate_for_cash_withdrawal | 3 |
| fiat_currency_support | supported_cards_and_currencies | 3 |
| top_up_failed | pending_top_up | 3 |
| top_up_failed | top_up_reverted | 3 |

## Confidence threshold

Automatic rate = share of queries at or above the threshold (answered without review, assuming an approved FAQ exists for the intent). Escalated queries go to human review.

| Threshold | Automatic rate | Accuracy when automatic | Wrong automatic answers | Escalated | Accuracy of escalated |
|---:|---:|---:|---:|---:|---:|
| 0.30 | 93.00% | 92.31% | 143 | 140 | 34.29% |
| 0.40 | 88.55% | 93.56% | 114 | 229 | 47.16% |
| 0.50 | 82.80% | 94.69% | 88 | 344 | 57.27% |
| 0.60 | 76.25% | 96.26% | 57 | 475 | 62.53% |
| 0.70 | 69.15% | 97.54% | 34 | 617 | 67.42% |
| 0.80 | 59.35% | 98.74% | 15 | 813 | 72.94% |
| 0.85 | 52.40% | 99.05% | 10 | 952 | 76.37% |
| 0.90 | 41.55% | 99.40% | 5 | 1169 | 80.33% |
| 0.95 | 27.35% | 100.00% | 0 | 1453 | 83.83% |

## Calibration

Expected calibration error (10 equal-width bins): 0.1186. A well-calibrated model has accuracy close to mean confidence in each bin.

| Confidence bin | Queries | Mean confidence | Accuracy |
|---|---:|---:|---:|
| 0.0-0.1 | 11 | 8.15% | 18.18% |
| 0.1-0.2 | 56 | 15.72% | 28.57% |
| 0.2-0.3 | 73 | 24.95% | 41.10% |
| 0.3-0.4 | 89 | 35.74% | 67.42% |
| 0.4-0.5 | 115 | 45.32% | 77.39% |
| 0.5-0.6 | 131 | 55.21% | 76.34% |
| 0.6-0.7 | 142 | 65.52% | 83.80% |
| 0.7-0.8 | 196 | 75.35% | 90.31% |
| 0.8-0.9 | 356 | 85.72% | 97.19% |
| 0.9-1.0 | 831 | 95.99% | 99.40% |

## Runtime

Single-query classification time on this machine (n=2000): mean 2.44 ms, median 1.90 ms, max 200.16 ms.

## Reproduce

```powershell
.\.venv\Scripts\python.exe classifier.py
.\.venv\Scripts\python.exe baseline_evaluation.py
```
