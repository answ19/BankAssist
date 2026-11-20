# Confidence and coverage: validation split

Generated 2026-11-19 by `python coverage_analysis.py`. Split: validation (Selected TF-IDF + LR n=2,000, DistilBERT n=2,000). Threshold 0.90. FAQ snapshot: 6 of 77 intents approved (faq_drafts.csv sha256 d8b19f23e7cf). Ollama wording checks: not run (eligibility replay only).

Approved intents: activate_my_card, card_arrival, compromised_card, declined_card_payment, lost_or_stolen_card, lost_or_stolen_phone.

Confidence-only = confidence at or above the threshold. FAQ-gated = also has an approved FAQ for the predicted intent, which is the application's rule for an automatic reply. Accuracy is intent accuracy within the eligible subset.

## At the configured threshold (0.90)

| Model | Accuracy (all) | ECE | Confidence-only eligible | Coverage | Accuracy | Errors | FAQ-gated eligible | Coverage | Correct | Errors |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Selected TF-IDF + LR | 88.25% | 0.1186 | 831 | 41.55% | 99.40% | 5 | 71 | 3.55% | 68 | 3 |
| DistilBERT | 90.95% | 0.0478 | 1,798 | 89.90% | 96.22% | 68 | 139 | 6.95% | 128 | 11 |

## Selected TF-IDF + LR: threshold sweep

| Threshold | Confidence-only eligible | Coverage | Accuracy | Errors | FAQ-gated eligible | Coverage | Correct | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1,656 | 82.80% | 94.69% | 88 | 124 | 6.20% | 117 | 7 |
| 0.70 | 1,383 | 69.15% | 97.54% | 34 | 106 | 5.30% | 101 | 5 |
| 0.80 | 1,187 | 59.35% | 98.74% | 15 | 92 | 4.60% | 88 | 4 |
| 0.85 | 1,048 | 52.40% | 99.05% | 10 | 85 | 4.25% | 81 | 4 |
| 0.90 | 831 | 41.55% | 99.40% | 5 | 71 | 3.55% | 68 | 3 |
| 0.95 | 547 | 27.35% | 100.00% | 0 | 43 | 2.15% | 43 | 0 |

Incorrect FAQ-gated predictions at 0.90 (3):

| Query | True intent | Predicted intent |
|---|---|---|
| My credit card transaction was declined for a loan payment. Can you tell me why? | top_up_failed | declined_card_payment |
| I tried to make a payment with my card and it was declined. Why is it pending? | pending_card_payment | declined_card_payment |
| Why did my payment get declined? | declined_transfer | declined_card_payment |

Reliability bins (10 equal-width bins):

| Bin | Queries | Mean confidence | Accuracy |
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

## DistilBERT: threshold sweep

| Threshold | Confidence-only eligible | Coverage | Accuracy | Errors | FAQ-gated eligible | Coverage | Correct | Errors |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | 1,959 | 97.95% | 92.24% | 152 | 156 | 7.80% | 138 | 18 |
| 0.70 | 1,896 | 94.80% | 94.25% | 109 | 147 | 7.35% | 135 | 12 |
| 0.80 | 1,863 | 93.15% | 95.12% | 91 | 146 | 7.30% | 134 | 12 |
| 0.85 | 1,839 | 91.95% | 95.60% | 81 | 142 | 7.10% | 131 | 11 |
| 0.90 | 1,798 | 89.90% | 96.22% | 68 | 139 | 6.95% | 128 | 11 |
| 0.95 | 1,735 | 86.75% | 97.52% | 43 | 134 | 6.70% | 127 | 7 |

Incorrect FAQ-gated predictions at 0.90 (11):

| Query | True intent | Predicted intent |
|---|---|---|
| My payment has not been accepted. | reverted_card_payment? | declined_card_payment |
| Can you tell me what i steps i should take since my card was stolen? | lost_or_stolen_phone | lost_or_stolen_card |
| My credit card transaction was declined for a loan payment. Can you tell me why? | top_up_failed | declined_card_payment |
| How do I re-order a damaged card? | card_not_working | lost_or_stolen_card |
| My cards were stolen | lost_or_stolen_phone | lost_or_stolen_card |
| I tried to charge my card but it was declined. A couple days ago I was able to charge it and it was accepted. Is something wrong? Will you double check for me please? | top_up_failed | declined_card_payment |
| I tried to pay through my account but the payment is not going through. What's going on? | pending_card_payment | declined_card_payment |
| I thought I left my card at a restaurant, but they claim not to have it and now I don't have any idea where it is. I'm so worried that someone might be using it fraudulently. Is there any way to look into this? | lost_or_stolen_card | compromised_card |
| Why did my payment get declined? | declined_transfer | declined_card_payment |
| im worried about fraud how do you protect my account | verify_my_identity | compromised_card |
| I'm having issues with my card. You guys keep declining it for some reason. I haven't had any issues up until this point. What is going on? I've tried a handful of times with the same, unfortunate, result! | declined_transfer | declined_card_payment |

Reliability bins (10 equal-width bins):

| Bin | Queries | Mean confidence | Accuracy |
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

## Figures

- `figures/coverage_validation.pdf`
- `figures/coverage_validation.png`
- `figures/reliability_validation.pdf`
- `figures/reliability_validation.png`

Prediction files:

- Selected TF-IDF + LR: `results/baseline_validation_predictions.csv`
- DistilBERT: `results/distilbert/validation_predictions.csv`
