# EDA summary (from `notebooks/01_eda.ipynb`)

Observed on `data/raw/train.csv` (10,003 rows, 77 intents) and the shared splits (7,999 train / 2,000 validation). Figures: `results/eda/`.

| Finding | Value |
|---|---|
| Missing or empty texts / labels | 0 / 0 |
| Conflicting labels for the same normalised question | 0 |
| Whitespace-only duplicate rows (4 pairs) | 8 rows; 4 removed in preprocessing, leaving 9,999 |
| Class size (raw) | min 35 (`contactless_not_working`), max 187 (`card_payment_fee_charged`), mean about 130, ratio 5.3 to 1 |
| Largest train/validation gap in an intent's share | 0.037 percentage points |
| Smallest class in train / validation | 28 / 7 queries |
| Query length | median 10 words, 95th percentile 29, 99th percentile 43, longest 79 |
| Queries with extra internal whitespace / edge whitespace | 456 / 9 |
| Non-ASCII queries (mostly £ and €) | 52 |
| Start lowercase / no final punctuation | 525 / 1,319 |
| Training vocabulary / validation tokens unseen in training | 2,186 word types / 0.90% (200 word types) |

Implications: cleaning stays minimal (whitespace only; keep case, punctuation and currency symbols); a 64-token limit suits DistilBERT;
the split is representative, but per-class validation scores for small intents are noisy; some intent pairs share vocabulary, which
supports confidence-based escalation. Two baseline accuracies exist in the repository (84.30% and 88.25%); the team should name one.
