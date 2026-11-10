# DistilBERT deep-model plan (Phase 3)

Owner: Suraj. Status when written: **plan written before training. Observed results are in docs/distilbert_results.md.**
Everything below under "Planned" is a plan, not an observed result. Observed findings come only from
`notebooks/01_eda.ipynb` and, after it is run, `docs/distilbert_results.md`.

## Purpose in the workflow

customer query -> preprocessing -> **intent classification** -> confidence check -> approved FAQ response or human review.
DistilBERT is a candidate replacement for the TF-IDF + Logistic Regression classifier. It must return the same
fields (`query`, `predicted_intent`, `confidence_score`, `top_intents`, `model_name`, `model_runtime_ms`) so
`routing.route_query` and the evaluation scripts work unchanged. `distilbert_classifier.predict` does this.

## Planned architecture

- Pretrained `distilbert-base-uncased` (6 transformer layers, hidden size 768, about 66 million parameters), loaded with
  `AutoModelForSequenceClassification` (a linear classification head with 77 outputs on top of the encoder).
- Input: the shared `text_clean` column, tokenised with the model's word-piece tokenizer, truncated to 64 tokens.
  The EDA shows a median of 10 words and a 99th percentile of 43 words, so 64 should rarely truncate; the training script
  prints the real truncation rate.
- Output: softmax over 77 intents. `confidence_score` = largest softmax probability, the same definition used for the baseline,
  so the 0.90 threshold analysis is comparable. It measures intent confidence, not answer correctness.
- Labels: ids from `data/processed/label_mapping.json`, original capitalisation kept.

## Planned training strategy

| Setting | Value | Reason |
|---|---|---|
| Data | `data/processed/train.csv` (7,999) for training; `validation.csv` (2,000) for epoch selection and threshold analysis | Team rule: shared splits; test set reserved |
| Optimiser | AdamW, learning rate 5e-5, weight decay 0.01 (not on bias/LayerNorm) | Common fine-tuning defaults for this model size |
| Schedule | Linear decay, 10% warmup | Stabilises early steps |
| Batch size / epochs | 32 / up to 8 (raised from 4 after the pilot was still improving) | About 250 steps per epoch |
| Early stopping | Stop after 2 epochs with no validation macro-F1 gain; keep best epoch | Limits overfitting on a small dataset |
| Seed | 42 | Reproducibility; a few extra seeds if time allows, to show variance |
| Class weighting | None initially | Imbalance is mild (5.3 to 1); revisit only if per-class F1 shows a problem |

Choosing the best epoch on validation makes validation scores slightly optimistic. The official test set stays untouched
until the final evaluation, as the team agreed.

## Resource assessment (estimates, to be confirmed by the pilot)

- Memory: weights, gradients and AdamW state for about 66 million parameters come to roughly 1 GB in 32-bit floats, plus small
  activations for batches of short sequences. This should fit a free Colab GPU comfortably. These are estimates, not measurements.
- Time: with a GPU, the full training set is expected to take minutes per epoch; on a laptop CPU it is expected to be much slower
  and may not be practical. The pilot records real time and hardware (`run_summary.json`) so this can be replaced by a measurement.
- Inference: single-query latency is measured on 200 validation queries after one warm-up call and reported next to the baseline's
  roughly 5 ms (Kashish's measurement) so the team can weigh accuracy against speed.

## Planned pilot, then full run

1. Code check (no download): `python train_distilbert.py --smoke-test`. Completed on a tiny random model; confirms the script and
   output format work. It says nothing about DistilBERT accuracy.
2. Pilot: `python train_distilbert.py --pilot-per-class 10` (770 training examples). Records time, memory, truncation rate and a
   first accuracy. Reported as a partial run, never as the final result.
3. Full run on Colab GPU if the pilot shows it is practical. If not, report the pilot honestly and state what remains.

## Comparison and risks

| Risk | Evidence so far | Mitigation |
|---|---|---|
| Little or no gain over TF-IDF | Only 0.90% of validation tokens are unseen in training, so vocabulary gaps are small | Report the measured difference either way; consider accuracy versus latency |
| Overconfident softmax probabilities | Not yet measured | Report ECE and the threshold table; the team picks the threshold on validation evidence |
| No GPU or limited Colab time | Unknown until the pilot | Pilot first; keep `--epochs` and `--pilot-per-class` adjustable |
| Noisy per-class results | Smallest validation class has 7 queries | Report macro-F1 and note small supports |
| Two different baseline numbers in the repo | 84.30% in `results/baseline/validation_predictions.csv`; 88.25% in `docs/baseline_results.md` | Team names one reference baseline before comparing |
| Outlier rows resembling FAQ text | A few of the longest raw rows read like answer text | Kept (labels are consistent); mention in the report |

## How the results connect to the team

- Kashish's `evaluation.py` can score `results/distilbert/validation_predictions.csv` directly (same columns as Anuja's baseline file).
- Sakshi's routing needs no change: swap `classifier.predict` for `distilbert_classifier.predict`.
- Anuja and Suraj: same label mapping and splits are used by construction.
