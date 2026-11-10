# TF-IDF + Logistic Regression baseline

## Settings

- Training data: `data/processed/train.csv` (7999 rows)
- Validation data: `data/processed/validation.csv` (2000 rows)
- Input: `text_clean`; target: `category`
- Pipeline: `TfidfVectorizer(ngram_range=(1, 2))` → `LogisticRegression(max_iter=1000, random_state=42)`
- TF-IDF uses its default L2 normalization; features remain sparse through the pipeline.
- `sublinear_tf` is not specified, so scikit-learn's default (`False`) applies. Logistic Regression `C` is not specified, so its default (`1.0`) applies.
- Fitted only on the training split. Intent names are the original category strings; probability columns are matched through `classifier.classes_`.
- No hyperparameter search or cross-validation is implemented in this script. The held-out validation split is evaluated after fitting and is not used for selection.
- No test data was read.
- scikit-learn: `1.9.1`

## Validation results

| Metric | Score |
| --- | ---: |
| Accuracy | 0.8430 |
| Macro-F1 | 0.8387 |
| Weighted-F1 | 0.8420 |

## Artifacts

- Fitted pipeline: `results/baseline/tfidf_logistic_regression.joblib`
- Validation predictions: `results/baseline/validation_predictions.csv`

The predictions file includes record ID, cleaned query, true and predicted intents, highest class probability, top three intents with probabilities (JSON), and model name.

## Comparison with `classifier.py`

The comparison below is verified from the current `classifier.py` and `train_baseline.py` source files. These are implementation differences; they do not represent a new training run.

| Setting | `classifier.py` | `train_baseline.py` |
| --- | --- | --- |
| TF-IDF n-grams | Grid searched: `(1, 1)` or `(1, 2)` | Fixed at `(1, 2)` |
| TF-IDF lowercase | `True` (explicit) | `True` (scikit-learn default) |
| TF-IDF sublinear term frequency | `True` (explicit) | `False` (scikit-learn default) |
| TF-IDF normalization | L2 (scikit-learn default) | L2 (scikit-learn default) |
| Logistic Regression `C` | Grid searched: `1.0`, `10.0`, or `100.0` | `1.0` (scikit-learn default) |
| Logistic Regression `max_iter` | `5000` | `1000` |
| Logistic Regression `random_state` | Not set | `42` |
| Selection procedure | `GridSearchCV`, macro-F1 scoring, 5-fold `StratifiedKFold` with shuffle and seed `42` | No cross-validation or hyperparameter search |

**TF-IDF fitting in cross-validation:** `classifier.py` puts the vectorizer and Logistic Regression in one scikit-learn `Pipeline`, then passes that pipeline to `GridSearchCV`. Consequently, for each parameter candidate and CV fold, the pipeline is fitted on that fold's training portion; TF-IDF vocabulary and IDF weights are learned within the fold and do not use that fold's held-out portion. With `GridSearchCV`'s default `refit=True`, after selection the best pipeline is fitted again on the full training split. The validation split is not loaded or used by `classifier.py`. `train_baseline.py` instead fits its pipeline once on the full training split, without CV.

## Reported validation results and decision

| Model | Accuracy | Macro-F1 | Weighted-F1 | Provenance |
| --- | ---: | ---: | ---: | --- |
| Original TF-IDF + Logistic Regression baseline | 0.8430 | 0.8387 | 0.8420 | Reported in this baseline's existing results; values preserved. |
| `classifier.py` model | Not included here | Not included here | Not included here | This comparison verifies implementation settings only. No classifier metrics were independently reproduced for this update. |

The baseline validation metrics above are the original reported results and are preserved as recorded. No independently reproduced metrics are reported. This update documents source-level differences only; no models were retrained and the shared splits were not changed.

`top_intents` remains a JSON-encoded list of objects with `intent` and `probability` fields in the predictions CSV.
