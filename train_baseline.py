"""Train and evaluate the shared TF-IDF + Logistic Regression baseline."""

import csv
import json
import sys
from pathlib import Path

import joblib
import pandas as pd
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
MAPPING_PATH = ROOT / "data" / "processed" / "label_mapping.json"
OUTPUT_DIR = ROOT / "results" / "baseline"
MODEL_PATH = OUTPUT_DIR / "tfidf_logistic_regression.joblib"
PREDICTIONS_PATH = OUTPUT_DIR / "validation_predictions.csv"
SUMMARY_PATH = OUTPUT_DIR / "baseline_summary.md"
MODEL_NAME = "tfidf_logistic_regression"
RANDOM_STATE = 42


def main():
    train = pd.read_csv(TRAIN_PATH)
    validation = pd.read_csv(VALIDATION_PATH)
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))

    required = {"record_id", "text_clean", "category"}
    for name, frame in (("training", train), ("validation", validation)):
        missing = required - set(frame.columns)
        if missing:
            raise ValueError("{} CSV is missing columns: {}".format(name, sorted(missing)))
    if set(train["category"]) != set(mapping):
        raise ValueError("Training categories do not match the shared label mapping.")
    if set(validation["category"]) - set(mapping):
        raise ValueError("Validation contains categories missing from the shared mapping.")
    if set(train["record_id"]) & set(validation["record_id"]):
        raise ValueError("Training and validation record IDs overlap.")

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2))),
        ("classifier", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])
    pipeline.fit(train["text_clean"], train["category"])

    classifier = pipeline.named_steps["classifier"]
    labels = classifier.classes_
    probabilities = pipeline.predict_proba(validation["text_clean"])
    predicted_indices = probabilities.argmax(axis=1)
    predicted = labels[predicted_indices]
    true = validation["category"].to_numpy()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "record_id", "query", "true_intent", "predicted_intent",
            "confidence_score", "top_intents", "model_name",
        ], lineterminator="\n")
        writer.writeheader()
        for row_index, (_, row) in enumerate(validation.iterrows()):
            order = probabilities[row_index].argsort()[::-1][:3]
            top_intents = [
                {"intent": str(labels[class_index]),
                 "probability": float(probabilities[row_index, class_index])}
                for class_index in order
            ]
            writer.writerow({
                "record_id": int(row["record_id"]),
                "query": row["text_clean"],
                "true_intent": row["category"],
                "predicted_intent": str(predicted[row_index]),
                "confidence_score": float(probabilities[row_index].max()),
                "top_intents": json.dumps(top_intents, ensure_ascii=False),
                "model_name": MODEL_NAME,
            })

    accuracy = accuracy_score(true, predicted)
    macro_f1 = f1_score(true, predicted, labels=labels, average="macro", zero_division=0)
    weighted_f1 = f1_score(true, predicted, labels=labels, average="weighted", zero_division=0)
    summary = """# TF-IDF + Logistic Regression baseline

## Settings

- Training data: `data/processed/train.csv` ({} rows)
- Validation data: `data/processed/validation.csv` ({} rows)
- Input: `text_clean`; target: `category`
- Pipeline: `TfidfVectorizer(ngram_range=(1, 2))` → `LogisticRegression(max_iter=1000, random_state=42)`
- TF-IDF uses its default L2 normalization; features remain sparse through the pipeline.
- Fitted only on the training split. Intent names are the original category strings; probability columns are matched through `classifier.classes_`.
- No test data was read.
- scikit-learn: `{}`

## Validation results

| Metric | Score |
| --- | ---: |
| Accuracy | {:.4f} |
| Macro-F1 | {:.4f} |
| Weighted-F1 | {:.4f} |

## Artifacts

- Fitted pipeline: `results/baseline/tfidf_logistic_regression.joblib`
- Validation predictions: `results/baseline/validation_predictions.csv`

The predictions file includes record ID, cleaned query, true and predicted intents, highest class probability, top three intents with probabilities (JSON), and model name.
""".format(len(train), len(validation), sklearn.__version__, accuracy, macro_f1, weighted_f1)
    SUMMARY_PATH.write_text(summary, encoding="utf-8")
    print("Training rows: {}; validation rows: {}; classes: {}".format(
        len(train), len(validation), len(labels)))
    print("Accuracy: {:.4f}; macro-F1: {:.4f}; weighted-F1: {:.4f}".format(
        accuracy, macro_f1, weighted_f1))
    print("Saved model: {}".format(MODEL_PATH))
    print("Saved predictions: {}".format(PREDICTIONS_PATH))
    print("Saved summary: {}".format(SUMMARY_PATH))
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError, KeyError) as error:
        print("Baseline failed: {}".format(error), file=sys.stderr)
        sys.exit(1)
