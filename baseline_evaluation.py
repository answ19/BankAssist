"""Evaluate the saved baseline on the validation split and write a results report.

Reads data/processed/validation.csv (never the official test set). Each query is
classified one at a time through classifier.predict, so runtimes reflect single
customer queries. Writes docs/baseline_results.md and
results/baseline_validation_predictions.csv.

    python classifier.py            # once, to train
    python baseline_evaluation.py
"""

import csv
import platform
import sys
from datetime import date
from pathlib import Path

import sklearn

import classifier
import evaluation
import routing


ROOT = Path(__file__).resolve().parent
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
REPORT_PATH = ROOT / "docs" / "baseline_results.md"
PREDICTIONS_PATH = ROOT / "results" / "baseline_validation_predictions.csv"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]


def pct(value):
    return "n/a" if value is None else "{:.2%}".format(value)


def build_report(model, metrics, sweep, calibration, runtime, confusions, count):
    params = model.get_params()
    lowest = sorted(metrics["per_class"].items(), key=lambda item: (item[1]["f1"], item[0]))
    lines = [
        "# Baseline results: TF-IDF + Logistic Regression",
        "",
        "Date: {}. Split: `data/processed/validation.csv` ({} queries, 77 intents). "
        "The official BANKING77 test set was not used.".format(date.today(), count),
        "",
        "Model: `{}` trained on `data/processed/train.csv` only. Hyperparameters were "
        "chosen by 5-fold stratified cross-validation (macro-F1) inside the training split: "
        "ngram_range={}, C={}, sublinear_tf=True. "
        "Confidence score = largest predicted class probability.".format(
            classifier.MODEL_NAME, params["tfidf__ngram_range"], params["logreg__C"]),
        "",
        "Environment: Python {}, scikit-learn {}, {}.".format(
            platform.python_version(), sklearn.__version__, platform.processor() or platform.machine()),
        "",
        "## Classification",
        "",
        "| Measure | Result |",
        "|---|---:|",
        "| Accuracy | {} |".format(pct(metrics["accuracy"])),
        "| Macro-F1 | {} |".format(pct(metrics["macro_f1"])),
        "| Weighted-F1 | {} |".format(pct(metrics["weighted_f1"])),
        "",
        "Lowest per-class F1 (all 77 are in the predictions CSV and can be recomputed):",
        "",
        "| Intent | Precision | Recall | F1 | Support |",
        "|---|---:|---:|---:|---:|",
    ]
    for intent, scores in lowest[:10]:
        lines.append("| {} | {} | {} | {} | {} |".format(
            intent, pct(scores["precision"]), pct(scores["recall"]), pct(scores["f1"]),
            scores["support"]))
    lines.extend([
        "",
        "Most frequent confusions:",
        "",
        "| True intent | Predicted intent | Count |",
        "|---|---|---:|",
    ])
    lines.extend("| {} | {} | {} |".format(true, pred, n) for n, true, pred in confusions)
    lines.extend([
        "",
        "## Confidence threshold",
        "",
        "Automatic rate = share of queries at or above the threshold (answered without "
        "review, assuming an approved FAQ exists for the intent). Escalated queries go to "
        "human review.",
        "",
        "| Threshold | Automatic rate | Accuracy when automatic | Wrong automatic answers | "
        "Escalated | Accuracy of escalated |",
        "|---:|---:|---:|---:|---:|---:|",
    ])
    for row in sweep:
        lines.append("| {:.2f} | {} | {} | {} | {} | {} |".format(
            row["threshold"], pct(row["automatic_rate"]), pct(row["automatic_accuracy"]),
            row["wrong_automatic_answers"], row["escalated"], pct(row["escalated_accuracy"])))
    lines.extend([
        "",
        "## Calibration",
        "",
        "Expected calibration error (10 equal-width bins): {:.4f}. A well-calibrated model "
        "has accuracy close to mean confidence in each bin.".format(calibration["ece"]),
        "",
        "| Confidence bin | Queries | Mean confidence | Accuracy |",
        "|---|---:|---:|---:|",
    ])
    for row in calibration["bins"]:
        lines.append("| {:.1f}-{:.1f} | {} | {} | {} |".format(
            row["low"], row["high"], row["count"], pct(row["mean_confidence"]),
            pct(row["accuracy"])))
    lines.extend([
        "",
        "## Runtime",
        "",
        "Single-query classification time on this machine (n={}): mean {:.2f} ms, "
        "median {:.2f} ms, max {:.2f} ms.".format(
            runtime["count"], runtime["mean_ms"], runtime["median_ms"], runtime["max_ms"]),
        "",
        "## Reproduce",
        "",
        "```powershell",
        r".\.venv\Scripts\python.exe classifier.py",
        r".\.venv\Scripts\python.exe baseline_evaluation.py",
        "```",
        "",
    ])
    return "\n".join(lines)


def main():
    try:
        model = classifier.load_model()
        texts, labels = classifier.read_split(VALIDATION_PATH)
    except (OSError, ValueError, csv.Error) as error:
        print("Evaluation failed: {}".format(error), file=sys.stderr)
        return 1
    predictions = [classifier.predict(text, model) for text in texts]
    predicted = [p["predicted_intent"] for p in predictions]
    confidences = [p["confidence_score"] for p in predictions]
    correct = [p == t for p, t in zip(predicted, labels)]
    metrics = evaluation.classification_metrics(labels, predicted,
                                                list(classifier.read_label_mapping()))
    sweep = evaluation.threshold_sweep(confidences, correct, THRESHOLDS)
    calibration = evaluation.calibration_bins(confidences, correct)
    runtime = evaluation.runtime_summary(p["model_runtime_ms"] for p in predictions)

    PREDICTIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with PREDICTIONS_PATH.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["query", "true_intent", "predicted_intent", "confidence_score",
                         "correct", "second_intent", "second_score"])
        for p, true, ok in zip(predictions, labels, correct):
            writer.writerow([p["query"], true, p["predicted_intent"],
                             "{:.6f}".format(p["confidence_score"]), int(ok),
                             p["top_intents"][1]["intent"],
                             "{:.6f}".format(p["top_intents"][1]["probability"])])
    report = build_report(model, metrics, sweep, calibration, runtime,
                          evaluation.top_confusions(metrics), len(texts))
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    current = evaluation.threshold_sweep(confidences, correct, [routing.CONFIDENCE_THRESHOLD])[0]
    print("At the current threshold {:.2f}: {} answered automatically, {} accurate.".format(
        routing.CONFIDENCE_THRESHOLD, pct(current["automatic_rate"]),
        pct(current["automatic_accuracy"])))
    print("Saved: {}\nSaved: {}".format(REPORT_PATH, PREDICTIONS_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
