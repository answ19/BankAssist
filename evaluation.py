"""Compute BankAssist evaluation metrics from logged requests and reviews.

Metrics are computed only from recorded data. Any measure without data is
reported as "Pending" rather than estimated. Uses only the Python standard library.
"""

import argparse
import statistics
import sys
import time
from collections import Counter

import database


def safe_divide(numerator, denominator):
    return numerator / denominator if denominator else None


def classification_metrics(y_true, y_pred, labels=None):
    """Accuracy, macro/weighted F1, per-class scores, and a confusion matrix.

    labels defaults to every label seen in y_true or y_pred, sorted. Pass the
    label-mapping keys to include intents with no examples. Precision, recall,
    or F1 with a zero denominator counts as 0 (the scikit-learn default).
    """
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths.")
    if not y_true:
        return None
    labels = sorted(set(y_true) | set(y_pred)) if labels is None else list(labels)
    index = {label: position for position, label in enumerate(labels)}
    unknown = sorted((set(y_true) | set(y_pred)) - set(index))
    if unknown:
        raise ValueError("Labels missing from the label list: {}".format(", ".join(unknown)))
    matrix = [[0] * len(labels) for _ in labels]
    for true, pred in zip(y_true, y_pred):
        matrix[index[true]][index[pred]] += 1

    per_class = {}
    for label, position in index.items():
        true_positive = matrix[position][position]
        support = sum(matrix[position])
        predicted = sum(row[position] for row in matrix)
        precision = safe_divide(true_positive, predicted) or 0.0
        recall = safe_divide(true_positive, support) or 0.0
        f1 = safe_divide(2 * precision * recall, precision + recall) or 0.0
        per_class[label] = {"precision": precision, "recall": recall,
                            "f1": f1, "support": support}
    total = len(y_true)
    return {
        "count": total,
        "accuracy": sum(t == p for t, p in zip(y_true, y_pred)) / total,
        "macro_f1": sum(c["f1"] for c in per_class.values()) / len(labels),
        "weighted_f1": sum(c["f1"] * c["support"] for c in per_class.values()) / total,
        "per_class": per_class,
        "labels": labels,
        "confusion_matrix": matrix,
    }


def recall_at_k(relevant_ids, ranked_ids, k):
    """Share of queries whose relevant FAQ appears in the top k retrieved FAQs."""
    if len(relevant_ids) != len(ranked_ids):
        raise ValueError("relevant_ids and ranked_ids have different lengths.")
    if not relevant_ids:
        return None
    return sum(rel in ranked[:k] for rel, ranked in zip(relevant_ids, ranked_ids)) / len(relevant_ids)


def threshold_sweep(confidences, correct, thresholds):
    """For each threshold: share answered automatically and accuracy on each side.

    correct[i] says whether prediction i matched the true intent. This shows the
    trade-off between automatic coverage and wrong automatic answers.
    """
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct have different lengths.")
    rows = []
    for threshold in thresholds:
        answered = [ok for score, ok in zip(confidences, correct) if score >= threshold]
        escalated = [ok for score, ok in zip(confidences, correct) if score < threshold]
        rows.append({
            "threshold": threshold,
            "automatic_rate": safe_divide(len(answered), len(correct)),
            "automatic_accuracy": safe_divide(sum(answered), len(answered)),
            "wrong_automatic_answers": len(answered) - sum(answered),
            "escalated": len(escalated),
            "escalated_accuracy": safe_divide(sum(escalated), len(escalated)),
        })
    return rows


def calibration_bins(confidences, correct, bins=10):
    """Reliability table and expected calibration error (ECE) over equal-width bins."""
    if len(confidences) != len(correct):
        raise ValueError("confidences and correct have different lengths.")
    if not confidences:
        return None
    groups = [[] for _ in range(bins)]
    for score, ok in zip(confidences, correct):
        groups[min(int(score * bins), bins - 1)].append((score, ok))
    rows, ece = [], 0.0
    for number, group in enumerate(groups):
        if not group:
            continue
        mean_confidence = sum(score for score, _ in group) / len(group)
        accuracy = sum(ok for _, ok in group) / len(group)
        ece += len(group) / len(confidences) * abs(accuracy - mean_confidence)
        rows.append({"low": number / bins, "high": (number + 1) / bins, "count": len(group),
                     "mean_confidence": mean_confidence, "accuracy": accuracy})
    return {"bins": rows, "ece": ece}


def top_confusions(metrics, limit=10):
    """Most frequent (true intent, predicted intent) errors from classification_metrics."""
    labels, matrix = metrics["labels"], metrics["confusion_matrix"]
    pairs = [(matrix[i][j], labels[i], labels[j])
             for i in range(len(labels)) for j in range(len(labels))
             if i != j and matrix[i][j]]
    return sorted(pairs, key=lambda pair: (-pair[0], pair[1], pair[2]))[:limit]


def review_metrics(records):
    """Escalation, automatic-answer, and reviewer-decision rates.

    Escalated means the request was not answered automatically, whether it is
    still pending or already reviewed. Reviewer agreement counts reviews where the
    reviewer accepted the predicted intent: approved, or edited/rejected with no
    different corrected intent. A rejection without a corrected intent gives no
    intent judgement and is excluded from agreement.
    """
    total = len(records)
    escalated = sum(r["status"] != "answered" for r in records)
    decisions = Counter(r["decision"] for r in records if r["decision"])
    reviewed = sum(decisions.values())
    judged = [r for r in records if r["decision"]
              and not (r["decision"] == "rejected" and not r["corrected_intent"])]
    agreed = sum(r["corrected_intent"] in (None, r["predicted_intent"]) for r in judged)
    return {
        "requests": total,
        "automatic_answer_rate": safe_divide(total - escalated, total),
        "escalation_rate": safe_divide(escalated, total),
        "still_pending": sum(r["status"] == "pending_review" for r in records),
        "reviews": reviewed,
        "approval_rate": safe_divide(decisions["approved"], reviewed),
        "edit_rate": safe_divide(decisions["edited"], reviewed),
        "rejection_rate": safe_divide(decisions["rejected"], reviewed),
        "reviewer_agreement": safe_divide(agreed, len(judged)),
        "agreement_reviews": len(judged),
    }


def runtime_summary(milliseconds):
    values = [value for value in milliseconds if value is not None]
    if not values:
        return None
    return {"count": len(values), "mean_ms": statistics.mean(values),
            "median_ms": statistics.median(values), "max_ms": max(values)}


def time_faq_lookups(intents, repeats=20, db_path=database.DB_PATH):
    """Measure get_faq_by_intent latency in milliseconds (read-only)."""
    timings = []
    for _ in range(repeats):
        for intent in intents:
            start = time.perf_counter()
            database.get_faq_by_intent(intent, db_path)
            timings.append((time.perf_counter() - start) * 1000)
    return runtime_summary(timings)


def evaluate_records(records, labels=None):
    labelled = [r for r in records if r["true_intent"] and r["predicted_intent"]]
    return {
        "classification": classification_metrics(
            [r["true_intent"] for r in labelled],
            [r["predicted_intent"] for r in labelled], labels),
        "review": review_metrics(records),
        "model_runtime": runtime_summary(r["model_runtime_ms"] for r in records),
        "ollama_runtime": runtime_summary(r.get("ollama_runtime_ms") for r in records),
    }


def fmt(value, kind="rate"):
    if value is None:
        return "Pending"
    if kind == "ms":
        return "{:.1f} ms".format(value)
    return "{:.2%}".format(value) if kind == "rate" else str(value)


def format_evaluation_table(results, faq_lookup=None):
    classification = results["classification"] or {}
    review = results["review"]
    runtime = results["model_runtime"] or {}
    ollama = results["ollama_runtime"] or {}
    lookup = faq_lookup or {}
    rows = [
        ("Accuracy", fmt(classification.get("accuracy")),
         "Correct predictions divided by labelled requests (n={})".format(
             classification.get("count", 0))),
        ("Macro-F1", fmt(classification.get("macro_f1")), "Average F1 across intents"),
        ("Weighted-F1", fmt(classification.get("weighted_f1")), "F1 weighted by class support"),
        ("FAQ retrieval accuracy", "Pending", "Relevant FAQ retrieved for the query (manual check)"),
        ("Automatic answer rate", fmt(review["automatic_answer_rate"]),
         "Answered requests divided by all requests (n={})".format(review["requests"])),
        ("Escalation rate", fmt(review["escalation_rate"]),
         "Pending-review or reviewed requests divided by all requests"),
        ("Approval rate", fmt(review["approval_rate"]),
         "Approved reviews divided by all reviews (n={})".format(review["reviews"])),
        ("Edit rate", fmt(review["edit_rate"]), "Edited reviews divided by all reviews"),
        ("Rejection rate", fmt(review["rejection_rate"]), "Rejected reviews divided by all reviews"),
        ("Reviewer agreement", fmt(review["reviewer_agreement"]),
         "Reviews keeping the predicted intent (n={})".format(review["agreement_reviews"])),
        ("Average classifier time", fmt(runtime.get("mean_ms"), "ms"),
         "Mean logged classifier prediction time per query (n={})".format(
             runtime.get("count", 0))),
        ("Average Ollama response time", fmt(ollama.get("mean_ms"), "ms"),
         "Mean logged local LLM response time per query (n={})".format(
             ollama.get("count", 0))),
        ("Average FAQ lookup time", fmt(lookup.get("mean_ms"), "ms"),
         "Mean SQLite FAQ lookup time (n={})".format(lookup.get("count", 0))),
    ]
    lines = ["| Measure | Result | How it is obtained |", "|---|---:|---|"]
    lines.extend("| {} | {} | {} |".format(*row) for row in rows)
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db", default=str(database.DB_PATH), help="SQLite database path")
    parser.add_argument("--include-test-cases", action="store_true",
                        help="include hand-entered routing test scores (not model results)")
    args = parser.parse_args(argv)
    records = database.get_evaluation_records(args.include_test_cases, args.db)
    intents = sorted({r["predicted_intent"] for r in records if r["predicted_intent"]})
    lookup = time_faq_lookups(intents, db_path=args.db) if intents else None
    if args.include_test_cases:
        print("NOTE: includes hand-entered test scores; these are not model results.\n")
    print(format_evaluation_table(evaluate_records(records), lookup))
    return 0


if __name__ == "__main__":
    sys.exit(main())
