"""Fine-tune DistilBERT on the shared training split and evaluate on the validation split.

Uses data/processed/train.csv for training and data/processed/validation.csv for epoch
selection, early stopping and threshold analysis. The official test set is never read.
Writes validation predictions in the shared format plus a results report.

    python train_distilbert.py --pilot-per-class 10    # pilot: 10 training examples per intent
    python train_distilbert.py                         # full run
    python train_distilbert.py --smoke-test            # tiny random model, offline code check only

Pilot and smoke runs are labelled as such in their reports and write to results/distilbert_pilot/
so they cannot be mistaken for the full run.
"""

import argparse
import copy
import csv
import json
import platform
import random
import sys
import time
from datetime import date
from pathlib import Path

import numpy as np

import distilbert_classifier as dc
import evaluation

ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
MAPPING_PATH = ROOT / "data" / "processed" / "label_mapping.json"
THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95]
LATENCY_QUERIES = 200


def read_split(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    return ([row["record_id"] for row in rows], [row["text_clean"] for row in rows],
            [row["category"] for row in rows])


class HashTokenizer:
    """Offline stand-in tokenizer for --smoke-test only (no vocabulary download)."""

    def __init__(self, vocab_size=2000):
        self.vocab_size = vocab_size

    def __call__(self, texts, truncation=False, max_length=None):
        ids = []
        for text in texts:
            seq = [101] + [3 + (sum(ord(c) for c in w) % (self.vocab_size - 5))
                           for w in text.lower().split()] + [102]
            if truncation and max_length:
                seq = seq[:max_length]
            ids.append(seq)
        return {"input_ids": ids}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--model-name", default=dc.BASE_MODEL)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--learning-rate", type=float, default=5e-5)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-ratio", type=float, default=0.1)
    p.add_argument("--max-length", type=int, default=64)
    p.add_argument("--patience", type=int, default=2, help="epochs without val macro-F1 gain before stopping")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--pilot-per-class", type=int, default=0,
                   help="train on only N examples per intent (0 = full training split)")
    p.add_argument("--smoke-test", action="store_true", help="tiny random model, no downloads")
    p.add_argument("--train-path", type=Path, default=TRAIN_PATH)
    p.add_argument("--validation-path", type=Path, default=VALIDATION_PATH)
    p.add_argument("--results-dir", type=Path, default=None)
    p.add_argument("--report-path", type=Path, default=None)
    p.add_argument("--model-dir", type=Path, default=dc.MODEL_DIR)
    return p.parse_args(argv)


def set_seed(seed):
    import torch
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_model_and_tokenizer(args, num_labels, id2label, label2id):
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DistilBertConfig, DistilBertForSequenceClassification)
    if args.smoke_test:
        config = DistilBertConfig(vocab_size=2000, dim=32, hidden_dim=64, n_layers=1, n_heads=2,
                                  max_position_embeddings=128, num_labels=num_labels,
                                  id2label=id2label, label2id=label2id)
        return DistilBertForSequenceClassification(config), HashTokenizer(2000)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    return model, tokenizer


def truncation_rate(tokenizer, texts, max_length):
    lengths = [len(ids) for ids in tokenizer(list(texts))["input_ids"]]
    return float(np.mean([n > max_length for n in lengths])), int(np.percentile(lengths, 99)), max(lengths)


def evaluate_bundle(bundle, texts, labels, label_names):
    probabilities = dc.probabilities_for(bundle, texts)
    predicted = [bundle["classes"][i] for i in probabilities.argmax(axis=1)]
    metrics = evaluation.classification_metrics(labels, predicted, label_names)
    return probabilities, predicted, metrics


def run(args):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_linear_schedule_with_warmup

    set_seed(args.seed)
    mapping = json.loads(MAPPING_PATH.read_text(encoding="utf-8"))
    classes = [name for name, _ in sorted(mapping.items(), key=lambda kv: kv[1])]
    label2id = {name: i for i, name in enumerate(classes)}
    id2label = {i: name for name, i in label2id.items()}

    _, train_texts, train_labels = read_split(args.train_path)
    val_ids, val_texts, val_labels = read_split(args.validation_path)
    missing = set(classes) - set(train_labels)
    if missing:
        raise ValueError("Training split lacks intents: {}".format(", ".join(sorted(missing))))
    if args.pilot_per_class:
        keep = dc.stratified_subset(train_labels, args.pilot_per_class, args.seed)
        train_texts = [train_texts[i] for i in keep]
        train_labels = [train_labels[i] for i in keep]
    if args.smoke_test:
        val_ids, val_texts, val_labels = val_ids[:300], val_texts[:300], val_labels[:300]
    partial = bool(args.pilot_per_class or args.smoke_test)
    mode = "SMOKE TEST (tiny random model; code check only)" if args.smoke_test else (
        "PILOT ({} examples per intent)".format(args.pilot_per_class) if args.pilot_per_class
        else "FULL RUN")

    results_dir = args.results_dir or ROOT / "results" / ("distilbert_pilot" if partial else "distilbert")
    report_path = args.report_path or (results_dir / "report.md" if partial
                                       else ROOT / "docs" / "distilbert_results.md")
    results_dir.mkdir(parents=True, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, tokenizer = build_model_and_tokenizer(args, len(classes), id2label, label2id)
    model.to(device)
    n_params = sum(p.numel() for p in model.parameters())
    trunc, p99_len, max_len = truncation_rate(tokenizer, train_texts, args.max_length)
    print("Mode: {} | device: {} | parameters: {:,} | train {} / val {}".format(
        mode, device, n_params, len(train_texts), len(val_texts)))
    print("Token lengths: {:.2%} of training queries exceed max_length={} (99th pct {}, max {})".format(
        trunc, args.max_length, p99_len, max_len))

    encoded = tokenizer(train_texts, truncation=True, max_length=args.max_length)["input_ids"]
    y = np.array([label2id[label] for label in train_labels])

    def collate(batch_indices):
        ids, mask = dc._pad_batch([encoded[i] for i in batch_indices])
        return (torch.tensor(ids), torch.tensor(mask), torch.tensor(y[batch_indices]))

    generator = torch.Generator().manual_seed(args.seed)
    loader = DataLoader(list(range(len(encoded))), batch_size=args.batch_size, shuffle=True,
                        collate_fn=lambda b: collate(np.array(b)), generator=generator)
    steps = len(loader) * args.epochs
    no_decay = ("bias", "LayerNorm.weight")
    groups = [
        {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)],
         "weight_decay": args.weight_decay},
        {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)],
         "weight_decay": 0.0}]
    optimizer = torch.optim.AdamW(groups, lr=args.learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, int(args.warmup_ratio * steps), steps)

    bundle = {"model": model, "tokenizer": tokenizer, "classes": classes,
              "max_length": args.max_length, "device": device, "model_name": dc.MODEL_NAME}
    history, best_f1, best_state, stale = [], -1.0, None, 0
    run_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_start, total_loss = time.perf_counter(), 0.0
        for ids, mask, labels in loader:
            optimizer.zero_grad()
            out = model(input_ids=ids.to(device), attention_mask=mask.to(device), labels=labels.to(device))
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            total_loss += out.loss.item()
        _, _, metrics = evaluate_bundle(bundle, val_texts, val_labels, classes)
        row = {"epoch": epoch, "train_loss": total_loss / len(loader),
               "val_accuracy": metrics["accuracy"], "val_macro_f1": metrics["macro_f1"],
               "seconds": time.perf_counter() - epoch_start}
        history.append(row)
        print("Epoch {epoch}: loss {train_loss:.4f}  val acc {val_accuracy:.4f}  "
              "val macro-F1 {val_macro_f1:.4f}  ({seconds:.0f}s)".format(**row))
        if metrics["macro_f1"] > best_f1:
            best_f1, stale = metrics["macro_f1"], 0
            best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
            best_epoch = epoch
        else:
            stale += 1
            if stale >= args.patience:
                print("Early stopping: no validation macro-F1 gain for {} epochs.".format(args.patience))
                break
    train_seconds = time.perf_counter() - run_start

    model.load_state_dict(best_state)
    probabilities, predicted, metrics = evaluate_bundle(bundle, val_texts, val_labels, classes)
    confidences = probabilities.max(axis=1).tolist()
    correct = [p == t for p, t in zip(predicted, val_labels)]
    sweep = evaluation.threshold_sweep(confidences, correct, THRESHOLDS)
    calibration = evaluation.calibration_bins(confidences, correct)

    latency_texts = val_texts[:LATENCY_QUERIES]
    dc.predict(latency_texts[0], bundle)  # warm-up, not counted
    single = [dc.predict(t, bundle)["model_runtime_ms"] for t in latency_texts]
    runtime = evaluation.runtime_summary(single)

    pred_path = results_dir / "validation_predictions.csv"
    with pred_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["record_id", "query", "true_intent", "predicted_intent",
                         "confidence_score", "top_intents", "model_name"])
        for i, text in enumerate(val_texts):
            order = np.argsort(-probabilities[i], kind="stable")[:3]
            top = [{"intent": classes[j], "probability": float(probabilities[i][j])} for j in order]
            writer.writerow([val_ids[i], text, val_labels[i], predicted[i],
                             float(confidences[i]), json.dumps(top), dc.MODEL_NAME])

    summary = {
        "mode": mode, "date": str(date.today()), "best_epoch": best_epoch,
        "settings": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                     if k not in ("train_path", "validation_path", "results_dir", "report_path", "model_dir")},
        "train_examples": len(train_texts), "validation_examples": len(val_texts),
        "parameters": n_params, "device": device,
        "gpu": torch.cuda.get_device_name(0) if device == "cuda" else None,
        "peak_gpu_memory_mb": (torch.cuda.max_memory_allocated() / 2**20) if device == "cuda" else None,
        "python": platform.python_version(), "torch": torch.__version__,
        "train_seconds": train_seconds, "history": history,
        "truncation_rate_train": trunc, "p99_token_length": p99_len, "max_token_length": max_len,
        "accuracy": metrics["accuracy"], "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"], "ece": calibration["ece"],
        "single_query_runtime_ms": runtime,
    }
    (results_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(build_report(summary, metrics, sweep, calibration), encoding="utf-8")

    if not args.smoke_test:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(args.model_dir)
        tokenizer.save_pretrained(args.model_dir)
        (args.model_dir / "bankassist_meta.json").write_text(
            json.dumps({"classes": classes, "max_length": args.max_length}), encoding="utf-8")
    print("Validation: accuracy {:.4f}, macro-F1 {:.4f}, ECE {:.4f}".format(
        metrics["accuracy"], metrics["macro_f1"], calibration["ece"]))
    print("Saved: {}\nSaved: {}\nSaved: {}".format(pred_path, results_dir / "run_summary.json", report_path))
    return summary


def pct(value):
    return "n/a" if value is None else "{:.2%}".format(value)


def build_report(s, metrics, sweep, calibration):
    st, rt = s["settings"], s["single_query_runtime_ms"]
    lines = [
        "# DistilBERT results ({})".format(s["mode"]), "",
        "Date: {}. Training split: {} examples. Validation split: {} queries. "
        "The official test set was not used.".format(s["date"], s["train_examples"], s["validation_examples"]), "",
    ]
    if "SMOKE" in s["mode"] or "PILOT" in s["mode"]:
        lines += ["**This is a partial run. Its numbers must not be reported as the final DistilBERT result.**", ""]
    lines += [
        "## Setup", "",
        "| Item | Value |", "|---|---|",
        "| Model | {} ({:,} parameters) |".format(
            "tiny random DistilBERT (smoke test)" if st["smoke_test"] else st["model_name"], s["parameters"]),
        "| Hardware | {} |".format(s["gpu"] or "CPU"),
        "| Epochs run / best epoch | {} / {} (max {}, patience {}) |".format(
            len(s["history"]), s["best_epoch"], st["epochs"], st["patience"]),
        "| Batch size, learning rate | {}, {} |".format(st["batch_size"], st["learning_rate"]),
        "| Warmup ratio, weight decay | {}, {} |".format(st["warmup_ratio"], st["weight_decay"]),
        "| Max sequence length | {} (truncated: {} of training queries; 99th pct length {}, longest {}) |".format(
            st["max_length"], pct(s["truncation_rate_train"]), s["p99_token_length"], s["max_token_length"]),
        "| Seed | {} |".format(st["seed"]),
        "| Training time | {:.0f} s |".format(s["train_seconds"]),
        "| Peak GPU memory | {} |".format("n/a" if s["peak_gpu_memory_mb"] is None else "{:.0f} MB".format(s["peak_gpu_memory_mb"])),
        "| Software | Python {}, torch {} |".format(s["python"], s["torch"]), "",
        "Best epoch is chosen by validation macro-F1, so validation scores are slightly optimistic; "
        "the untouched test set gives the unbiased estimate.", "",
        "## Training history", "",
        "| Epoch | Train loss | Val accuracy | Val macro-F1 | Seconds |", "|---:|---:|---:|---:|---:|"]
    for h in s["history"]:
        lines.append("| {} | {:.4f} | {} | {} | {:.0f} |".format(
            h["epoch"], h["train_loss"], pct(h["val_accuracy"]), pct(h["val_macro_f1"]), h["seconds"]))
    lowest = sorted(metrics["per_class"].items(), key=lambda kv: (kv[1]["f1"], kv[0]))[:10]
    lines += ["", "## Validation results (best epoch)", "",
              "| Measure | Result |", "|---|---:|",
              "| Accuracy | {} |".format(pct(metrics["accuracy"])),
              "| Macro-F1 | {} |".format(pct(metrics["macro_f1"])),
              "| Weighted-F1 | {} |".format(pct(metrics["weighted_f1"])), "",
              "Lowest per-class F1:", "", "| Intent | Precision | Recall | F1 | Support |", "|---|---:|---:|---:|---:|"]
    for name, sc in lowest:
        lines.append("| {} | {} | {} | {} | {} |".format(
            name, pct(sc["precision"]), pct(sc["recall"]), pct(sc["f1"]), sc["support"]))
    lines += ["", "Most frequent confusions:", "", "| True intent | Predicted intent | Count |", "|---|---|---:|"]
    lines += ["| {} | {} | {} |".format(t, p, n) for n, t, p in evaluation.top_confusions(metrics)]
    lines += ["", "## Confidence threshold", "",
              "| Threshold | Automatic rate | Accuracy when automatic | Wrong automatic answers | Escalated |",
              "|---:|---:|---:|---:|---:|"]
    for r in sweep:
        lines.append("| {:.2f} | {} | {} | {} | {} |".format(
            r["threshold"], pct(r["automatic_rate"]), pct(r["automatic_accuracy"]),
            r["wrong_automatic_answers"], r["escalated"]))
    lines += ["", "Calibration (ECE, 10 bins): {:.4f}.".format(calibration["ece"]), "",
              "| Confidence bin | Queries | Mean confidence | Accuracy |", "|---|---:|---:|---:|"]
    for r in calibration["bins"]:
        lines.append("| {:.1f}-{:.1f} | {} | {} | {} |".format(
            r["low"], r["high"], r["count"], pct(r["mean_confidence"]), pct(r["accuracy"])))
    lines += ["", "## Runtime", "",
              "Single-query classifier time on this hardware (n={}, after one warm-up call): mean {:.2f} ms, "
              "median {:.2f} ms, max {:.2f} ms.".format(rt["count"], rt["mean_ms"], rt["median_ms"], rt["max_ms"]), "",
              "## Reproduce", "", "```", "python train_distilbert.py {}".format(
                  "--pilot-per-class {}".format(st["pilot_per_class"]) if st["pilot_per_class"] else ""), "```", ""]
    return "\n".join(lines)


def main(argv=None):
    args = parse_args(argv)
    try:
        run(args)
    except (OSError, ValueError, csv.Error, ImportError) as error:
        print("DistilBERT run failed: {}".format(error), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
