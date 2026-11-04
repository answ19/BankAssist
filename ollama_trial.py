"""Small local-model trial: classify validation queries with an Ollama model.

Compares the local model with the baseline on the same stratified sample of the
validation split (never the official test set) and measures local runtime. The
model may only answer with one of the 77 intent labels (JSON-schema output).

Requires Ollama running locally with the model pulled, e.g.:
    ollama pull qwen2.5:3b
    python ollama_trial.py --per-intent 1
Set OLLAMA_MODEL or pass --model to try another model.
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
import urllib.error
import urllib.request
from collections import defaultdict
from datetime import date
from pathlib import Path

import classifier


ROOT = Path(__file__).resolve().parent
VALIDATION_PATH = ROOT / "data" / "processed" / "validation.csv"
REPORT_PATH = ROOT / "docs" / "ollama_trial.md"
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:3b")


def sample_validation(per_intent, seed=42):
    """Pick per_intent queries for each intent, reproducibly."""
    texts, labels = classifier.read_split(VALIDATION_PATH)
    by_intent = defaultdict(list)
    for text, label in zip(texts, labels):
        by_intent[label].append(text)
    generator = random.Random(seed)
    return [(text, label) for label in sorted(by_intent)
            for text in generator.sample(by_intent[label], min(per_intent, len(by_intent[label])))]


def build_prompt(labels):
    return ("You classify a banking customer's question into exactly one intent label. "
            "Choose the single best label from this list:\n" + "\n".join(labels) +
            '\nReply only with JSON: {"intent": "<label>"}')


def ask_ollama(query, labels, model, system_prompt):
    payload = {
        "model": model,
        "stream": False,
        "options": {"temperature": 0},
        "format": {"type": "object", "required": ["intent"],
                   "properties": {"intent": {"type": "string", "enum": labels}}},
        "messages": [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": query}],
    }
    request = urllib.request.Request(OLLAMA_URL + "/api/chat",
                                     data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=300) as response:
        body = json.loads(response.read().decode("utf-8"))
    runtime_ms = (time.perf_counter() - start) * 1000
    try:
        intent = json.loads(body["message"]["content"])["intent"]
    except (KeyError, ValueError, TypeError):
        intent = None
    return intent, runtime_ms


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=OLLAMA_MODEL, help="Ollama model name")
    parser.add_argument("--per-intent", type=int, default=1, help="queries per intent")
    args = parser.parse_args(argv)
    labels = list(classifier.read_label_mapping())
    sample = sample_validation(args.per_intent)
    system_prompt = build_prompt(labels)
    baseline = classifier.load_model()
    try:
        ask_ollama(sample[0][0], labels, args.model, system_prompt)  # load model; not timed
    except urllib.error.URLError as error:
        print("Cannot reach Ollama at {} ({}). Is it running, and is {} pulled?".format(
            OLLAMA_URL, error, args.model), file=sys.stderr)
        return 1

    rows = []
    for number, (query, true) in enumerate(sample, start=1):
        intent, runtime_ms = ask_ollama(query, labels, args.model, system_prompt)
        base = classifier.predict(query, baseline)
        rows.append((query, true, intent, runtime_ms, base["predicted_intent"],
                     base["model_runtime_ms"]))
        print("{}/{} {:.0f} ms".format(number, len(sample), runtime_ms), end="\r")
    count = len(rows)
    local_correct = sum(r[2] == r[1] for r in rows)
    base_correct = sum(r[4] == r[1] for r in rows)
    invalid = sum(r[2] is None for r in rows)
    local_ms = [r[3] for r in rows]
    lines = [
        "# Ollama local-model trial",
        "",
        "Date: {}. Model: `{}` via Ollama at {}. Sample: {} validation queries "
        "({} per intent, seed 42). The official test set was not used.".format(
            date.today(), args.model, OLLAMA_URL, count, args.per_intent),
        "",
        "The model sees the 77 intent labels and must answer with one of them (JSON schema). "
        "Temperature 0. The first, untimed call loads the model into memory.",
        "",
        "| Measure | {} | Baseline (TF-IDF + LR) |".format(args.model),
        "|---|---:|---:|",
        "| Accuracy on this sample | {:.2%} ({}/{}) | {:.2%} ({}/{}) |".format(
            local_correct / count, local_correct, count, base_correct / count, base_correct, count),
        "| Unparseable replies | {} | 0 |".format(invalid),
        "| Mean time per query | {:.0f} ms | {:.2f} ms |".format(
            statistics.mean(local_ms), statistics.mean(r[5] for r in rows)),
        "| Median time per query | {:.0f} ms | {:.2f} ms |".format(
            statistics.median(local_ms), statistics.median(r[5] for r in rows)),
        "",
        "With one query per intent, each intent counts once, so accuracy here is also a "
        "rough balanced accuracy. The sample is small; treat differences of a few queries "
        "as noise.",
        "",
        "## Local-model errors",
        "",
        "| Query | True intent | {} | Baseline |".format(args.model),
        "|---|---|---|---|",
    ]
    for query, true, intent, _, base_intent, _ in rows:
        if intent != true:
            lines.append("| {} | {} | {} | {} |".format(
                query.replace("|", r"\|"), true, intent, base_intent))
    report = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(report)
    print("Saved: {}".format(REPORT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
