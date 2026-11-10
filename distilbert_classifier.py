"""DistilBERT intent classifier helpers, in the team's shared prediction format.

Same output keys as classifier.predict (TF-IDF baseline), so routing.route_query,
evaluation.py and the reviewer workflow work unchanged:
    {"query", "predicted_intent", "confidence_score", "top_intents",
     "model_name", "model_runtime_ms"}
confidence_score is the largest softmax probability. model_runtime_ms is the
classifier prediction time only (Ollama time is tracked separately).

torch and transformers are imported lazily, so the pure helpers below work (and
are unit-tested) on machines without them.

    python train_distilbert.py --pilot-per-class 10     # small pilot
    python train_distilbert.py                          # full run (GPU recommended)
"""

import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent
BASE_MODEL = "distilbert-base-uncased"
MODEL_DIR = ROOT / "models" / "distilbert"
MODEL_NAME = "distilbert-base-uncased-finetuned"


def prediction_from_probabilities(query, probabilities, classes, runtime_ms,
                                  model_name=MODEL_NAME, top_k=3):
    """Build the shared prediction dict from one row of class probabilities."""
    probabilities = np.asarray(probabilities, dtype=float)
    if probabilities.ndim != 1 or len(probabilities) != len(classes):
        raise ValueError("Need one probability per class ({} classes, got shape {}).".format(
            len(classes), probabilities.shape))
    order = np.argsort(-probabilities, kind="stable")[:top_k]
    top = [(str(classes[i]), float(probabilities[i])) for i in order]
    return {
        "query": query,
        "predicted_intent": top[0][0],
        "confidence_score": top[0][1],
        "top_intents": top,
        "model_name": model_name,
        "model_runtime_ms": float(runtime_ms),
    }


def stratified_subset(labels, per_class, seed=42):
    """Indices of up to `per_class` examples of every label (reproducible pilot subset)."""
    rng = np.random.RandomState(seed)
    by_label = {}
    for index, label in enumerate(labels):
        by_label.setdefault(label, []).append(index)
    chosen = []
    for label in sorted(by_label):
        indices = by_label[label]
        take = min(per_class, len(indices))
        chosen.extend(int(i) for i in rng.choice(indices, size=take, replace=False))
    return sorted(chosen)


def softmax(logits):
    logits = np.asarray(logits, dtype=float)
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def _pad_batch(id_lists, pad_id=0):
    width = max(len(ids) for ids in id_lists)
    ids = np.full((len(id_lists), width), pad_id, dtype=np.int64)
    mask = np.zeros((len(id_lists), width), dtype=np.int64)
    for row, seq in enumerate(id_lists):
        ids[row, :len(seq)] = seq
        mask[row, :len(seq)] = 1
    return ids, mask


def probabilities_for(bundle, texts, batch_size=64):
    """Softmax probabilities, shape (len(texts), n_classes), using a loaded model bundle."""
    import torch

    model, tokenizer = bundle["model"], bundle["tokenizer"]
    device = bundle["device"]
    model.eval()
    out = []
    with torch.no_grad():
        for start in range(0, len(texts), batch_size):
            chunk = list(texts[start:start + batch_size])
            encoded = tokenizer(chunk, truncation=True, max_length=bundle["max_length"])
            ids, mask = _pad_batch(encoded["input_ids"])
            logits = model(input_ids=torch.tensor(ids, device=device),
                           attention_mask=torch.tensor(mask, device=device)).logits
            out.append(softmax(logits.float().cpu().numpy()))
    return np.vstack(out)


def predict(query, bundle, top_k=3):
    """Classify one query (same call shape as classifier.predict)."""
    from inspect_data import normalize_question

    start = time.perf_counter()
    probabilities = probabilities_for(bundle, [normalize_question(query)], batch_size=1)[0]
    runtime_ms = (time.perf_counter() - start) * 1000
    return prediction_from_probabilities(query, probabilities, bundle["classes"], runtime_ms,
                                         bundle.get("model_name", MODEL_NAME), top_k)


def load_model(model_dir=MODEL_DIR, device=None):
    """Load a model saved by train_distilbert.py."""
    import json
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    model_dir = Path(model_dir)
    meta_path = model_dir / "bankassist_meta.json"
    if not meta_path.is_file():
        raise ValueError("No trained DistilBERT model at {}; run train_distilbert.py first.".format(
            model_dir))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    return {"model": model, "tokenizer": tokenizer, "classes": meta["classes"],
            "max_length": meta["max_length"], "device": device,
            "model_name": MODEL_NAME}
