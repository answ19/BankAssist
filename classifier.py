"""TF-IDF + Logistic Regression baseline intent classifier.

Trained on data/processed/train.csv only. Hyperparameters are chosen by 5-fold
cross-validation inside the training split, so the validation split stays unseen.

Prediction output format (shared with Anuja's baseline, used by routing.route_query):
    {"query", "predicted_intent", "confidence_score", "top_intents",
     "model_name", "model_runtime_ms"}
confidence_score is the largest class probability from predict_proba; top_intents
lists the top three as [{"intent": ..., "probability": ...}, ...], highest first.

    python classifier.py            # train and save models/baseline.joblib
"""

import csv
import json
import sys
import time
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline

from inspect_data import normalize_question


ROOT = Path(__file__).resolve().parent
TRAIN_PATH = ROOT / "data" / "processed" / "train.csv"
MAPPING_PATH = ROOT / "data" / "processed" / "label_mapping.json"
MODEL_PATH = ROOT / "models" / "baseline.joblib"
MODEL_NAME = "tfidf-logreg-baseline"
RANDOM_STATE = 42
PARAM_GRID = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "logreg__C": [1.0, 10.0, 100.0],
}


def read_split(path):
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    return [row["text_clean"] for row in rows], [row["category"] for row in rows]


def read_label_mapping(path=MAPPING_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_pipeline():
    return Pipeline([
        ("tfidf", TfidfVectorizer(lowercase=True, sublinear_tf=True)),
        ("logreg", LogisticRegression(max_iter=5000)),
    ])


def train(train_path=TRAIN_PATH, model_path=MODEL_PATH):
    """Fit on the training split, save the best pipeline, and return the search."""
    texts, labels = read_split(train_path)
    missing = set(read_label_mapping()) - set(labels)
    if missing:
        raise ValueError("Training split lacks intents: {}".format(", ".join(sorted(missing))))
    search = GridSearchCV(
        build_pipeline(), PARAM_GRID, scoring="f1_macro", n_jobs=-1,
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE))
    search.fit(texts, labels)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(search.best_estimator_, model_path)
    return search


def load_model(model_path=MODEL_PATH):
    if not Path(model_path).is_file():
        raise ValueError("No trained model at {}; run python classifier.py first.".format(
            model_path))
    return joblib.load(model_path)


def predict(query, model, top_k=3):
    """Classify one query and return it in the shared prediction-output format."""
    start = time.perf_counter()
    probabilities = model.predict_proba([normalize_question(query)])[0]
    ranked = sorted(zip(model.classes_, probabilities), key=lambda pair: -pair[1])
    runtime_ms = (time.perf_counter() - start) * 1000
    return {
        "query": query,
        "predicted_intent": str(ranked[0][0]),
        "confidence_score": float(ranked[0][1]),
        "top_intents": [{"intent": str(intent), "probability": float(score)}
                        for intent, score in ranked[:top_k]],
        "model_name": MODEL_NAME,
        "model_runtime_ms": runtime_ms,
    }


def main():
    try:
        search = train()
    except (OSError, ValueError, csv.Error) as error:
        print("Training failed: {}".format(error), file=sys.stderr)
        return 1
    print("Best cross-validated macro-F1 on the training split: {:.4f}".format(
        search.best_score_))
    print("Best parameters: {}".format(search.best_params_))
    print("Saved model: {}".format(MODEL_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
