# BankAssist

A local banking customer-support prototype developed by Team TriNova
for CECS 551 — Advanced Artificial Intelligence.

## Project Overview

BankAssist identifies the intent of a banking question and retrieves
an approved FAQ answer. Questions that do not meet the confidence
threshold, or lack an approved FAQ, are sent for human review.

Workflow:
Customer query → preprocessing → intent classification →
confidence and FAQ checks → automatic response or human review.

The prototype does not access bank accounts or perform transactions.

## Team

| Member | Main Responsibility |
|---|---|
| Anuja Sawant | Preprocessing, shared splits, baseline comparison and integration |
| Suraj Papanna | Dataset exploration and DistilBERT experiments |
| Sakshi Tejwani | Architecture, FAQ content and Ollama workflow |
| Kashish Jethmalani | SQLite, routing, human-review tools and evaluation |

All members contribute to code, documentation and peer review.

## Current Status

Phase 3 — Preliminary Data Analysis and Feasibility Analysis.

Completed:
- Dataset inspection, preprocessing and shared data splits.
- EDA notebook, figures and summary.
- TF-IDF + Logistic Regression baseline.
- DistilBERT pilot and full training experiment.
- SQLite storage and confidence-based routing.
- Human-review interfaces in Streamlit and the terminal.
- Local Ollama trials and evaluation scripts.

The current application uses the TF-IDF baseline.
DistilBERT is evaluated separately as a candidate replacement.

## Dataset

BANKING77:
https://github.com/PolyAI-LDN/task-specific-datasets

- Original training records: 10,003.
- Records after removing four duplicate copies: 9,999.
- Shared training split: 7,999.
- Shared validation split: 2,000.
- Intent classes: 77.

The official test set remains reserved for final evaluation.
All models use the same saved splits and label mapping.

## Validation Results

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| Selected TF-IDF + Logistic Regression | 88.25% | 88.30% |
| DistilBERT | 90.95% | 91.25% |

These are validation results, not official test-set results.

Details:
- `docs/baseline_results.md`
- `docs/distilbert_results.md`

## Response and Review Policy

An automatic response requires:
1. Intent confidence of at least 0.90.
2. An approved FAQ for the predicted intent.

Otherwise, the request goes to human review.

Six FAQs are currently approved for the demonstration.
The remaining 71 entries are unapproved drafts.

Ollama (`qwen2.5:3b`) can optionally phrase an approved FAQ answer.
It is disabled by default. Intent confidence does not measure
the factual correctness of an answer.

## Local Setup

Activate your Python environment, then install dependencies:

```powershell
python -m pip install -r requirements.txt
```

Train the baseline if its local model file is unavailable:

```powershell
python classifier.py
```

Create or update the FAQ database:

```powershell
python setup_database.py
```

## Run the Application

Classify and route a question:

```powershell
python pipeline.py "My card has not arrived"
```

Optionally enable Ollama, with Ollama running locally
and `qwen2.5:3b` already downloaded:

```powershell
python pipeline.py --ollama "My card has not arrived"
```

Review pending requests using either interface:

```powershell
python -m streamlit run review_app.py
```

```powershell
python review_cli.py
```

## Tests

```powershell
python -m unittest discover -s tests -v
```

Some deep-model tests require the additional dependencies
in `requirements-deep.txt`.

## DistilBERT Experiment

Use a separate environment or Colab GPU runtime:

```powershell
python -m pip install -r requirements.txt -r requirements-deep.txt
python train_distilbert.py --epochs 8
```

Eight epochs match the recorded full-run configuration.
New runs may differ across hardware and software environments.

Model binaries are excluded from Git and must be retained
locally or rebuilt.

## Project Structure

- `data/` — raw data, shared splits, labels and FAQ content.
- `notebooks/` — EDA and Colab experiment notebooks.
- `results/` — predictions, figures and experiment summaries.
- `docs/` — implementation notes, evaluation and demonstrations.
- `tests/` — automated tests.
- Root Python files — preprocessing, models, routing and review tools.

## Working Guidelines

- Use the shared splits and original intent labels.
- Fit learned preprocessing only on training data.
- Keep the official test set reserved for final evaluation.
- Distinguish experiments from the deployed application workflow.
- Report measured results with their limitations.
