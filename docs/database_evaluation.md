# BankAssist: database, routing, and evaluation

Owner: Kashish Jethmalani (Database and Evaluation Lead). Branch: `kashish-database-evaluation`.

`database.py`, `routing.py`, and `evaluation.py` store and evaluate the classifier's
output. They do not depend on the final label mapping (intents are stored as plain text)
and use only the Python standard library. The classifier needs scikit-learn and the
reviewer screen needs Streamlit (`requirements.txt`).

| File | Purpose |
| --- | --- |
| `database.py` | SQLite tables and functions for FAQs, requests, and reviews |
| `routing.py` | Confidence routing: approved FAQ answer or `pending_review` |
| `evaluation.py` | Classification, review, and runtime metrics; prints the evaluation table |
| `review_app.py` | Minimal Streamlit reviewer screen (approve, edit, reject, notes) |
| `demo_review.py` | Threshold test cases and the end-to-end human-review example |
| `classifier.py` | TF-IDF + Logistic Regression baseline; shared prediction format |
| `baseline_evaluation.py` | Validation metrics, threshold sweep, calibration, runtime |
| `pipeline.py` | Classify, route, and log one query |
| `setup_database.py` | Build `bankassist.db` and load the FAQ CSV |
| `ollama_trial.py` | Local-model classification trial against the baseline |

## Tables (`bankassist.db`)

- **faqs**: `faq_id, intent, question, answer, source, approved, created_at`. Only rows
  with `approved = 1` are ever retrieved.
- **requests**: `request_id, query, predicted_intent, confidence_score, faq_id,
  system_response, status, created_at`, plus five columns added for evaluation:
  `true_intent` (gold label, when known), `model_name`, `model_runtime_ms` (classifier
  prediction time only), `ollama_runtime_ms` (local LLM response time, tracked
  separately), and `is_test_case` (1 for hand-entered scores; excluded from reported
  results by default). `initialize_database` adds missing columns to older databases.
- **reviews**: `review_id, request_id, decision, corrected_intent, final_response,
  reviewer_notes, reviewed_at`. One review per request.

Statuses: `answered`, `pending_review`, `approved`, `edited`, `rejected`.

## Routing rule

`CONFIDENCE_THRESHOLD` defaults to 0.90 and can be overridden with the
`BANKASSIST_CONFIDENCE_THRESHOLD` environment variable or the `threshold` argument.

- Confidence >= threshold **and** an approved FAQ exists for the predicted intent:
  status `answered`; the customer sees the approved FAQ answer.
- Otherwise: status `pending_review`; the customer sees an escalation message. Any FAQ
  found is stored as evidence for the reviewer but is not shown to the customer.

Every routed request is saved, so the review history is traceable.

## Review decisions

- **Approve**: the retrieved FAQ answer becomes the final response. Not allowed when no
  FAQ was retrieved.
- **Edit**: the reviewer writes the final response and may correct the intent.
- **Reject**: no FAQ answer is sent; an optional referral message may be stored.

## Metric definitions

- Accuracy, macro-F1, weighted-F1, per-class F1, confusion matrix: computed over requests
  that have both `true_intent` and `predicted_intent`. Zero denominators count as 0,
  matching scikit-learn's default.
- Escalation rate: requests not answered automatically (pending or reviewed) / all requests.
- Approval, edit, and rejection rates: decisions of each type / all reviews.
- Reviewer agreement: reviews that kept the predicted intent (no different
  `corrected_intent`). A rejection with no corrected intent is excluded, since it gives no
  intent judgement.
- Recall@k: `evaluation.recall_at_k`, ready for when FAQ retrieval returns ranked results.
- Runtime: mean/median/max of logged `model_runtime_ms`; SQLite FAQ lookup time is measured
  directly by `time_faq_lookups`.

Metrics with no recorded data are shown as **Pending**, never estimated.

## How to run

```powershell
python -m unittest discover -s tests -v            # all tests
python demo_review.py --reset --auto-review       # scripted review, prints saved rows
python demo_review.py --reset                     # leave requests pending, then:
$env:BANKASSIST_DB_PATH="bankassist_demo.db"; streamlit run review_app.py
python evaluation.py --db bankassist.db           # evaluation table (real results only)
```

`demo_review.py` uses five validation-set queries with **hand-entered** predictions and
scores (0.95, 0.90, 0.82, 0.65, plus an out-of-scope query at 0.40) and placeholder FAQ
answers. All of its rows are marked `is_test_case = 1`. They test the routing and review
code only and are not model results or evidence that the confidence score is calibrated.

## Integration decisions

These were first chosen to unblock integration, then confirmed by Anuja: the
prediction fields, the confidence definition (highest class probability, top three
intents), evaluating 0.90 on validation data, drafts staying at `approved = 0`, human
review for any query without an approved FAQ, and `qwen2.5:3b`. The confidence score
measures confidence in the predicted intent, not the correctness of an answer. The
routing result also returns `faq_answer` and `status` for storage.

Ownership going forward: Anuja owns baseline training and shares the saved model and
validation predictions for evaluation; Sakshi coordinates FAQ review and the Ollama
integration. A small set of FAQs is reviewed first for the demonstration.

| Item | Choice | Where |
|---|---|---|
| Prediction output | `query, predicted_intent, confidence_score, top_intents, model_name, model_runtime_ms` | `classifier.predict` |
| Confidence score | Largest class probability of TF-IDF + Logistic Regression (`predict_proba`) | `classifier.py` |
| Threshold | Kept at 0.90; validation evidence in `docs/baseline_results.md` | `routing.CONFIDENCE_THRESHOLD` |
| FAQ file | 77 drafts, one per intent, all `approved = 0` until a team member reviews them | `data/faq/faq_drafts.csv` |
| Ollama model | `qwen2.5:3b` (about 2 GB; fits CPU-only on a 16 GB laptop) | `ollama_trial.py`, `OLLAMA_MODEL` |

The baseline here was built because no shared classifier existed yet; it is offered to
Anuja for review and reuse. If her retrained model or Suraj's DistilBERT replaces it,
only `classifier.predict` must keep the same output keys.

## Full workflow

```powershell
py -3.11 -m venv .venv   # Python 3.11+ for the pinned versions; 3.14 was used here
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe classifier.py              # train on train split
.\.venv\Scripts\python.exe baseline_evaluation.py     # validation results report
.\.venv\Scripts\python.exe setup_database.py --reset  # bankassist.db + FAQs
.\.venv\Scripts\python.exe pipeline.py "My card was stolen"
.\.venv\Scripts\streamlit.exe run review_app.py       # review pending requests
.\.venv\Scripts\python.exe evaluation.py              # evaluation table from the database
.\.venv\Scripts\python.exe ollama_trial.py            # needs Ollama + qwen2.5:3b
```

## Approving FAQs

Read each row of `data/faq/faq_drafts.csv`. Set `approved` to 1 only when the answer is
correct for the intent and safe to send, and change `source` to name the reviewer. Then
run `setup_database.py` (no `--reset`): FAQs are updated in place, matched by intent and
question, and logged requests and reviews are kept. Intents without an approved FAQ always
go to human review, whatever the confidence.

Six FAQs are approved for the Phase 3 demo: `activate_my_card`, `card_arrival`,
`compromised_card`, `declined_card_payment`, `lost_or_stolen_card`, and
`lost_or_stolen_phone`. `pending_cash_withdrawal` was held back because its draft
says the amount "is usually returned", a policy claim the team should confirm. The
end-to-end demonstration is recorded in `docs/human_review_demo.md`.
