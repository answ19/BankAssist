# BankAssist: database, routing, and evaluation

Owner: Kashish Jethmalani (Database and Evaluation Lead). Branch: `kashish-database-evaluation`.

These modules store and evaluate the shared classifier's output. They do not classify
anything and do not depend on the final label mapping: intents are stored as plain text.
They use only the Python standard library; the reviewer screen also needs Streamlit.

| File | Purpose |
| --- | --- |
| `database.py` | SQLite tables and functions for FAQs, requests, and reviews |
| `routing.py` | Confidence routing: approved FAQ answer or `pending_review` |
| `evaluation.py` | Classification, review, and runtime metrics; prints the evaluation table |
| `review_app.py` | Minimal Streamlit reviewer screen (approve, edit, reject, notes) |
| `demo_review.py` | Threshold test cases and the end-to-end human-review example |

## Tables (`bankassist.db`)

- **faqs**: `faq_id, intent, question, answer, source, approved, created_at`. Only rows
  with `approved = 1` are ever retrieved.
- **requests**: `request_id, query, predicted_intent, confidence_score, faq_id,
  system_response, status, created_at`, plus four columns added for evaluation:
  `true_intent` (gold label, when known), `model_name`, `model_runtime_ms`, and
  `is_test_case` (1 for hand-entered scores; excluded from reported results by default).
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
python -m unittest tests.test_database tests.test_routing tests.test_evaluation -v
python demo_review.py --reset --auto-review       # scripted review, prints saved rows
python demo_review.py --reset                     # leave requests pending, then:
$env:BANKASSIST_DB_PATH="bankassist_demo.db"; streamlit run review_app.py
python evaluation.py --db bankassist.db           # evaluation table (real results only)
```

`demo_review.py` uses five validation-set queries with **hand-entered** predictions and
scores (0.95, 0.90, 0.82, 0.65, plus an out-of-scope query at 0.40) and placeholder FAQ
answers. All of its rows are marked `is_test_case = 1`. They test the routing and review
code only and are not model results or evidence that the confidence score is calibrated.

## Connecting the classifier

When the shared prediction function is ready, pass its output to
`routing.route_query(query, predicted_intent, confidence_score, true_intent=...,
model_name=..., model_runtime_ms=...)`. Load approved FAQs with
`database.load_faqs_from_csv(path, known_intents=label_mapping)` so an FAQ with a
misspelled intent fails loudly instead of never being retrieved.

## Still needed from the team

- Final prediction-output format and confidence-score method (Anuja, Suraj)
- Approved FAQ file with a `source` for each answer (Sakshi)
- Selected Ollama model name, to log runtimes against it (Sakshi)
