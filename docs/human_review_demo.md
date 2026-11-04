# Human-review demonstration (real model output)

Exported from `bankassist.db` on 2026-11-03. Every prediction and confidence score comes from the trained baseline (`tfidf-logreg-baseline`); none are hand-entered. Threshold 0.90. Six FAQs were approved for this demo (see `data/faq/faq_drafts.csv`); every other intent goes to review regardless of confidence.

Requests 1-2 are ad-hoc queries routed before any FAQ was approved. Requests 3-8 are the first validation-split query (file order) for each approved intent, logged with their dataset label. Review decisions were made by Claude at Kashish Jethmalani's request through `database.save_review`, the same function the Streamlit screen uses.

| # | Query | Predicted intent | Confidence | True intent | Status | Corrected intent | Reviewer notes |
|---:|---|---|---:|---|---|---|---|
| 1 | My card was stolen last night | lost_or_stolen_card | 0.82 | n/a | edited |  | Intent correct (0.82, below threshold). No FAQ was approved when this was routed, so the now-approved lost/stolen card FAQ is sent. |
| 2 | Which stocks should I buy? | declined_transfer | 0.11 | n/a | rejected |  | Out of scope: investment advice is not a banking intent. Low confidence (0.11) correctly sent it to review. |
| 3 | I need to report a stolen card | lost_or_stolen_card | 0.89 | lost_or_stolen_card | approved |  | Intent and FAQ correct; confidence 0.89 just below 0.90. |
| 4 | There's a possibility that my card was compromised. | compromised_card | 0.18 | compromised_card | approved |  | Intent correct despite low confidence (0.18); FAQ fits. |
| 5 | What do I do if my smart phone was stolen? | lost_or_stolen_phone | 0.97 | lost_or_stolen_phone | answered |  |  |
| 6 | Can you tell me why I haven't received my new card yet? | card_arrival | 0.81 | card_arrival | approved |  | Intent and FAQ correct; confidence 0.81. |
| 7 | I want to activate the card. | activate_my_card | 0.98 | activate_my_card | answered |  |  |
| 8 | Why am I being declined? | declined_transfer | 0.80 | declined_card_payment | edited | declined_card_payment | Predicted declined_transfer is wrong for the dataset label. The query is ambiguous (card or transfer); a live reviewer would ask. Sent the declined card payment FAQ. |

## Demo metrics

Eight requests only: these show the workflow end to end and are not a model evaluation (see `docs/baseline_results.md` for the 2,000-query validation results). Classifier time here includes each script's first-call warm-up.

| Measure | Result | How it is obtained |
|---|---:|---|
| Accuracy | 83.33% | Correct predictions divided by labelled requests (n=6) |
| Macro-F1 | 71.43% | Average F1 across intents |
| Weighted-F1 | 83.33% | F1 weighted by class support |
| FAQ retrieval accuracy | Pending | Relevant FAQ retrieved for the query (manual check) |
| Automatic answer rate | 25.00% | Answered requests divided by all requests (n=8) |
| Escalation rate | 75.00% | Pending-review or reviewed requests divided by all requests |
| Approval rate | 50.00% | Approved reviews divided by all reviews (n=6) |
| Edit rate | 33.33% | Edited reviews divided by all reviews |
| Rejection rate | 16.67% | Rejected reviews divided by all reviews |
| Reviewer agreement | 80.00% | Reviews keeping the predicted intent (n=5) |
| Average classifier time | 66.3 ms | Mean logged classifier prediction time per query (n=8) |
| Average Ollama response time | Pending | Mean logged local LLM response time per query (n=0) |
| Average FAQ lookup time | Pending | Mean SQLite FAQ lookup time (n=0) |
