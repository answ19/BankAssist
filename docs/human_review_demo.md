# Human-review demonstration (real model output)

Exported from `bankassist.db` on 2026-11-10. Every prediction and confidence score comes from the trained baseline (`tfidf-logreg-baseline`); none are hand-entered. Threshold 0.90.

**Who did what.**

- **FAQ answers:** the 77 drafts in `data/faq/faq_drafts.csv` are drafts for team review. Kashish Jethmalani reviewed and approved six of them for this demo on 2026-11-09. The other 71 are still unapproved drafts, so those intents always go to review regardless of confidence.
- **Review decisions:** every decision, final response and reviewer note below was entered by Kashish Jethmalani in `review_cli.py`. This terminal version of the review screen was used because the Streamlit page would not open on that laptop. It saves through `database.save_review`, the same function `review_app.py` uses.

Requests 1-2 are ad-hoc queries routed before any FAQ was approved. Requests 3, 4, 6, 8, 9 and 10 are the first validation-split query (file order) for each approved intent, logged with their dataset label. Requests 9 and 10 were re-routed on 2026-11-10 against the human-approved FAQs. They replace requests 5 and 7, which had been routed before these approvals were made; the predictions and scores are unchanged.

| # | Query | Predicted intent | Confidence | True intent | Status | Corrected intent | Response sent | Reviewer notes |
|---:|---|---|---:|---|---|---|---|---|
| 1 | My card was stolen last night | lost_or_stolen_card | 0.82 | n/a | edited |  | I'm sorry your card was stolen. Please freeze or block your card in the app immediately and contact support. You can also order a replacement card through the app. | The predicted intent (lost_or_stolen_card) is correct with confidence 0.82. However, the system failed to retrieve the relevant approved FAQ 43 for stolen cards. The response has been corrected using the approved FAQ guidance. Recommend improving FAQ retrieval to prevent unnecessary escalation. |
| 2 | Which stocks should I buy? | declined_transfer | 0.11 | n/a | rejected |  | (none) | Incorrect intent classification. The query is about investment advice, not a declined transfer. Confidence is very low (0.11). This should be classified as out-of-scope or investment advice rather than declined_transfer. |
| 3 | I need to report a stolen card | lost_or_stolen_card | 0.89 | lost_or_stolen_card | edited |  | I'm sorry your card was stolen. Please freeze or block your card in the app immediately and contact support. You can also order a replacement card through the app. | Intent classification is correct (0.89 confidence). FAQ 43 provides an approved answer, but the system unnecessarily escalated the request instead of using the FAQ guidance. |
| 4 | There's a possibility that my card was compromised. | compromised_card | 0.18 | compromised_card | edited |  | If you suspect your card has been compromised, please freeze or block it in the app right away and contact support immediately. Support can help you review suspicious transactions and order a replacement card. | The predicted intent (compromised_card) is correct despite the low confidence score (0.18). FAQ 24 provides relevant approved guidance. The system unnecessarily escalated instead of providing the available FAQ answer. Recommend improving intent confidence and FAQ response retrieval. |
| 6 | Can you tell me why I haven't received my new card yet? | card_arrival | 0.81 | card_arrival | edited |  | You can check the status of your new card order in the app. If your card hasn't arrived by the expected delivery date, please contact support through the app. They can check the delivery status and arrange a replacement if needed. | The predicted intent (card_arrival) is correct with high confidence (0.81). FAQ 13 provides a relevant approved answer, but the system unnecessarily escalated the request. The response should use the approved FAQ guidance instead. |
| 8 | Why am I being declined? | declined_transfer | 0.80 | declined_card_payment | edited | declined_card_payment | Could you please clarify whether you're referring to a declined card payment or a bank transfer? This will help us assist you with the correct issue. | The predicted intent (declined_transfer) is incorrect. The dataset's actual label is declined_card_payment. Although the query is ambiguous, the corrected intent should reflect the dataset label. No approved FAQ was retrieved, so a clarification response is appropriate. |
| 9 | What do I do if my smart phone was stolen? | lost_or_stolen_phone | 0.97 | lost_or_stolen_phone | answered |  | If you can, log in to the app from another device and freeze or block your card, then contact support immediately so they can help secure your account. Support can also help you regain access to the app on a new device. |  |
| 10 | I want to activate the card. | activate_my_card | 0.98 | activate_my_card | answered |  | You can activate your card in the app by following the card activation steps. If activation fails, make sure your app is up to date and try again, then contact support in the app if it still doesn't work. |  |

**How to read the reviewer notes.** Requests 1, 3, 4, 6 and 8 had confidence below 0.90, so sending them to review is the configured routing policy, not a system error. They were logged before the review screen showed the escalation reason, and the screen labelled the held-back FAQ as "Retrieved FAQ"; both screens now label it as held back and state the reason. Where a note calls an escalation "unnecessary" or a retrieval failure, it means the predicted intent was correct and the approved FAQ would have been a suitable answer, not that routing malfunctioned. Request 1 was routed before any FAQ had been approved. Three predictions the reviewer judged correct (requests 1, 3 and 6) scored between 0.81 and 0.89, which motivates the threshold sensitivity check in `docs/baseline_results.md` (at 0.80, 59.35% of validation queries are answered automatically with 98.74% accuracy).

## Demo metrics

Eight requests only: these show the workflow end to end and are not a model evaluation (see `docs/baseline_results.md` for the 2,000-query validation results). Classifier time here includes each script's first-call warm-up. FAQ lookup time is measured the same way as `python evaluation.py`: 20 repeated SQLite lookups for each predicted intent.

| Measure | Result | How it is obtained |
|---|---:|---|
| Accuracy | 83.33% | Correct predictions divided by labelled requests (n=6) |
| Macro-F1 | 71.43% | Average F1 across intents |
| Weighted-F1 | 83.33% | F1 weighted by class support |
| FAQ retrieval accuracy | Pending | Relevant FAQ retrieved for the query (manual check) |
| Automatic answer rate | 25.00% | Answered requests divided by all requests (n=8) |
| Escalation rate | 75.00% | Pending-review or reviewed requests divided by all requests |
| Approval rate | 0.00% | Approved reviews divided by all reviews (n=6) |
| Edit rate | 83.33% | Edited reviews divided by all reviews |
| Rejection rate | 16.67% | Rejected reviews divided by all reviews |
| Reviewer agreement | 80.00% | Reviews keeping the predicted intent (n=5) |
| Average classifier time | 142.7 ms | Mean logged classifier prediction time per query (n=8) |
| Average Ollama response time | Pending | Mean logged local LLM response time per query (n=0) |
| Average FAQ lookup time | 2.9 ms | Mean SQLite FAQ lookup time (n=120) |
