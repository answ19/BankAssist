"""Run the routing threshold test cases and the human-review example end to end.

Queries and true intents come from data/processed/validation.csv, but every
predicted intent and confidence score here is HAND-ENTERED to exercise the
routing code. Rows are saved with is_test_case = 1 and are not model results.
The FAQ answers are placeholders, not team-approved FAQs.

    python demo_review.py --reset                 # route, then review in Streamlit
    python demo_review.py --reset --auto-review   # route and apply scripted reviews
"""

import argparse
import sys
from pathlib import Path

import database
import evaluation
import routing


ROOT = Path(__file__).resolve().parent
DEMO_DB_PATH = ROOT / "bankassist_demo.db"
TEST_SOURCE = "TEST FIXTURE: placeholder text, not a team-approved FAQ"
TEST_MODEL = "TEST: hand-entered score"

TEST_FAQS = [
    ("lost_or_stolen_card", "What should I do if my card is lost or stolen?",
     "[PLACEHOLDER] Freeze the card in the app and contact support to order a replacement."),
    ("card_arrival", "My new card has not arrived. What should I do?",
     "[PLACEHOLDER] Check the delivery status in the app; contact support if it is overdue."),
    ("pending_cash_withdrawal", "Why is my cash withdrawal still pending?",
     "[PLACEHOLDER] Withdrawals can show as pending until the ATM operator confirms them."),
]

# (validation record_id, query, true intent, hand-entered prediction, test score)
TEST_CASES = [
    (1479, "I need to report a stolen card", "lost_or_stolen_card",
     "lost_or_stolen_card", 0.95),
    (107, "My card hasn't came in the mail yet?", "card_arrival", "card_arrival", 0.90),
    (870, "My balance is not what I thought it was following an ATM withdrawal",
     "pending_cash_withdrawal", "pending_cash_withdrawal", 0.82),
    (824, "Why does it still charge me even though I was unable to withdraw money "
          "from the ATM?", "pending_cash_withdrawal", "card_arrival", 0.65),
    (None, "Which stocks should I buy this year?", None, "card_arrival", 0.40),
]

# Scripted decisions for --auto-review, applied in order to the pending requests.
AUTO_REVIEWS = [
    ("approved", None, None, "Prediction and FAQ match the query."),
    ("edited", "pending_cash_withdrawal", TEST_FAQS[2][2],
     "Wrong intent predicted; replaced with the pending cash withdrawal FAQ."),
    ("rejected", None, "We cannot give investment advice. Please speak to a financial adviser.",
     "Out of scope: investment advice is not a supported banking intent."),
]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(DEMO_DB_PATH), help="demo database path")
    parser.add_argument("--reset", action="store_true", help="delete the demo database first")
    parser.add_argument("--auto-review", action="store_true",
                        help="apply scripted reviewer decisions instead of using Streamlit")
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    if db_path.exists():
        if not args.reset:
            print("{} exists; rerun with --reset to recreate it.".format(db_path),
                  file=sys.stderr)
            return 1
        db_path.unlink()

    database.initialize_database(db_path)
    for intent, question, answer in TEST_FAQS:
        database.add_faq(intent, question, answer, TEST_SOURCE, approved=True, db_path=db_path)

    print("Threshold: {:.2f} (TEST scores, not model output)\n".format(
        routing.CONFIDENCE_THRESHOLD))
    for record_id, query, true_intent, predicted, score in TEST_CASES:
        result = routing.route_query(query, predicted, score, true_intent=true_intent,
                                     model_name=TEST_MODEL, is_test_case=True,
                                     db_path=db_path)
        print("#{request_id} score {confidence_score:.2f} -> {status:<14} "
              "predicted={predicted_intent} faq_id={faq_id}".format(**result))
        print("    query: {}".format(query))

    if args.auto_review:
        print("\nApplying scripted reviewer decisions:")
        for request, review in zip(database.get_pending_reviews(db_path), AUTO_REVIEWS):
            decision, corrected, response, notes = review
            database.save_review(request["request_id"], decision, response, corrected,
                                 notes, db_path)
            print("#{} -> {}".format(request["request_id"], decision))
        print("\nSaved records (requests LEFT JOIN reviews):")
        for row in database.get_evaluation_records(include_test_cases=True, db_path=db_path):
            print("#{request_id} status={status} decision={decision} "
                  "corrected_intent={corrected_intent}".format(**row))
            print("    response sent: {}".format(
                row["final_response"] if row["decision"] else row["system_response"]))
        print("\nTEST-CASE metrics (routing check only, not model results):\n")
        records = database.get_evaluation_records(include_test_cases=True, db_path=db_path)
        print(evaluation.format_evaluation_table(evaluation.evaluate_records(records)))
    else:
        print("\nReview the pending requests with:")
        print('    $env:BANKASSIST_DB_PATH="{}"; streamlit run review_app.py'.format(db_path))
    return 0


if __name__ == "__main__":
    sys.exit(main())
