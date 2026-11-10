"""Terminal version of the review screen, for when the Streamlit page cannot open.

Shows each pending request with its FAQ evidence and saves the reviewer's
decision through database.save_review, the same function review_app.py uses.

    python review_cli.py
Set BANKASSIST_DB_PATH or pass --db to review another database.
"""

import argparse
import json
import os
import sys
from pathlib import Path

import database


ROOT = Path(__file__).resolve().parent
MAPPING_PATH = ROOT / "data" / "processed" / "label_mapping.json"


def ask(prompt, choices=None):
    while True:
        answer = input(prompt).strip()
        if choices is None or answer.lower() in choices:
            return answer.lower() if choices else answer
        print("Please type one of: {}".format(", ".join(sorted(choices))))


def show(request, position, total):
    print("\n" + "=" * 70)
    print("Request #{} ({} of {} waiting)".format(request["request_id"], position, total))
    if request["is_test_case"]:
        print("TEST CASE: the prediction and score were entered by hand, not model output.")
    print("Customer query:   {}".format(request["query"]))
    print("Predicted intent: {}".format(request["predicted_intent"] or "None"))
    score = request["confidence_score"]
    print("Confidence score: {}".format("None" if score is None else "{:.2f}".format(score)))
    if request["faq_id"] is None:
        print("Retrieved FAQ:    none (no approved FAQ for the predicted intent)")
    else:
        print("Retrieved FAQ {}: Q: {}".format(request["faq_id"], request["faq_question"]))
        print("                  A: {}".format(request["faq_answer"]))
    print("System response:  {}".format(request["system_response"]))


def review(request, labels):
    """Ask for one decision; return save_review arguments, or None to skip."""
    choices = {"a", "e", "r", "s", "q"} if request["faq_id"] else {"e", "r", "s", "q"}
    menu = "[a]pprove, " if request["faq_id"] else ""
    choice = ask("\nDecision: {}[e]dit, [r]eject, [s]kip, [q]uit: ".format(menu), choices)
    if choice in ("s", "q"):
        return choice
    corrected = response = None
    if choice == "e":
        while True:
            corrected = ask("Corrected intent (Enter to keep '{}'): ".format(
                request["predicted_intent"])) or None
            if corrected is None or corrected in labels:
                break
            print("'{}' is not in the label mapping; check the spelling.".format(corrected))
        prompt = "Final response to send"
        if request["faq_answer"]:
            prompt += " (Enter to send the FAQ answer above)"
        while not response:
            response = ask(prompt + ": ") or request["faq_answer"]
    elif choice == "r":
        response = ask("Message to customer (optional, Enter for none): ") or None
    notes = ask("Reviewer notes: ") or None
    decision = {"a": "approved", "e": "edited", "r": "rejected"}[choice]
    shown = "the FAQ answer above" if decision == "approved" else (response or "-")
    print("\nAbout to save: {} | corrected intent: {} | response: {} | notes: {}".format(
        decision, corrected or "-", shown, notes or "-"))
    if ask("Save this decision? [y/n]: ", {"y", "n"}) == "n":
        return "s"
    return decision, response, corrected, notes


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=os.environ.get("BANKASSIST_DB_PATH", str(database.DB_PATH)),
                        help="SQLite database path")
    args = parser.parse_args(argv)
    database.initialize_database(args.db)
    labels = set(json.loads(MAPPING_PATH.read_text(encoding="utf-8")))
    pending = database.get_pending_reviews(args.db)
    if not pending:
        print("No requests are waiting for review.")
        return 0
    saved = 0
    for position, request in enumerate(pending, start=1):
        show(request, position, len(pending))
        result = review(request, labels)
        if result == "q":
            break
        if result == "s":
            print("Skipped; it stays pending.")
            continue
        decision, response, corrected, notes = result
        try:
            database.save_review(request["request_id"], decision, response, corrected,
                                 notes, args.db)
        except ValueError as error:
            print("Not saved: {}".format(error))
            continue
        saved += 1
        print("Saved request #{} as {}.".format(request["request_id"], decision))
    remaining = len(database.get_pending_reviews(args.db))
    print("\nSaved {} decision(s); {} request(s) still waiting.".format(saved, remaining))
    return 0


if __name__ == "__main__":
    sys.exit(main())
