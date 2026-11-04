"""Create or update bankassist.db from the FAQ file.

The FAQ CSV is the source of truth. Only rows with approved = 1 are ever shown to
customers, so a team member must read each draft and set approved to 1 before
BankAssist can answer that intent automatically. Rerunning after editing the CSV
updates FAQs in place (matched by intent and question) and keeps logged requests
and reviews. --reset deletes the database first, including those records.

    python setup_database.py           # create, or update FAQs in place
    python setup_database.py --reset   # rebuild from scratch
"""

import argparse
import sys
from pathlib import Path

import classifier
import database


ROOT = Path(__file__).resolve().parent
FAQ_PATH = ROOT / "data" / "faq" / "faq_drafts.csv"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--db", default=str(database.DB_PATH), help="SQLite database path")
    parser.add_argument("--faqs", default=str(FAQ_PATH), help="FAQ CSV path")
    parser.add_argument("--reset", action="store_true",
                        help="delete the database first (also deletes logged requests and reviews)")
    args = parser.parse_args(argv)
    db_path = Path(args.db)
    if args.reset and db_path.exists():
        db_path.unlink()
    database.initialize_database(db_path)
    labels = classifier.read_label_mapping()
    try:
        ids = database.load_faqs_from_csv(args.faqs, db_path, known_intents=labels)
    except (OSError, ValueError) as error:
        print("FAQ load failed; nothing was written: {}".format(error), file=sys.stderr)
        return 1
    approved = sorted(intent for intent in labels
                      if database.get_faq_by_intent(intent, db_path) is not None)
    print("Loaded {} FAQs into {}; {} of {} intents have an approved FAQ.".format(
        len(ids), db_path, len(approved), len(labels)))
    if approved:
        print("Approved: {}".format(", ".join(approved)))
    if len(approved) < len(labels):
        print("Intents without an approved FAQ always go to human review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
