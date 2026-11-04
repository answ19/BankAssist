"""Create bankassist.db and load the FAQ file.

The FAQ CSV is the source of truth. Only rows with approved = 1 are ever shown to
customers, so a team member must read each draft and set approved to 1 before
BankAssist can answer that intent automatically. Rerun with --reset after editing.

    python setup_database.py --reset
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
    if db_path.exists():
        if not args.reset:
            print("{} exists; rerun with --reset to rebuild it.".format(db_path), file=sys.stderr)
            return 1
        db_path.unlink()
    database.initialize_database(db_path)
    try:
        ids = database.load_faqs_from_csv(args.faqs, db_path,
                                          known_intents=classifier.read_label_mapping())
    except (OSError, ValueError) as error:
        db_path.unlink()
        print("FAQ load failed: {}".format(error), file=sys.stderr)
        return 1
    approved = sum(database.get_faq_by_intent(intent, db_path) is not None
                   for intent in classifier.read_label_mapping())
    print("Loaded {} FAQs into {}; {} of 77 intents have an approved FAQ.".format(
        len(ids), db_path, approved))
    if approved < 77:
        print("Intents without an approved FAQ always go to human review.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
