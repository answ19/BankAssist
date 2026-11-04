"""Check SQLite storage of FAQs, requests, and review decisions."""

import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path

import database


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "test.db"
        database.initialize_database(self.db)

    def tearDown(self):
        self.directory.cleanup()

    def pending_request(self, faq_id=None):
        return database.save_request("Where is my card?", "card_arrival", 0.5, faq_id,
                                     "escalated", "pending_review", db_path=self.db)

    def test_initialize_is_repeatable(self):
        database.initialize_database(self.db)
        with closing(sqlite3.connect(str(self.db))) as connection:
            tables = {row[0] for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'")}
        self.assertTrue({"faqs", "requests", "reviews"} <= tables)

    def test_only_approved_faqs_are_retrieved(self):
        database.add_faq("card_arrival", "Q draft", "A draft", "draft", False, self.db)
        self.assertIsNone(database.get_faq_by_intent("card_arrival", self.db))
        first = database.add_faq("card_arrival", "Q1", "A1", "team", True, self.db)
        database.add_faq("card_arrival", "Q2", "A2", "team", True, self.db)
        self.assertEqual(database.get_faq_by_intent("card_arrival", self.db)["faq_id"], first)

    def test_unusual_intent_names_are_stored_as_data(self):
        for intent in ("reverted_card_payment?", "Refund_not_showing_up", "x'; DROP TABLE faqs; --"):
            with self.subTest(intent=intent):
                database.add_faq(intent, "Q", "A", "team", True, self.db)
                self.assertEqual(database.get_faq_by_intent(intent, self.db)["intent"], intent)

    def test_blank_fields_and_bad_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "answer must not be blank"):
            database.add_faq("card_arrival", "Q", "  ", "team", True, self.db)
        with self.assertRaisesRegex(ValueError, "Unknown request status"):
            database.save_request("Q", "a", 0.5, None, None, "done", db_path=self.db)
        with self.assertRaises(sqlite3.IntegrityError):
            database.save_request("Q", "a", 1.5, None, None, "answered", db_path=self.db)

    def test_load_faqs_from_csv_checks_intents_and_flags(self):
        path = Path(self.directory.name) / "faqs.csv"
        path.write_text("intent,question,answer,source,approved\n"
                        "card_arrival,Q,A,team,yes\nbad_intent,Q,A,team,1\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "bad_intent"):
            database.load_faqs_from_csv(path, self.db, known_intents={"card_arrival"})
        path.write_text("intent,question,answer,source,approved\n"
                        "card_arrival,Q,A,team,maybe\n", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "invalid approved value"):
            database.load_faqs_from_csv(path, self.db)
        path.write_text("intent,question,answer,source,approved\n"
                        "card_arrival,Q,A,team,1\n", encoding="utf-8")
        self.assertEqual(len(database.load_faqs_from_csv(path, self.db)), 1)

    def test_pending_reviews_include_faq_evidence(self):
        faq_id = database.add_faq("card_arrival", "FAQ Q", "FAQ A", "team", True, self.db)
        request_id = self.pending_request(faq_id)
        database.save_request("Q", "a", 0.99, None, "A", "answered", db_path=self.db)
        pending = database.get_pending_reviews(self.db)
        self.assertEqual([row["request_id"] for row in pending], [request_id])
        self.assertEqual((pending[0]["faq_question"], pending[0]["faq_answer"]),
                         ("FAQ Q", "FAQ A"))

    def test_review_decisions_update_request_status(self):
        faq_id = database.add_faq("card_arrival", "FAQ Q", "FAQ A", "team", True, self.db)
        approved = self.pending_request(faq_id)
        edited = self.pending_request(faq_id)
        rejected = self.pending_request()
        database.save_review(approved, "approved", "ignored", db_path=self.db)
        database.save_review(edited, "edited", "New answer", "card_delivery_estimate",
                             "wrong intent", self.db)
        database.save_review(rejected, "rejected", reviewer_notes="out of scope", db_path=self.db)
        records = {r["request_id"]: r for r in database.get_evaluation_records(db_path=self.db)}
        self.assertEqual(records[approved]["status"], "approved")
        self.assertEqual(records[approved]["final_response"], "FAQ A")
        self.assertEqual(records[edited]["status"], "edited")
        self.assertEqual(records[edited]["corrected_intent"], "card_delivery_estimate")
        self.assertEqual(records[rejected]["status"], "rejected")
        self.assertIsNone(records[rejected]["final_response"])
        self.assertEqual(database.get_pending_reviews(self.db), [])

    def test_invalid_reviews_leave_the_request_pending(self):
        no_faq = self.pending_request()
        with self.assertRaisesRegex(ValueError, "no FAQ answer to approve"):
            database.save_review(no_faq, "approved", db_path=self.db)
        with self.assertRaisesRegex(ValueError, "needs a final response"):
            database.save_review(no_faq, "edited", " ", db_path=self.db)
        with self.assertRaisesRegex(ValueError, "Unknown review decision"):
            database.save_review(no_faq, "maybe", db_path=self.db)
        with self.assertRaisesRegex(ValueError, "does not exist"):
            database.save_review(999, "rejected", db_path=self.db)
        self.assertEqual(len(database.get_pending_reviews(self.db)), 1)
        database.save_review(no_faq, "rejected", db_path=self.db)
        with self.assertRaisesRegex(ValueError, "not pending_review"):
            database.save_review(no_faq, "rejected", db_path=self.db)

    def test_test_cases_are_excluded_from_evaluation_by_default(self):
        database.save_request("real", "a", 0.9, None, None, "answered", db_path=self.db)
        database.save_request("test", "a", 0.9, None, None, "answered",
                              is_test_case=True, db_path=self.db)
        self.assertEqual([r["query"] for r in database.get_evaluation_records(db_path=self.db)],
                         ["real"])
        self.assertEqual(len(database.get_evaluation_records(True, self.db)), 2)


if __name__ == "__main__":
    unittest.main()
