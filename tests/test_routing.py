"""Check confidence routing with hand-entered test scores (not model output)."""

import tempfile
import unittest
from pathlib import Path

import database
import routing


class RoutingTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "test.db"
        database.initialize_database(self.db)
        self.faq_id = database.add_faq("card_arrival", "Q", "Approved answer", "team",
                                       True, self.db)

    def tearDown(self):
        self.directory.cleanup()

    def route(self, score, intent="card_arrival", **options):
        return routing.route_query("Where is my card?", intent, score, threshold=0.90,
                                   is_test_case=True, db_path=self.db, **options)

    def test_handoff_threshold_cases(self):
        for score, status in ((0.95, "answered"), (0.90, "answered"),
                              (0.82, "pending_review"), (0.65, "pending_review")):
            with self.subTest(score=score):
                result = self.route(score)
                self.assertEqual(result["status"], status)
                self.assertEqual(result["faq_id"], self.faq_id)
                self.assertEqual(result["faq_answer"], "Approved answer")
                expected = ("Approved answer" if status == "answered"
                            else routing.ESCALATION_MESSAGE)
                self.assertEqual(result["system_response"], expected)

    def test_every_routed_request_is_saved(self):
        result = self.route(0.82, true_intent="card_arrival", model_name="test",
                            model_runtime_ms=12.5)
        saved = database.get_evaluation_records(True, self.db)
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["request_id"], result["request_id"])
        self.assertEqual(saved[0]["status"], "pending_review")
        self.assertEqual(saved[0]["model_runtime_ms"], 12.5)
        self.assertEqual(saved[0]["is_test_case"], 1)

    def test_high_confidence_without_approved_faq_is_escalated(self):
        result = self.route(0.99, intent="terminate_account")
        self.assertEqual(result["status"], "pending_review")
        self.assertIsNone(result["faq_id"])
        self.assertIsNone(result["faq_answer"])
        self.assertEqual(self.route(0.99, intent=None)["status"], "pending_review")

    def test_threshold_is_configurable(self):
        result = routing.route_query("Where is my card?", "card_arrival", 0.82,
                                     threshold=0.80, db_path=self.db)
        self.assertEqual(result["status"], "answered")

    def test_invalid_scores_are_rejected_without_saving(self):
        for score in (1.2, -0.1, float("nan"), "0.95", True, None):
            with self.subTest(score=score), self.assertRaises(ValueError):
                self.route(score)
        with self.assertRaises(ValueError):
            routing.route_query("Q", "card_arrival", 0.5, threshold=2, db_path=self.db)
        self.assertEqual(database.get_evaluation_records(True, self.db), [])


if __name__ == "__main__":
    unittest.main()
