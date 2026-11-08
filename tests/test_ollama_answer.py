"""Check the optional Ollama wording step with a fake Ollama (no real model needed)."""

import json
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from unittest import mock

import database
import ollama_answer
import routing

FAQ = {"intent": "card_arrival", "question": "Where is my new card?",
       "answer": "Your card is posted after your order is confirmed. Track it in the app."}


class FakeOllamaServer:
    """A tiny local web server that answers /api/chat like Ollama does."""

    def __init__(self, reply):
        outer = self
        self.reply, self.requests = reply, []

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers["Content-Length"])
                outer.requests.append(json.loads(self.rfile.read(length)))
                body = json.dumps({"message": {"content": outer.reply}}).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(body)

            def log_message(self, *args):
                pass

        self.server = HTTPServer(("127.0.0.1", 0), Handler)
        self.url = "http://127.0.0.1:{}".format(self.server.server_port)
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class WriteAnswerTests(unittest.TestCase):
    def ask(self, reply, query="where is my card"):
        fake = FakeOllamaServer(reply)
        self.addCleanup(fake.close)
        return ollama_answer.write_answer(query, FAQ, url=fake.url), fake

    def test_good_reply_is_used_and_timed(self):
        result, fake = self.ask("Your card is on its way. You can track it in the app.")
        self.assertEqual(result["status"], "ok")
        self.assertGreaterEqual(result["runtime_ms"], 0)
        sent = fake.requests[0]
        self.assertEqual(sent["options"]["temperature"], 0)
        self.assertFalse(sent["stream"])
        self.assertIn("Approved answer: " + FAQ["answer"], sent["messages"][1]["content"])

    def test_needs_review_reply_is_not_sent(self):
        for reply in ("NEEDS_REVIEW", "needs_review.", ""):
            with self.subTest(reply=reply):
                result, _ = self.ask(reply)
                self.assertEqual(result["status"], "needs_review")
                self.assertIsNone(result["answer"])

    def test_made_up_number_is_caught(self):
        result, _ = self.ask("Your card arrives in 5 days.")
        self.assertEqual(result["status"], "needs_review")
        self.assertIn("5", result["reason"])

    def test_review_request_spelled_differently_is_not_sent(self):
        for reply in ("NEEDS REVIEW", "Needs-review.", "needs  review"):
            with self.subTest(reply=reply):
                result, _ = self.ask(reply)
                self.assertEqual(result["status"], "needs_review")
                self.assertIsNone(result["answer"])

    def test_made_up_amount_in_words_is_caught(self):
        for reply in ("A replacement costs five dollars.", "Your card arrives within a week.",
                      "Replacements are free."):
            with self.subTest(reply=reply):
                result, _ = self.ask(reply)
                self.assertEqual(result["status"], "needs_review")
                self.assertIn("words not found", result["reason"])

    def test_words_from_the_faq_are_allowed(self):
        faq = dict(FAQ, answer="Freeze your card immediately. Replacements usually take a few days.")
        fake = FakeOllamaServer("Please freeze your card immediately; a new one takes a few days.")
        self.addCleanup(fake.close)
        self.assertEqual(ollama_answer.write_answer("lost card", faq, url=fake.url)["status"], "ok")

    def test_card_number_is_never_repeated(self):
        result, _ = self.ask("Thanks, I checked card 4111111111111111 for you.")
        self.assertEqual(result["status"], "needs_review")

    def test_numbers_from_the_faq_are_allowed(self):
        faq = dict(FAQ, answer="Cards arrive in 7 to 10 working days.")
        fake = FakeOllamaServer("It usually takes 7 to 10 working days.")
        self.addCleanup(fake.close)
        self.assertEqual(ollama_answer.write_answer("how long", faq, url=fake.url)["status"], "ok")

    def test_unreachable_ollama_is_reported_not_crashed(self):
        result = ollama_answer.write_answer("hi", FAQ, url="http://127.0.0.1:9")
        self.assertEqual(result["status"], "unavailable")
        self.assertIsNone(result["runtime_ms"])

    def test_topic_names_are_cleaned(self):
        self.assertEqual(ollama_answer.readable_topic("Refund_not_showing_up"), "refund not showing up")
        self.assertEqual(ollama_answer.readable_topic("reverted_card_payment?"), "reverted card payment")


class RoutingWithOllamaTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "test.db"
        database.initialize_database(self.db)
        database.add_faq("card_arrival", FAQ["question"], FAQ["answer"], "team", True, self.db)

    def tearDown(self):
        self.directory.cleanup()

    def route(self, score, written, **options):
        with mock.patch("ollama_answer.write_answer", return_value=written) as patched:
            result = routing.route_query("Where is my card?", "card_arrival", score,
                                         threshold=0.90, db_path=self.db, **options)
        return result, patched

    def test_off_by_default_ollama_is_never_called(self):
        result, patched = self.route(0.95, {})
        self.assertEqual(result["system_response"], FAQ["answer"])
        patched.assert_not_called()

    def test_high_confidence_uses_ollama_wording_and_saves_runtime(self):
        written = {"status": "ok", "answer": "Friendly wording.", "reason": "ok", "runtime_ms": 3100.0}
        result, _ = self.route(0.95, written, use_ollama=True)
        self.assertEqual((result["status"], result["system_response"]), ("answered", "Friendly wording."))
        saved = database.get_evaluation_records(True, self.db)[0]
        self.assertEqual(saved["ollama_runtime_ms"], 3100.0)

    def test_needs_review_from_model_sends_to_human(self):
        written = {"status": "needs_review", "answer": None, "reason": "model asked for review", "runtime_ms": 900.0}
        result, _ = self.route(0.99, written, use_ollama=True)
        self.assertEqual(result["status"], "pending_review")
        self.assertEqual(result["system_response"], routing.ESCALATION_MESSAGE)
        self.assertEqual(result["ollama_note"], "model asked for review")

    def test_ollama_down_falls_back_to_approved_faq_text(self):
        written = {"status": "unavailable", "answer": None, "reason": "Ollama unavailable: x", "runtime_ms": None}
        result, _ = self.route(0.95, written, use_ollama=True)
        self.assertEqual((result["status"], result["system_response"]), ("answered", FAQ["answer"]))

    def test_low_confidence_never_calls_ollama(self):
        result, patched = self.route(0.82, {}, use_ollama=True)
        self.assertEqual(result["status"], "pending_review")
        patched.assert_not_called()

    def test_no_approved_faq_never_calls_ollama(self):
        with mock.patch("ollama_answer.write_answer") as patched:
            result = routing.route_query("q", "change_pin", 0.99, threshold=0.90,
                                         db_path=self.db, use_ollama=True)
        self.assertEqual(result["status"], "pending_review")
        patched.assert_not_called()

    def test_escalation_reason_and_draft_are_saved_for_reviewer(self):
        written = {"status": "needs_review", "answer": "It takes 5 days.",
                   "reason": "numbers not found in the FAQ: ['5']", "runtime_ms": 1.0}
        result, _ = self.route(0.99, written, use_ollama=True)
        pending = database.get_pending_reviews(self.db)[0]
        self.assertEqual(pending["request_id"], result["request_id"])
        self.assertIn("numbers not found", pending["review_reason"])
        self.assertEqual(pending["ollama_draft"], "It takes 5 days.")

    def test_low_confidence_reason_is_saved(self):
        self.route(0.82, {}, use_ollama=True)
        reason = database.get_pending_reviews(self.db)[0]["review_reason"]
        self.assertEqual(reason, "confidence 0.82 below threshold 0.90")

    def test_old_database_gains_new_columns(self):
        import sqlite3
        old = Path(self.directory.name) / "old.db"
        with sqlite3.connect(str(old)) as connection:
            connection.execute("CREATE TABLE requests (request_id INTEGER PRIMARY KEY, "
                               "query TEXT, status TEXT, created_at TEXT)")
        connection.close()
        database.initialize_database(old)
        with sqlite3.connect(str(old)) as connection:
            columns = {row[1] for row in connection.execute("PRAGMA table_info(requests)")}
        connection.close()
        self.assertTrue({"review_reason", "ollama_draft", "ollama_runtime_ms"} <= columns)

    def test_reviewer_approval_still_works_after_ollama_escalation(self):
        written = {"status": "needs_review", "answer": None, "reason": "r", "runtime_ms": 1.0}
        result, _ = self.route(0.99, written, use_ollama=True)
        database.save_review(result["request_id"], "approved", db_path=self.db)
        saved = database.get_evaluation_records(True, self.db)[0]
        self.assertEqual((saved["status"], saved["final_response"]), ("approved", FAQ["answer"]))


if __name__ == "__main__":
    unittest.main()
