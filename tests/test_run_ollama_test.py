"""Smoke-test run_ollama_test.py with a fake Ollama server."""

import csv
import tempfile
import unittest
from pathlib import Path

import run_ollama_test
from test_ollama_answer import FakeOllamaServer


class RunOllamaTestTests(unittest.TestCase):
    def run_with_reply(self, reply):
        fake = FakeOllamaServer(reply)
        self.addCleanup(fake.close)
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        code = run_ollama_test.main(["--url", fake.url, "--out-dir", folder.name])
        return code, Path(folder.name)

    def test_writes_both_reports_and_scores_every_question(self):
        code, out = self.run_with_reply("Your card is on its way. You can check the app.")
        self.assertEqual(code, 0)
        with (out / "results" / "ollama_answer_test.csv").open(encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        self.assertEqual(len(rows), 20)
        self.assertTrue(all(r["status"] == "ok" for r in rows))
        # a model that always answers must fail the "needs_review" questions
        failed = [r["id"] for r in rows if r["matches_expected"] == "False"]
        self.assertEqual(sorted(failed, key=int), ["6", "9", "14", "17", "18", "20"])
        report = (out / "docs" / "ollama_answer_test.md").read_text(encoding="utf-8")
        self.assertIn("Accurate, nothing added?", report)

    def test_repeated_card_number_is_flagged(self):
        self.assertTrue(run_ollama_test.repeats_query_digits(
            "my card is 4111 1111 1111 1111", "Card 4111111111111111 is fine"))
        self.assertFalse(run_ollama_test.repeats_query_digits("hi 12", "ok 12"))

    def test_unreachable_ollama_stops_with_error_code(self):
        folder = tempfile.TemporaryDirectory()
        self.addCleanup(folder.cleanup)
        self.assertEqual(run_ollama_test.main(["--url", "http://127.0.0.1:9",
                                               "--out-dir", folder.name]), 1)


if __name__ == "__main__":
    unittest.main()
