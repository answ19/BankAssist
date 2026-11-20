"""Check the eligibility replay in coverage_analysis (no plotting needed)."""

import tempfile
import unittest
from pathlib import Path

import coverage_analysis


class CoverageAnalysisTests(unittest.TestCase):
    def test_eligibility_applies_threshold_then_faq(self):
        true = ["card_arrival", "card_arrival", "top_up_failed", "lost_or_stolen_card"]
        predicted = ["card_arrival", "declined_card_payment", "top_up_failed",
                     "lost_or_stolen_card"]
        scores = [0.95, 0.92, 0.97, 0.85]
        result = coverage_analysis.eligibility(
            true, predicted, scores, ["card_arrival", "declined_card_payment",
                                      "lost_or_stolen_card"], 0.90)
        confident = result["confidence_only"]
        self.assertEqual((confident["eligible"], confident["correct"], confident["errors"]),
                         (3, 2, 1))
        gated = result["faq_gated"]  # top_up_failed has no approved FAQ; 0.85 is too low
        self.assertEqual((gated["eligible"], gated["correct"], gated["errors"]), (2, 1, 1))
        self.assertEqual(gated["coverage"], 0.5)
        self.assertEqual(result["faq_gated_errors"], [1])

    def test_empty_subset_has_no_accuracy(self):
        result = coverage_analysis.eligibility(["a"], ["a"], [0.5], ["a"], 0.90)
        self.assertIsNone(result["confidence_only"]["accuracy"])
        self.assertEqual(result["faq_gated"]["eligible"], 0)

    def test_faq_snapshot_reads_approvals(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "faq.csv"
            path.write_text("intent,question,answer,source,approved\n"
                            "card_arrival,Q,A,team,1\nterminate_account,Q,A,team,0\n",
                            encoding="utf-8")
            snapshot = coverage_analysis.read_faq_snapshot(path)
        self.assertEqual(snapshot["approved"], ["card_arrival"])
        self.assertEqual(snapshot["total"], 2)
        self.assertEqual(len(snapshot["sha256"]), 12)


if __name__ == "__main__":
    unittest.main()
