"""Check evaluation metrics against hand-computed values."""

import unittest

import evaluation


def record(status, decision=None, predicted="a", corrected=None, true=None, runtime=None):
    return {"status": status, "decision": decision, "predicted_intent": predicted,
            "corrected_intent": corrected, "true_intent": true, "model_runtime_ms": runtime}


class EvaluationTests(unittest.TestCase):
    def test_classification_metrics_match_hand_calculation(self):
        # a: P=2/3, R=1, F1=0.8 (support 2); b: P=1, R=1/2, F1=2/3 (support 2)
        result = evaluation.classification_metrics(["a", "a", "b", "b"], ["a", "a", "a", "b"])
        self.assertAlmostEqual(result["accuracy"], 0.75)
        self.assertAlmostEqual(result["per_class"]["a"]["f1"], 0.8)
        self.assertAlmostEqual(result["per_class"]["b"]["f1"], 2 / 3)
        self.assertAlmostEqual(result["macro_f1"], (0.8 + 2 / 3) / 2)
        self.assertAlmostEqual(result["weighted_f1"], (0.8 + 2 / 3) / 2)
        self.assertEqual(result["confusion_matrix"], [[2, 0], [1, 1]])

    def test_label_list_includes_unseen_intents(self):
        result = evaluation.classification_metrics(["a"], ["a"], labels=["a", "b"])
        self.assertEqual(result["per_class"]["b"]["f1"], 0.0)
        self.assertAlmostEqual(result["macro_f1"], 0.5)
        with self.assertRaisesRegex(ValueError, "missing from the label list"):
            evaluation.classification_metrics(["c"], ["a"], labels=["a"])

    def test_empty_inputs_are_pending_not_zero(self):
        self.assertIsNone(evaluation.classification_metrics([], []))
        self.assertIsNone(evaluation.recall_at_k([], [], 1))
        self.assertIsNone(evaluation.runtime_summary([None]))
        table = evaluation.format_evaluation_table(evaluation.evaluate_records([]))
        self.assertIn("| Accuracy | Pending |", table)
        self.assertIn("| Escalation rate | Pending |", table)

    def test_recall_at_k(self):
        ranked = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        self.assertAlmostEqual(evaluation.recall_at_k([1, 5, 10], ranked, 1), 1 / 3)
        self.assertAlmostEqual(evaluation.recall_at_k([1, 5, 10], ranked, 3), 2 / 3)

    def test_review_metrics(self):
        records = [
            record("answered"),
            record("pending_review"),
            record("approved", "approved"),
            record("edited", "edited", corrected="b"),
            record("edited", "edited"),
            record("rejected", "rejected"),
        ]
        result = evaluation.review_metrics(records)
        self.assertAlmostEqual(result["automatic_answer_rate"], 1 / 6)
        self.assertAlmostEqual(result["escalation_rate"], 5 / 6)
        self.assertEqual(result["still_pending"], 1)
        self.assertEqual(result["reviews"], 4)
        self.assertAlmostEqual(result["approval_rate"], 1 / 4)
        self.assertAlmostEqual(result["edit_rate"], 2 / 4)
        self.assertAlmostEqual(result["rejection_rate"], 1 / 4)
        # The plain rejection gives no intent judgement, leaving 2 of 3 agreeing.
        self.assertEqual(result["agreement_reviews"], 3)
        self.assertAlmostEqual(result["reviewer_agreement"], 2 / 3)

    def test_evaluate_records_uses_only_labelled_requests(self):
        records = [record("answered", true="a", runtime=10.0),
                   record("answered", predicted="b", true="a", runtime=30.0),
                   record("pending_review")]
        result = evaluation.evaluate_records(records)
        self.assertEqual(result["classification"]["count"], 2)
        self.assertAlmostEqual(result["classification"]["accuracy"], 0.5)
        self.assertEqual(result["model_runtime"]["mean_ms"], 20.0)


if __name__ == "__main__":
    unittest.main()
