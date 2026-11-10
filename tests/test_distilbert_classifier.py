import tempfile
import unittest
from pathlib import Path

import numpy as np

import distilbert_classifier as dc

try:
    import torch  # noqa: F401
    import transformers  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False


class PredictionFormatTests(unittest.TestCase):
    def test_shared_prediction_format(self):
        result = dc.prediction_from_probabilities("hello", [0.1, 0.6, 0.3], ["a", "b", "c"], 2.5)
        self.assertEqual(set(result), {"query", "predicted_intent", "confidence_score",
                                       "top_intents", "model_name", "model_runtime_ms"})
        self.assertEqual(result["predicted_intent"], "b")
        self.assertAlmostEqual(result["confidence_score"], 0.6)
        self.assertEqual([name for name, _ in result["top_intents"]], ["b", "c", "a"])
        self.assertEqual(result["model_runtime_ms"], 2.5)

    def test_top_k_limits_length(self):
        result = dc.prediction_from_probabilities("q", [0.4, 0.3, 0.2, 0.1], list("abcd"), 1, top_k=2)
        self.assertEqual(len(result["top_intents"]), 2)

    def test_wrong_number_of_probabilities_is_rejected(self):
        with self.assertRaises(ValueError):
            dc.prediction_from_probabilities("q", [0.5, 0.5], ["a", "b", "c"], 1)

    def test_softmax_rows_sum_to_one(self):
        probs = dc.softmax(np.array([[1000.0, 1001.0, 999.0], [0.0, 0.0, 0.0]]))
        np.testing.assert_allclose(probs.sum(axis=1), [1.0, 1.0])
        self.assertTrue(np.all(np.isfinite(probs)))


class StratifiedSubsetTests(unittest.TestCase):
    def test_takes_up_to_n_per_label_and_is_reproducible(self):
        labels = ["a"] * 10 + ["b"] * 3 + ["c"] * 6
        first = dc.stratified_subset(labels, 4, seed=1)
        self.assertEqual(first, dc.stratified_subset(labels, 4, seed=1))
        picked = [labels[i] for i in first]
        self.assertEqual((picked.count("a"), picked.count("b"), picked.count("c")), (4, 3, 4))

    def test_pad_batch(self):
        ids, mask = dc._pad_batch([[5, 6, 7], [8]])
        self.assertEqual(ids.tolist(), [[5, 6, 7], [8, 0, 0]])
        self.assertEqual(mask.tolist(), [[1, 1, 1], [1, 0, 0]])


@unittest.skipUnless(HAVE_TORCH, "torch and transformers are not installed")
class SmokeTrainingTests(unittest.TestCase):
    def test_smoke_run_writes_shared_format_predictions(self):
        import csv
        import json
        import train_distilbert

        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            code = train_distilbert.main(["--smoke-test", "--epochs", "1", "--batch-size", "64",
                                          "--results-dir", str(out), "--report-path", str(out / "r.md")])
            self.assertEqual(code, 0)
            with (out / "validation_predictions.csv").open(encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(list(rows[0]), ["record_id", "query", "true_intent", "predicted_intent",
                                             "confidence_score", "top_intents", "model_name"])
            self.assertEqual(len(json.loads(rows[0]["top_intents"])), 3)
            self.assertTrue(0 <= float(rows[0]["confidence_score"]) <= 1)
            self.assertIn("SMOKE TEST", (out / "r.md").read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
