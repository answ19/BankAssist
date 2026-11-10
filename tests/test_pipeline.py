"""Check the shared prediction format and the classify -> route -> log flow."""

import tempfile
import unittest
from pathlib import Path

import classifier
import database
import pipeline


def toy_model():
    texts = ["where is my card", "card not arrived yet", "my card never came",
             "card was stolen", "I lost my card", "someone stole my card"]
    labels = ["card_arrival"] * 3 + ["lost_or_stolen_card"] * 3
    return classifier.build_pipeline().fit(texts, labels)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.db = Path(self.directory.name) / "test.db"
        database.initialize_database(self.db)
        self.model = toy_model()

    def tearDown(self):
        self.directory.cleanup()

    def test_prediction_output_format(self):
        result = classifier.predict("  my card was   stolen ", self.model, top_k=2)
        self.assertEqual(set(result), {"query", "predicted_intent", "confidence_score",
                                       "top_intents", "model_name", "model_runtime_ms"})
        self.assertEqual(result["predicted_intent"], "lost_or_stolen_card")
        self.assertEqual(result["top_intents"][0],
                         {"intent": result["predicted_intent"],
                          "probability": result["confidence_score"]})
        self.assertEqual(len(result["top_intents"]), 2)
        self.assertGreaterEqual(result["top_intents"][0]["probability"],
                                result["top_intents"][1]["probability"])
        self.assertTrue(0 <= result["confidence_score"] <= 1)
        self.assertGreaterEqual(result["model_runtime_ms"], 0)

    def test_answer_query_logs_model_output(self):
        database.add_faq("lost_or_stolen_card", "Q", "Freeze it.", "team", True, self.db)
        answered = pipeline.answer_query("my card was stolen", self.model, threshold=0.0,
                                         true_intent="lost_or_stolen_card", db_path=self.db)
        escalated = pipeline.answer_query("my card was stolen", self.model, threshold=1.0,
                                          db_path=self.db)
        self.assertEqual(answered["status"], "answered")
        self.assertEqual(answered["system_response"], "Freeze it.")
        self.assertEqual(escalated["status"], "pending_review")
        saved = database.get_evaluation_records(db_path=self.db)
        self.assertEqual([r["status"] for r in saved], ["answered", "pending_review"])
        self.assertEqual(saved[0]["model_name"], classifier.MODEL_NAME)
        self.assertEqual(saved[0]["true_intent"], "lost_or_stolen_card")
        self.assertIsNotNone(saved[0]["model_runtime_ms"])
        self.assertEqual(saved[0]["is_test_case"], 0)


if __name__ == "__main__":
    unittest.main()
