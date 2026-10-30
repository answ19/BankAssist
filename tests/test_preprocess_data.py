"""Check the input guards and leakage checks without using the official test set."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import preprocess_data as preprocessing


class PreprocessingTests(unittest.TestCase):
    def test_first_source_ids_and_original_values_are_preserved(self):
        rows = [
            {"text": "  Pay\t my bill  ", "category": "bill"},
            {"text": "Pay my bill", "category": "bill"},
            {"text": "Balance?", "category": "balance"},
        ]
        original, working, removed = preprocessing.prepare_records(rows)
        self.assertEqual([row["record_id"] for row in original], [1, 2, 3])
        self.assertEqual([row["record_id"] for row in working], [1, 3])
        self.assertEqual([row["record_id"] for row in removed], [2])
        self.assertEqual(working[0]["text"], rows[0]["text"])
        self.assertEqual(working[0]["text_clean"], "Pay my bill")
        self.assertNotIn("record_id", rows[0])

    def test_conflicting_labels_stop_before_first_copy_is_selected(self):
        rows = [{"text": "Pay bill", "category": "a"},
                {"text": " Pay  bill ", "category": "b"}]
        with self.assertRaisesRegex(ValueError, "Conflicting labels") as caught:
            preprocessing.prepare_records(rows)
        self.assertIn('"record_id": 1', str(caught.exception))
        self.assertIn('"record_id": 2', str(caught.exception))
        self.assertIn('"category": "b"', str(caught.exception))

    def test_missing_and_blank_inputs_stop_processing(self):
        for field in ("text", "category"):
            for value in (None, "", " \t "):
                with self.subTest(field=field, value=value):
                    row = {"text": "Question?", "category": "intent"}
                    row[field] = value
                    with self.assertRaisesRegex(ValueError, "source record IDs: 1"):
                        preprocessing.prepare_records([row])

    def test_case_and_punctuation_remain_distinct(self):
        _, working, _ = preprocessing.prepare_records([
            {"text": text, "category": "intent"}
            for text in ("Pay bill", "pay bill", "Pay bill?")])
        self.assertEqual(len(working), 3)

    def test_split_checks_detect_id_and_question_leakage_and_missing_intents(self):
        _, rows, _ = preprocessing.prepare_records([
            {"text": str(index), "category": "a" if index % 2 else "b"}
            for index in range(4)])
        mapping = {"a": 0, "b": 1}
        preprocessing.validate_splits(rows, rows[:2], rows[2:], mapping)
        with self.assertRaisesRegex(ValueError, "Record IDs overlap"):
            preprocessing.validate_splits(rows, rows[:2], [rows[0]] + rows[2:], mapping)
        duplicate = dict(rows[2], text=rows[0]["text"], text_clean=rows[0]["text_clean"])
        with self.assertRaisesRegex(ValueError, "Normalized questions overlap"):
            preprocessing.validate_splits([rows[0], rows[1], duplicate, rows[3]],
                                          rows[:2], [duplicate, rows[3]], mapping)
        with self.assertRaisesRegex(ValueError, "does not contain every intent"):
            preprocessing.validate_splits(rows, rows[::2], rows[1::2], mapping)

    def test_invalid_inputs_do_not_replace_existing_outputs(self):
        cases = [
            ("text,category\nHello,a\n Hello ,b\n", "Conflicting labels"),
            ("text,category\nHello,a\n", "Working count discrepancy"),
        ]
        for source, expected_error in cases:
            with self.subTest(error=expected_error), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                raw = root / "train.csv"
                raw.write_text(source, encoding="utf-8")
                outputs = root / "processed"
                outputs.mkdir()
                sentinel = outputs / "train.csv"
                sentinel.write_text("existing output", encoding="utf-8")
                report = root / "summary.md"
                errors = io.StringIO()
                with patch.multiple(preprocessing, INPUT_PATH=raw, OUTPUT_DIR=outputs,
                                    REPORT_PATH=report), contextlib.redirect_stderr(errors):
                    self.assertEqual(preprocessing.main(), 1)
                self.assertIn(expected_error, errors.getvalue())
                self.assertEqual(raw.read_text(encoding="utf-8"), source)
                self.assertEqual(sentinel.read_text(encoding="utf-8"), "existing output")
                self.assertFalse((outputs / "validation.csv").exists())
                self.assertFalse(report.exists())


if __name__ == "__main__":
    unittest.main()
