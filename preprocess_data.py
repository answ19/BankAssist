"""Deduplicate BANKING77 training records and save a shared stratified split."""

import csv
import hashlib
import json
import platform
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy
import scipy
import sklearn
from sklearn.model_selection import train_test_split

from inspect_data import duplicate_counts, is_blank, normalize_question, read_training_csv


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_DIR = ROOT / "data" / "processed"
REPORT_PATH = ROOT / "docs" / "preprocessing_summary.md"
EXPECTED_WORKING_COUNT = 9999
EXPECTED_INTENTS = 77
RANDOM_STATE = 42
VALIDATION_FRACTION = 0.2


def prepare_records(rows):
    """Assign source IDs, validate all labels, then keep each question's first row."""
    records = []
    question_records = defaultdict(list)
    invalid = []
    for position, row in enumerate(rows, start=1):
        record = dict(row, record_id=position)
        records.append(record)
        if is_blank(row["text"]) or is_blank(row["category"]):
            invalid.append(position)
        if not is_blank(row["text"]):
            record["text_clean"] = normalize_question(row["text"])
            question_records[record["text_clean"]].append(record)

    # Inspect every normalized group before choosing any retained records.
    conflicts = {
        question: group for question, group in question_records.items()
        if len({row["category"] for row in group
                if not is_blank(row["category"])}) > 1
    }
    if conflicts:
        details = ["Conflicting labels in {} normalized question group(s).".format(
            len(conflicts))]
        for question, group in sorted(conflicts.items()):
            details.append(json.dumps({
                "normalized_question": question,
                "records": [{"record_id": row["record_id"],
                             "category": row["category"]} for row in group],
            }, ensure_ascii=True))
        raise ValueError("\n".join(details))
    if invalid:
        raise ValueError("Missing or blank text/category at source record IDs: {}".format(
            ", ".join(map(str, invalid))))

    # Dict insertion order follows the original CSV record order.
    kept = [group[0] for group in question_records.values()]
    removed = [row for group in question_records.values() for row in group[1:]]
    return records, kept, removed


def validate_splits(working, train, validation, label_mapping):
    """Fail on leakage, missing/duplicated records, or missing intents."""
    expected_ids = {row["record_id"] for row in working}
    train_ids = {row["record_id"] for row in train}
    validation_ids = {row["record_id"] for row in validation}
    if len(expected_ids) != len(working):
        raise ValueError("Working data has duplicate record IDs.")
    if len(train_ids) != len(train) or len(validation_ids) != len(validation):
        raise ValueError("A split has duplicate record IDs.")
    if train_ids & validation_ids:
        raise ValueError("Record IDs overlap between splits.")
    if train_ids | validation_ids != expected_ids:
        raise ValueError("Split record IDs do not cover exactly the working data.")
    train_questions = {normalize_question(row["text"]) for row in train}
    validation_questions = {normalize_question(row["text"]) for row in validation}
    if train_questions & validation_questions:
        raise ValueError("Normalized questions overlap between splits.")
    if len(train_questions) != len(train) or len(validation_questions) != len(validation):
        raise ValueError("A split has duplicate normalized questions.")
    for name, split in (("training", train), ("validation", validation)):
        if {row["category"] for row in split} != set(label_mapping):
            raise ValueError("The {} split does not contain every intent.".format(name))
    by_id = {row["record_id"]: row for row in working}
    for row in train + validation:
        if row != by_id[row["record_id"]]:
            raise ValueError("A split record differs from its source working record.")
        if row["text_clean"] != normalize_question(row["text"]):
            raise ValueError("text_clean differs from whitespace-only normalization.")


def write_csv(path, columns, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_saved_csv(path):
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    for row in rows:
        row["record_id"] = int(row["record_id"])
    return rows


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_report(records, working, removed, train, validation, mapping, source_hash):
    original_counts = Counter(row["category"] for row in records)
    working_counts = Counter(row["category"] for row in working)
    train_counts = Counter(row["category"] for row in train)
    validation_counts = Counter(row["category"] for row in validation)
    duplicate_groups, duplicate_rows, extra_copies = duplicate_counts(
        row["text_clean"] for row in records)
    changed = sum(row["text"] != row["text_clean"] for row in records)
    lines = [
        "# BANKING77 preprocessing and shared split",
        "",
        "Source: `data/raw/train.csv`. Source SHA-256: `{}`.".format(source_hash),
        "The source hash was checked before processing and after saving the outputs; "
        "the raw file is unchanged. Only this training source is read. "
        "The official test set was not opened. No TF-IDF, learned preprocessing, "
        "or model was fitted.",
        "",
        "## Counts and checks",
        "",
        "| Check | Result |",
        "| --- | ---: |",
        "| Original records | {} |".format(len(records)),
        "| Missing or blank questions/labels | 0 |",
        "| Original intents | {} |".format(len(original_counts)),
        "| Normalized duplicate groups | {} |".format(duplicate_groups),
        "| Records participating in duplicate groups | {} |".format(duplicate_rows),
        "| Extra copies removed | {} |".format(extra_copies),
        "| Conflicting normalized question groups | 0 |",
        "| Expected working records | {} |".format(EXPECTED_WORKING_COUNT),
        "| Actual working records | {} |".format(len(working)),
        "| Working count discrepancy | {} |".format(len(working) - EXPECTED_WORKING_COUNT),
        "| Training records | {} |".format(len(train)),
        "| Validation records | {} |".format(len(validation)),
        "| Intents in training | {} |".format(len(train_counts)),
        "| Intents in validation | {} |".format(len(validation_counts)),
        "| Record ID overlap between splits | 0 |",
        "| Normalized question overlap between splits | 0 |",
        "| Original records with changed whitespace in text_clean | {} |".format(changed),
        "",
        "Checks also verify unique IDs and normalized questions within each split, "
        "complete coverage of working records, exact equality to retained source values, "
        "and correct text_clean values. Saved CSVs and JSON are read back and checked "
        "against the in-memory results before this report is saved.",
        "",
        "## Cleaning, IDs, and duplicates",
        "",
        "`record_id` is the 1-based original CSV data-record position, excluding the "
        "header (not a physical line number). IDs are assigned before deduplication "
        "and never renumbered; gaps identify removed copies. They remain stable for "
        "this source file and its record ordering, identified by the source hash above.",
        "",
        "Original columns, including `text` and `category`, retain their decoded CSV "
        "values. `text_clean` reuses `normalize_question` from `inspect_data.py`: "
        "`' '.join(text.split())`. This trims leading/trailing whitespace and collapses "
        "runs of Python-recognized whitespace to one ASCII space. Case, punctuation, "
        "spelling, and labels are preserved. No other text transformations are applied.",
        "",
        "Every normalized question group is checked for distinct original labels "
        "before any record is removed. Conflicts are reported to stderr with normalized "
        "questions, source IDs, and labels; the script exits nonzero without writing "
        "new outputs. Missing/blank inputs also stop processing. For nonconflicting "
        "groups, keep the earliest original CSV record. A working count other than "
        "9,999, or an intent count other than 77, is reported and stops output creation. "
        "Existing output files from a previous successful run are left in place on "
        "these input-validation failures.",
        "",
        "| Retained record ID | Removed record ID | Category |",
        "| ---: | ---: | --- |",
    ]
    first_ids = {row["text_clean"]: row["record_id"] for row in working}
    for row in sorted(removed, key=lambda row: row["record_id"]):
        lines.append("| {} | {} | {} |".format(
            first_ids[row["text_clean"]], row["record_id"], row["category"]))
    lines.extend([
        "",
        "## Split and label mapping",
        "",
        "Deduplicate before splitting, preserving retained source order. Call "
        "`sklearn.model_selection.train_test_split(working, test_size=0.2, "
        "random_state=42, stratify=[row['category'] for row in working])` "
        "with the default `shuffle=True`. Save each split in the order returned. "
        "The 20% validation size rounds up to 2,000; 7,999 records remain for training. "
        "Per-class integer allocation approximates 80/20 proportions.",
        "",
        "`data/processed/label_mapping.json` is a JSON object mapping each original "
        "category string to a zero-based integer. Ordering is `sorted(set(categories))` "
        "using Python's case-sensitive Unicode lexicographic order; IDs are assigned "
        "by enumeration from 0 to 76. The JSON keys are written in that same order. "
        "Both splits use this shared mapping. CSV columns are `record_id`, the original "
        "columns in source order, then `text_clean`. No numeric label column is added.",
        "",
        "## Class distributions",
        "",
        "| Dataset | Minimum class count | Maximum class count |",
        "| --- | ---: | ---: |",
    ])
    for name, counts in (("Original", original_counts), ("Working", working_counts),
                         ("Training", train_counts), ("Validation", validation_counts)):
        lines.append("| {} | {} | {} |".format(name, min(counts.values()), max(counts.values())))
    lines.extend([
        "",
        "| Label ID | Category | Original | Working | Training | Validation |",
        "| ---: | --- | ---: | ---: | ---: | ---: |",
    ])
    for category, label_id in mapping.items():
        lines.append("| {} | {} | {} | {} | {} | {} |".format(
            label_id, category, original_counts[category], working_counts[category],
            train_counts[category], validation_counts[category]))
    lines.extend([
        "",
        "## Reproduction",
        "",
        "Run from the repository root. Paths resolve relative to `preprocess_data.py`, "
        "so the script also works from another working directory. Successful reruns "
        "overwrite the three processed artifacts and this summary deterministically "
        "for the same source, code, and dependency versions.",
        "",
        "Verified environment: Python {}; NumPy {}; SciPy {}; scikit-learn {}.".format(
            platform.python_version(), numpy.__version__, scipy.__version__, sklearn.__version__),
        "",
        "Using the existing Python 3.11 environment (PowerShell):",
        "",
        "```powershell",
        r".\.venv311\Scripts\python.exe preprocess_data.py",
        r".\.venv311\Scripts\python.exe -m unittest discover -s tests -v",
        "```",
        "",
        "Or create a fresh environment with the verified versions:",
        "",
        "```powershell",
        r"py -3.11 -m venv .venv-preprocessing",
        r".\.venv-preprocessing\Scripts\python.exe -m pip install numpy=={} scipy=={} scikit-learn=={}".format(
            numpy.__version__, scipy.__version__, sklearn.__version__),
        r".\.venv-preprocessing\Scripts\python.exe preprocess_data.py",
        "```",
        "",
        "Use the saved splits and mapping across models; fit any future learned "
        "preprocessing on the saved training split only.",
        "",
        "## Output fingerprints",
        "",
        "| Artifact | SHA-256 |",
        "| --- | --- |",
    ])
    for name in ("train.csv", "validation.csv", "label_mapping.json"):
        lines.append("| data/processed/{} | `{}` |".format(name, sha256(OUTPUT_DIR / name)))
    return "\n".join(lines) + "\n"


def main():
    try:
        source_hash = sha256(INPUT_PATH)
        original_columns, rows = read_training_csv(INPUT_PATH)
        if {"record_id", "text_clean"} & set(original_columns):
            raise ValueError("Source contains reserved record_id/text_clean columns.")
        records, working, removed = prepare_records(rows)
        if len(working) != EXPECTED_WORKING_COUNT:
            raise ValueError("Working count discrepancy: expected {}, found {} (removed {}).".format(
                EXPECTED_WORKING_COUNT, len(working), len(removed)))
        labels = sorted({row["category"] for row in working})
        if len(labels) != EXPECTED_INTENTS:
            raise ValueError("Intent count discrepancy: expected {}, found {}.".format(
                EXPECTED_INTENTS, len(labels)))
        mapping = {category: label_id for label_id, category in enumerate(labels)}
        train, validation = train_test_split(
            working, test_size=VALIDATION_FRACTION, random_state=RANDOM_STATE,
            stratify=[row["category"] for row in working])
        validate_splits(working, train, validation, mapping)
        if sha256(INPUT_PATH) != source_hash:
            raise ValueError("Raw training CSV changed during processing; no outputs saved.")

        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        columns = ["record_id"] + original_columns + ["text_clean"]
        write_csv(OUTPUT_DIR / "train.csv", columns, train)
        write_csv(OUTPUT_DIR / "validation.csv", columns, validation)
        mapping_path = OUTPUT_DIR / "label_mapping.json"
        mapping_path.write_text(json.dumps(mapping, ensure_ascii=True, indent=2) + "\n", encoding="utf-8")
        saved_train = read_saved_csv(OUTPUT_DIR / "train.csv")
        saved_validation = read_saved_csv(OUTPUT_DIR / "validation.csv")
        if saved_train != train or saved_validation != validation:
            raise ValueError("Saved CSVs do not match the computed split.")
        if json.loads(mapping_path.read_text(encoding="utf-8")) != mapping:
            raise ValueError("Saved label mapping does not match the computed mapping.")
        validate_splits(working, saved_train, saved_validation, mapping)
        if sha256(INPUT_PATH) != source_hash:
            raise ValueError("Raw training CSV changed while outputs were saved.")
        report = build_report(records, working, removed, train, validation, mapping, source_hash)
        REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
        REPORT_PATH.write_text(report, encoding="utf-8")
    except (OSError, ValueError, csv.Error) as error:
        print("Preprocessing failed: {}".format(error), file=sys.stderr)
        return 1
    print("Original: {}; removed: {}; working: {} (expected {}).".format(
        len(records), len(removed), len(working), EXPECTED_WORKING_COUNT))
    print("Training: {}; validation: {}; intents in both: {}.".format(
        len(train), len(validation), len(mapping)))
    print("PASS: no conflicts, ID/question leakage, or raw-file changes; saved outputs verified.")
    print("Saved splits and mapping: {}".format(OUTPUT_DIR))
    print("Saved summary: {}".format(REPORT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
