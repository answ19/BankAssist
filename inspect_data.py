"""Inspect training data without modifying it. Uses only the Python standard library."""

import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent
INPUT_PATH = ROOT / "data" / "raw" / "train.csv"
OUTPUT_PATH = ROOT / "docs" / "data_initial_check.md"


def normalize_question(text):
    """Trim and collapse whitespace; preserve case, punctuation, and spelling."""
    return " ".join(text.split())


def is_blank(value):
    return value is None or not value.strip()


def duplicate_counts(values):
    """Return repeated groups, participating rows, and copies beyond the first."""
    repeated_sizes = [size for size in Counter(values).values() if size > 1]
    return (
        len(repeated_sizes),
        sum(repeated_sizes),
        sum(size - 1 for size in repeated_sizes),
    )


def read_training_csv(path):
    if not path.is_file():
        raise ValueError("Training data is missing: {}".format(path))
    # utf-8-sig accepts UTF-8 with or without a byte-order mark.
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle, strict=True)
        columns = reader.fieldnames or []
        if len(columns) != len(set(columns)):
            raise ValueError("CSV contains duplicate column names.")
        missing_columns = {"text", "category"} - set(columns)
        if missing_columns:
            raise ValueError(
                "CSV is missing required columns: {}".format(
                    ", ".join(sorted(missing_columns))
                )
            )
        rows = []
        for record_number, row in enumerate(reader, start=1):
            if None in row:
                raise ValueError(
                    "CSV data record {} has more fields than the header.".format(
                        record_number
                    )
                )
            rows.append(row)
    return columns, rows


def quoted(value):
    """Keep report values on one line and protect Markdown table separators."""
    return json.dumps(value, ensure_ascii=True).replace("|", r"\|")


def build_report(columns, rows, source_sha256):
    texts = [row["text"] for row in rows]
    labels = [row["category"] for row in rows]
    valid_texts = [text for text in texts if not is_blank(text)]
    class_counts = Counter(label for label in labels if not is_blank(label))
    exact = duplicate_counts(valid_texts)
    normalized = duplicate_counts(
        normalize_question(text) for text in valid_texts
    )

    question_labels = defaultdict(Counter)
    question_records = defaultdict(list)
    for record_number, row in enumerate(rows, start=1):
        if is_blank(row["text"]):
            continue
        question = normalize_question(row["text"])
        question_records[question].append(record_number)
        if not is_blank(row["category"]):
            question_labels[question][row["category"]] += 1
    conflicts = {
        question: counts
        for question, counts in question_labels.items()
        if len(counts) > 1
    }
    conflict_rows = sum(len(question_records[question]) for question in conflicts)

    lines = [
        "# BankAssist: initial training-data check",
        "",
        "Source: data/raw/train.csv (resolved relative to inspect_data.py).",
        "Columns: {}.".format(", ".join(quoted(name) for name in columns)),
        "Source SHA-256: {}".format(source_sha256),
        "",
        "## Rows, labels, and empty values",
        "",
        "| Check | Count |",
        "| --- | ---: |",
        "| Total data rows | {} |".format(len(rows)),
        "| Unique nonblank intent labels | {} |".format(len(class_counts)),
    ]
    for name, values in (("text", texts), ("category", labels)):
        missing = sum(value is None or value == "" for value in values)
        empty = sum(value == "" for value in values)
        whitespace = sum(
            value is not None and value != "" and not value.strip()
            for value in values
        )
        lines.extend([
            "| Missing {} values | {} |".format(name, missing),
            "| Empty {} strings | {} |".format(name, empty),
            "| Whitespace-only {} strings (nonempty) | {} |".format(name, whitespace),
            "| Empty or whitespace-only {} strings | {} |".format(
                name, empty + whitespace
            ),
        ])
    lines.extend([
        "",
        "Missing means an absent CSV field or a zero-length field, including a "
        "quoted empty string. Empty strings are therefore included in both the "
        "missing and empty counts; these counts must not be added together. "
        "Whitespace-only means a nonempty string whose strip() result is empty. "
        "Literal strings such as NA, NULL, or NaN are preserved as strings, "
        "not silently treated as missing.",
        "",
        "## Duplicate questions",
        "",
        "| Comparison | Repeated question groups | Rows in repeated groups | Extra copies beyond first |",
        "| --- | ---: | ---: | ---: |",
        "| Exact original text | {} | {} | {} |".format(*exact),
        "| Trimmed and collapsed whitespace | {} | {} | {} |".format(*normalized),
        "",
        "A group is one distinct question with frequency n >= 2. Repeated "
        "question groups counts those groups; rows in repeated groups sums n; "
        "extra copies beyond first sums n - 1. For example, a question appearing "
        "three times contributes 1 group, 3 participating rows, and 2 extra copies. "
        "The example explains the calculation and is not a dataset finding.",
        "",
        "Exact comparison uses the original decoded CSV text without changes. "
        "Normalized comparison uses ' '.join(text.split()), trimming leading/"
        "trailing whitespace and collapsing runs of whitespace to one space. "
        "Case and punctuation are preserved. Missing and empty/whitespace-only "
        "questions are excluded from both duplicate comparisons and reported "
        "above. Labels do not affect duplicate-question counts. The two "
        "comparisons overlap and must not be added together.",
        "",
        "## Conflicting intent labels",
        "",
        "Normalized question groups with more than one distinct nonblank "
        "intent label: {}.".format(len(conflicts)),
        "Rows belonging to these conflicting groups: {}.".format(conflict_rows),
        "",
        "Conflict detection uses the same question normalization. Intent labels "
        "are compared exactly as stored; blank/missing labels are excluded from "
        "the label sets. The row count includes every row in a conflicting "
        "question group, even a row with a blank label. Record references below "
        "are 1-based CSV data-record positions, excluding the header, "
        "not physical file line numbers or newly assigned IDs.",
    ])
    if conflicts:
        lines.extend([
            "",
            "| Normalized question | Intent label: row count | CSV data records |",
            "| --- | --- | --- |",
        ])
        for question, counts in sorted(conflicts.items()):
            label_details = "; ".join(
                "{}: {}".format(quoted(label), count)
                for label, count in sorted(counts.items())
            )
            lines.append("| {} | {} | {} |".format(
                quoted(question),
                label_details,
                ", ".join(map(str, question_records[question])),
            ))
    else:
        lines.extend(["", "No conflicting intent labels found."])
    lines.extend(["", "## Smallest and largest classes", ""])
    if class_counts:
        smallest = min(class_counts.values())
        largest = max(class_counts.values())
        for title, count in (("Smallest", smallest), ("Largest", largest)):
            names = sorted(
                label for label, size in class_counts.items() if size == count
            )
            lines.append("{} class count: {} rows; intent(s): {}.".format(
                title, count, ", ".join(quoted(name) for name in names)
            ))
    else:
        lines.append("No nonblank intent labels; class minimum/maximum are undefined.")
    lines.extend([
        "",
        "Class counts and unique intent count use exact original, nonblank "
        "category strings. All rows with such labels count toward their class, "
        "including rows with missing text or duplicate questions. All tied "
        "smallest/largest classes are listed. Blank labels are reported above "
        "rather than counted as classes.",
        "",
        "## Preservation and reproduction",
        "",
        "All original columns and row values are read without modification. "
        "Normalization is used only in memory for comparison. No records are "
        "removed, no IDs are rewritten, no splits are created, and no model is "
        "trained. The official test set is not opened.",
        "",
        "Run from the project root (Python 3; no extra packages needed):",
        "",
        "    python inspect_data.py",
        "",
    ])
    return "\n".join(lines)


def main():
    try:
        before = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
        columns, rows = read_training_csv(INPUT_PATH)
        report = build_report(columns, rows, before)
        after = hashlib.sha256(INPUT_PATH.read_bytes()).hexdigest()
        if after != before:
            raise ValueError("Training CSV changed during inspection; no report saved.")
        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_PATH.write_text(report, encoding="utf-8")
    except (OSError, ValueError, csv.Error) as error:
        print("Inspection failed: {}".format(error), file=sys.stderr)
        return 1
    print(report)
    print("Saved summary: {}".format(OUTPUT_PATH))
    return 0


if __name__ == "__main__":
    sys.exit(main())
