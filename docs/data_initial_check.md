# BankAssist: initial training-data check

Source: data/raw/train.csv (resolved relative to inspect_data.py).
Columns: "text", "category".
Source SHA-256: b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b

## Rows, labels, and empty values

| Check | Count |
| --- | ---: |
| Total data rows | 10003 |
| Unique nonblank intent labels | 77 |
| Missing text values | 0 |
| Empty text strings | 0 |
| Whitespace-only text strings (nonempty) | 0 |
| Empty or whitespace-only text strings | 0 |
| Missing category values | 0 |
| Empty category strings | 0 |
| Whitespace-only category strings (nonempty) | 0 |
| Empty or whitespace-only category strings | 0 |

Missing means an absent CSV field or a zero-length field, including a quoted empty string. Empty strings are therefore included in both the missing and empty counts; these counts must not be added together. Whitespace-only means a nonempty string whose strip() result is empty. Literal strings such as NA, NULL, or NaN are preserved as strings, not silently treated as missing.

## Duplicate questions

| Comparison | Repeated question groups | Rows in repeated groups | Extra copies beyond first |
| --- | ---: | ---: | ---: |
| Exact original text | 0 | 0 | 0 |
| Trimmed and collapsed whitespace | 4 | 8 | 4 |

A group is one distinct question with frequency n >= 2. Repeated question groups counts those groups; rows in repeated groups sums n; extra copies beyond first sums n - 1. For example, a question appearing three times contributes 1 group, 3 participating rows, and 2 extra copies. The example explains the calculation and is not a dataset finding.

Exact comparison uses the original decoded CSV text without changes. Normalized comparison uses ' '.join(text.split()), trimming leading/trailing whitespace and collapsing runs of whitespace to one space. Case and punctuation are preserved. Missing and empty/whitespace-only questions are excluded from both duplicate comparisons and reported above. Labels do not affect duplicate-question counts. The two comparisons overlap and must not be added together.

## Conflicting intent labels

Normalized question groups with more than one distinct nonblank intent label: 0.
Rows belonging to these conflicting groups: 0.

Conflict detection uses the same question normalization. Intent labels are compared exactly as stored; blank/missing labels are excluded from the label sets. The row count includes every row in a conflicting question group, even a row with a blank label. Record references below are 1-based CSV data-record positions, excluding the header, not physical file line numbers or newly assigned IDs.

No conflicting intent labels found.

## Smallest and largest classes

Smallest class count: 35 rows; intent(s): "contactless_not_working".
Largest class count: 187 rows; intent(s): "card_payment_fee_charged".

Class counts and unique intent count use exact original, nonblank category strings. All rows with such labels count toward their class, including rows with missing text or duplicate questions. All tied smallest/largest classes are listed. Blank labels are reported above rather than counted as classes.

## Preservation and reproduction

All original columns and row values are read without modification. Normalization is used only in memory for comparison. No records are removed, no IDs are rewritten, no splits are created, and no model is trained. The official test set is not opened.

Run from the project root (Python 3; no extra packages needed):

    python inspect_data.py
