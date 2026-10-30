# BANKING77 preprocessing and shared split

Source: `data/raw/train.csv`. Source SHA-256: `b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b`.
The source hash was checked before processing and after saving the outputs; the raw file is unchanged. Only this training source is read. The official test set was not opened. No TF-IDF, learned preprocessing, or model was fitted.

## Counts and checks

| Check | Result |
| --- | ---: |
| Original records | 10003 |
| Missing or blank questions/labels | 0 |
| Original intents | 77 |
| Normalized duplicate groups | 4 |
| Records participating in duplicate groups | 8 |
| Extra copies removed | 4 |
| Conflicting normalized question groups | 0 |
| Expected working records | 9999 |
| Actual working records | 9999 |
| Working count discrepancy | 0 |
| Training records | 7999 |
| Validation records | 2000 |
| Intents in training | 77 |
| Intents in validation | 77 |
| Record ID overlap between splits | 0 |
| Normalized question overlap between splits | 0 |
| Original records with changed whitespace in text_clean | 466 |

Checks also verify unique IDs and normalized questions within each split, complete coverage of working records, exact equality to retained source values, and correct text_clean values. Saved CSVs and JSON are read back and checked against the in-memory results before this report is saved.

## Cleaning, IDs, and duplicates

`record_id` is the 1-based original CSV data-record position, excluding the header (not a physical line number). IDs are assigned before deduplication and never renumbered; gaps identify removed copies. They remain stable for this source file and its record ordering, identified by the source hash above.

Original columns, including `text` and `category`, retain their decoded CSV values. `text_clean` reuses `normalize_question` from `inspect_data.py`: `' '.join(text.split())`. This trims leading/trailing whitespace and collapses runs of Python-recognized whitespace to one ASCII space. Case, punctuation, spelling, and labels are preserved. No other text transformations are applied.

Every normalized question group is checked for distinct original labels before any record is removed. Conflicts are reported to stderr with normalized questions, source IDs, and labels; the script exits nonzero without writing new outputs. Missing/blank inputs also stop processing. For nonconflicting groups, keep the earliest original CSV record. A working count other than 9,999, or an intent count other than 77, is reported and stops output creation. Existing output files from a previous successful run are left in place on these input-validation failures.

| Retained record ID | Removed record ID | Category |
| ---: | ---: | --- |
| 1247 | 1291 | card_not_working |
| 1711 | 1725 | pin_blocked |
| 4595 | 4596 | atm_support |
| 6911 | 6966 | change_pin |

## Split and label mapping

Deduplicate before splitting, preserving retained source order. Call `sklearn.model_selection.train_test_split(working, test_size=0.2, random_state=42, stratify=[row['category'] for row in working])` with the default `shuffle=True`. Save each split in the order returned. The 20% validation size rounds up to 2,000; 7,999 records remain for training. Per-class integer allocation approximates 80/20 proportions.

`data/processed/label_mapping.json` is a JSON object mapping each original category string to a zero-based integer. Ordering is `sorted(set(categories))` using Python's case-sensitive Unicode lexicographic order; IDs are assigned by enumeration from 0 to 76. The JSON keys are written in that same order. Both splits use this shared mapping. CSV columns are `record_id`, the original columns in source order, then `text_clean`. No numeric label column is added.

## Class distributions

| Dataset | Minimum class count | Maximum class count |
| --- | ---: | ---: |
| Original | 35 | 187 |
| Working | 35 | 187 |
| Training | 28 | 149 |
| Validation | 7 | 38 |

| Label ID | Category | Original | Working | Training | Validation |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | Refund_not_showing_up | 162 | 162 | 130 | 32 |
| 1 | activate_my_card | 159 | 159 | 127 | 32 |
| 2 | age_limit | 110 | 110 | 88 | 22 |
| 3 | apple_pay_or_google_pay | 126 | 126 | 101 | 25 |
| 4 | atm_support | 87 | 86 | 69 | 17 |
| 5 | automatic_top_up | 127 | 127 | 102 | 25 |
| 6 | balance_not_updated_after_bank_transfer | 171 | 171 | 137 | 34 |
| 7 | balance_not_updated_after_cheque_or_cash_deposit | 181 | 181 | 145 | 36 |
| 8 | beneficiary_not_allowed | 156 | 156 | 125 | 31 |
| 9 | cancel_transfer | 157 | 157 | 126 | 31 |
| 10 | card_about_to_expire | 129 | 129 | 103 | 26 |
| 11 | card_acceptance | 59 | 59 | 47 | 12 |
| 12 | card_arrival | 153 | 153 | 122 | 31 |
| 13 | card_delivery_estimate | 112 | 112 | 90 | 22 |
| 14 | card_linking | 139 | 139 | 111 | 28 |
| 15 | card_not_working | 112 | 111 | 89 | 22 |
| 16 | card_payment_fee_charged | 187 | 187 | 149 | 38 |
| 17 | card_payment_not_recognised | 168 | 168 | 134 | 34 |
| 18 | card_payment_wrong_exchange_rate | 167 | 167 | 134 | 33 |
| 19 | card_swallowed | 61 | 61 | 49 | 12 |
| 20 | cash_withdrawal_charge | 177 | 177 | 142 | 35 |
| 21 | cash_withdrawal_not_recognised | 160 | 160 | 128 | 32 |
| 22 | change_pin | 122 | 121 | 97 | 24 |
| 23 | compromised_card | 86 | 86 | 69 | 17 |
| 24 | contactless_not_working | 35 | 35 | 28 | 7 |
| 25 | country_support | 129 | 129 | 103 | 26 |
| 26 | declined_card_payment | 153 | 153 | 122 | 31 |
| 27 | declined_cash_withdrawal | 173 | 173 | 138 | 35 |
| 28 | declined_transfer | 133 | 133 | 106 | 27 |
| 29 | direct_debit_payment_not_recognised | 182 | 182 | 145 | 37 |
| 30 | disposable_card_limits | 121 | 121 | 97 | 24 |
| 31 | edit_personal_details | 121 | 121 | 97 | 24 |
| 32 | exchange_charge | 121 | 121 | 97 | 24 |
| 33 | exchange_rate | 112 | 112 | 90 | 22 |
| 34 | exchange_via_app | 118 | 118 | 94 | 24 |
| 35 | extra_charge_on_statement | 166 | 166 | 133 | 33 |
| 36 | failed_transfer | 137 | 137 | 110 | 27 |
| 37 | fiat_currency_support | 126 | 126 | 101 | 25 |
| 38 | get_disposable_virtual_card | 97 | 97 | 78 | 19 |
| 39 | get_physical_card | 106 | 106 | 85 | 21 |
| 40 | getting_spare_card | 129 | 129 | 103 | 26 |
| 41 | getting_virtual_card | 98 | 98 | 78 | 20 |
| 42 | lost_or_stolen_card | 82 | 82 | 66 | 16 |
| 43 | lost_or_stolen_phone | 121 | 121 | 97 | 24 |
| 44 | order_physical_card | 120 | 120 | 96 | 24 |
| 45 | passcode_forgotten | 105 | 105 | 84 | 21 |
| 46 | pending_card_payment | 159 | 159 | 127 | 32 |
| 47 | pending_cash_withdrawal | 143 | 143 | 114 | 29 |
| 48 | pending_top_up | 149 | 149 | 119 | 30 |
| 49 | pending_transfer | 148 | 148 | 118 | 30 |
| 50 | pin_blocked | 115 | 114 | 91 | 23 |
| 51 | receiving_money | 95 | 95 | 76 | 19 |
| 52 | request_refund | 169 | 169 | 135 | 34 |
| 53 | reverted_card_payment? | 161 | 161 | 129 | 32 |
| 54 | supported_cards_and_currencies | 129 | 129 | 103 | 26 |
| 55 | terminate_account | 108 | 108 | 86 | 22 |
| 56 | top_up_by_bank_transfer_charge | 111 | 111 | 89 | 22 |
| 57 | top_up_by_card_charge | 114 | 114 | 91 | 23 |
| 58 | top_up_by_cash_or_cheque | 114 | 114 | 91 | 23 |
| 59 | top_up_failed | 145 | 145 | 116 | 29 |
| 60 | top_up_limits | 97 | 97 | 78 | 19 |
| 61 | top_up_reverted | 146 | 146 | 117 | 29 |
| 62 | topping_up_by_card | 103 | 103 | 82 | 21 |
| 63 | transaction_charged_twice | 175 | 175 | 140 | 35 |
| 64 | transfer_fee_charged | 172 | 172 | 138 | 34 |
| 65 | transfer_into_account | 113 | 113 | 90 | 23 |
| 66 | transfer_not_received_by_recipient | 171 | 171 | 137 | 34 |
| 67 | transfer_timing | 128 | 128 | 102 | 26 |
| 68 | unable_to_verify_identity | 102 | 102 | 82 | 20 |
| 69 | verify_my_identity | 104 | 104 | 83 | 21 |
| 70 | verify_source_of_funds | 113 | 113 | 90 | 23 |
| 71 | verify_top_up | 126 | 126 | 101 | 25 |
| 72 | virtual_card_not_working | 41 | 41 | 33 | 8 |
| 73 | visa_or_mastercard | 135 | 135 | 108 | 27 |
| 74 | why_verify_identity | 121 | 121 | 97 | 24 |
| 75 | wrong_amount_of_cash_received | 180 | 180 | 144 | 36 |
| 76 | wrong_exchange_rate_for_cash_withdrawal | 163 | 163 | 130 | 33 |

## Reproduction

Run from the repository root. Paths resolve relative to `preprocess_data.py`, so the script also works from another working directory. Successful reruns overwrite the three processed artifacts and this summary deterministically for the same source, code, and dependency versions.

Verified environment: Python 3.11.9; NumPy 2.4.6; SciPy 1.17.1; scikit-learn 1.9.1.

Using the existing Python 3.11 environment (PowerShell):

```powershell
.\.venv311\Scripts\python.exe preprocess_data.py
.\.venv311\Scripts\python.exe -m unittest discover -s tests -v
```

Or create a fresh environment with the verified versions:

```powershell
py -3.11 -m venv .venv-preprocessing
.\.venv-preprocessing\Scripts\python.exe -m pip install numpy==2.4.6 scipy==1.17.1 scikit-learn==1.9.1
.\.venv-preprocessing\Scripts\python.exe preprocess_data.py
```

Use the saved splits and mapping across models; fit any future learned preprocessing on the saved training split only.

## Output fingerprints

| Artifact | SHA-256 |
| --- | --- |
| data/processed/train.csv | `3068a34bd41aa623e31faaa272110809e8948ebeb15b064e432a13c860ebd0c0` |
| data/processed/validation.csv | `d9ae544bcc70a4477af09060470e8e45281f3786d43f7eb8b4d9df76e0d35a75` |
| data/processed/label_mapping.json | `7ed70c1e6727112af01cb7e3c7b473bc1e0b0c35c666e29a81279ab0635348e0` |
