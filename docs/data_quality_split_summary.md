# BankAssist Phase 3: data-quality and split summary

Date: 2026-10-28 (America/Los_Angeles)
Repository inspected at commit: f788a97
Status: Awaiting training data. No data processing or splitting performed.

## Verified repository findings

- README.md is the only project file in the working tree and tracked history.
- The working tree was clean before this report was added.
- Available local/cached branches: main and origin/main; no other teammate implementation or approved split files were found.
- README.md requires shared saved splits and label mapping, training-only fitting of learned preprocessing, and reservation of the official test set for final evaluation.
- No remote fetch or dataset download was performed; findings cover this checkout and its existing Git references.

## Specific input needed

Provide the official BANKING77 training CSV containing question text and intent labels. Suggested repository location: data/raw/banking77/train.csv. This is a proposed location, not an existing file. If the team already has approved split manifests and a label mapping, provide their existing locations/files as well so they can be validated and reused.

The official test set is not needed for this session and remains reserved.

## Data checks and splits

Actual columns, row count, missing values, empty/whitespace-only text, duplicate questions, conflicting labels, existing record IDs, and class counts: not assessable because no training data is present. No counts or quality claims have been inferred.

No original text or record IDs were changed. No cleaning, ID creation, split assignment, label mapping, or dataset artifacts were generated. No model was trained.

After training data is supplied, inspect its real schema and existing IDs first. Preserve the original text and stable source record IDs; use a separate text field for minimal whitespace cleaning. Validate approved splits against training records, including coverage, disjointness, class counts, and normalized-question overlap. If no approved splits exist, create reproducible group-aware training/validation splits using identical whitespace-normalized questions as groups, preserving class proportions where feasible. Record the seed, validation fraction, source fingerprint, grouping rule, and any conflicts or stratification constraints when those choices can be checked against the actual data.

## Reproduce the repository inspection

Run these commands in PowerShell from the repository root:

```powershell
git rev-parse --short HEAD
git status --short
git branch --all
git log --all --format='%h %s' --name-only
git ls-files
rg --files --hidden -g '!.git/**'
Get-Content README.md
```

Expected result at the inspected commit: README.md is the only tracked project file and no dataset or approved splits are present. After saving this report, the file inventory/status also includes docs/data_quality_split_summary.md.