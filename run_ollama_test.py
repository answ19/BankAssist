"""Test the Ollama wording step on a fixed list of questions and save the results.

Needs Ollama running with the model pulled (for example:  ollama pull qwen2.5:3b).

    python run_ollama_test.py
    python run_ollama_test.py --model llama3.2:3b     # try another model

For each question the script uses the APPROVED FAQ of the given intent (the intent
is fixed in the test file, so this tests the prompt, not the classifier) and
records what Ollama did, how long it took, and whether that matched what the team
expected. It writes:

    results/ollama_answer_test.csv   one row per question
    docs/ollama_answer_test.md       summary table and per-question table

The last column of the markdown table, "Accurate, nothing added?", is for a team
member to fill in by reading each answer. The script cannot judge that.
"""

import argparse
import csv
import re
import statistics
import sys
import tempfile
from datetime import date
from pathlib import Path

import classifier
import database
import ollama_answer

ROOT = Path(__file__).resolve().parent
QUESTIONS_PATH = ROOT / "data" / "ollama_test_questions.csv"
FAQ_PATH = ROOT / "data" / "faq" / "faq_drafts.csv"


def repeats_query_digits(query, answer):
    """True if the reply repeats a 4+ digit number from the customer's message."""
    wanted = set(re.findall(r"\d{4,}", query.replace(" ", "")))
    return bool(answer) and any(number in answer.replace(" ", "") for number in wanted)


def judge(expected, status, repeated):
    """Did Ollama do what the team expected?"""
    if repeated:
        return False
    if expected == "either":
        return status in ("ok", "needs_review")
    return status == expected


def run(questions, faq_lookup, model, url):
    rows = []
    for number, q in enumerate(questions, start=1):
        faq = faq_lookup(q["intent"])
        if faq is None:
            raise ValueError("No approved FAQ for intent {!r} (question {}).".format(
                q["intent"], q["id"]))
        result = ollama_answer.write_answer(q["query"], faq, model=model, url=url)
        if result["status"] == "unavailable":
            raise ConnectionError(result["reason"])
        repeated = repeats_query_digits(q["query"], result["answer"])
        rows.append({**q, "status": result["status"], "reason": result["reason"],
                     "answer": result["answer"] or "", "runtime_ms": result["runtime_ms"],
                     "matches_expected": judge(q["expected"], result["status"], repeated),
                     "repeats_number": repeated})
        print("{}/{} {:.0f} ms".format(number, len(questions), result["runtime_ms"]), end="\r")
    return rows


def esc(text):
    return str(text).replace("|", r"\|").replace("\n", " ")


def write_reports(rows, model, out_csv, out_md):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    fields = ["id", "query", "intent", "expected", "status", "matches_expected",
              "repeats_number", "runtime_ms", "reason", "answer", "note"]
    with out_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    times = [r["runtime_ms"] for r in rows]
    total = len(rows)
    matched = sum(r["matches_expected"] for r in rows)
    answered = sum(r["status"] == "ok" for r in rows)
    lines = [
        "# Ollama answer-wording test",
        "",
        "Date: {}. Model: `{}`. {} questions from `data/ollama_test_questions.csv`. "
        "Each question uses the approved FAQ of its given intent, so this tests the "
        "prompt and safety checks, not the classifier. Temperature 0.".format(
            date.today(), model, total),
        "",
        "| Measure | Result |",
        "|---|---:|",
        "| Ollama behaved as expected | {:.0%} ({}/{}) |".format(matched / total, matched, total),
        "| Wrote an answer (status ok) | {} |".format(answered),
        "| Sent to review (needs_review) | {} |".format(total - answered),
        "| Replies that repeated a number from the customer | {} |".format(
            sum(r["repeats_number"] for r in rows)),
        "| Mean time per question | {:.0f} ms |".format(statistics.mean(times)),
        "| Median time per question | {:.0f} ms |".format(statistics.median(times)),
        "| Slowest question | {:.0f} ms |".format(max(times)),
        "",
        "\"Behaved as expected\" only checks the decision (answer or review). It does not "
        "prove an answer is correct. Read every answer and fill in the last column by hand.",
        "",
        "| # | Question | Expected | Result | As expected | Reply or reason | Accurate, nothing added? |",
        "|---:|---|---|---|:---:|---|---|",
    ]
    for r in rows:
        shown = r["answer"] if r["answer"] else r["reason"]
        lines.append("| {} | {} | {} | {} | {} | {} | |".format(
            r["id"], esc(r["query"]), r["expected"], r["status"],
            "yes" if r["matches_expected"] else "NO", esc(shown)))
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines[:15])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", default=ollama_answer.OLLAMA_MODEL)
    parser.add_argument("--url", default=ollama_answer.OLLAMA_URL)
    parser.add_argument("--questions", default=str(QUESTIONS_PATH))
    parser.add_argument("--faqs", default=str(FAQ_PATH))
    parser.add_argument("--out-dir", default=str(ROOT), help="folder holding results/ and docs/")
    args = parser.parse_args(argv)

    with Path(args.questions).open(encoding="utf-8-sig", newline="") as handle:
        questions = list(csv.DictReader(handle))

    with tempfile.TemporaryDirectory() as folder:
        db_path = Path(folder) / "ollama_test.db"
        database.initialize_database(db_path)
        database.load_faqs_from_csv(args.faqs, db_path, known_intents=classifier.read_label_mapping())
        lookup = lambda intent: database.get_faq_by_intent(intent, db_path)

        # warm-up call, not timed: loads the model into memory
        first = lookup(questions[0]["intent"])
        warm = ollama_answer.write_answer(questions[0]["query"], first, model=args.model, url=args.url)
        if warm["status"] == "unavailable":
            print("Cannot reach Ollama at {} ({}). Is it running, and is {} pulled?".format(
                args.url, warm["reason"], args.model), file=sys.stderr)
            return 1
        try:
            rows = run(questions, lookup, args.model, args.url)
        except (ConnectionError, ValueError) as error:
            print("Stopped: {}".format(error), file=sys.stderr)
            return 1

    out = Path(args.out_dir)
    print(write_reports(rows, args.model, out / "results" / "ollama_answer_test.csv",
                        out / "docs" / "ollama_answer_test.md"))
    print("\nSaved results/ollama_answer_test.csv and docs/ollama_answer_test.md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
