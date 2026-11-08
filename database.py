"""SQLite storage for approved FAQs, routed requests, and human-review decisions.

Intent names are stored as plain text so the module does not depend on the final
label mapping. Every query uses SQL parameters, never string formatting.
"""

import csv
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DB_PATH = ROOT / "bankassist.db"

REQUEST_STATUSES = ("answered", "pending_review", "approved", "edited", "rejected")
REVIEW_DECISIONS = ("approved", "edited", "rejected")

SCHEMA = """
CREATE TABLE IF NOT EXISTS faqs (
    faq_id INTEGER PRIMARY KEY AUTOINCREMENT,
    intent TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT NOT NULL,
    approved INTEGER NOT NULL DEFAULT 0 CHECK (approved IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS faqs_intent ON faqs (intent, approved);

CREATE TABLE IF NOT EXISTS requests (
    request_id INTEGER PRIMARY KEY AUTOINCREMENT,
    query TEXT NOT NULL,
    predicted_intent TEXT,
    confidence_score REAL
        CHECK (confidence_score IS NULL OR confidence_score BETWEEN 0 AND 1),
    faq_id INTEGER REFERENCES faqs (faq_id),
    system_response TEXT,
    status TEXT NOT NULL CHECK (status IN
        ('answered', 'pending_review', 'approved', 'edited', 'rejected')),
    true_intent TEXT,
    model_name TEXT,
    model_runtime_ms REAL,
    ollama_runtime_ms REAL,
    review_reason TEXT,
    ollama_draft TEXT,
    is_test_case INTEGER NOT NULL DEFAULT 0 CHECK (is_test_case IN (0, 1)),
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS requests_status ON requests (status);

CREATE TABLE IF NOT EXISTS reviews (
    review_id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id INTEGER NOT NULL UNIQUE REFERENCES requests (request_id),
    decision TEXT NOT NULL CHECK (decision IN ('approved', 'edited', 'rejected')),
    corrected_intent TEXT,
    final_response TEXT,
    reviewer_notes TEXT,
    reviewed_at TEXT NOT NULL
);
"""


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path=DB_PATH):
    connection = sqlite3.connect(str(db_path))
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def blank_to_none(value):
    if value is None or not str(value).strip():
        return None
    return str(value).strip()


def require_text(name, value):
    value = blank_to_none(value)
    if value is None:
        raise ValueError("{} must not be blank.".format(name))
    return value


def initialize_database(db_path=DB_PATH):
    with closing(connect(db_path)) as connection, connection:
        connection.executescript(SCHEMA)
        # Databases created before these columns existed gain them here.
        columns = {row["name"] for row in connection.execute("PRAGMA table_info(requests)")}
        for name, kind in (("ollama_runtime_ms", "REAL"), ("review_reason", "TEXT"),
                           ("ollama_draft", "TEXT")):
            if name not in columns:
                connection.execute("ALTER TABLE requests ADD COLUMN {} {}".format(name, kind))


def add_faq(intent, question, answer, source, approved=False, db_path=DB_PATH):
    """Insert one FAQ and return its faq_id. Only approved FAQs are ever retrieved."""
    values = (require_text("intent", intent), require_text("question", question),
              require_text("answer", answer), require_text("source", source),
              int(bool(approved)), now())
    with closing(connect(db_path)) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO faqs (intent, question, answer, source, approved, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)", values)
        return cursor.lastrowid


def load_faqs_from_csv(csv_path, db_path=DB_PATH, known_intents=None):
    """Insert or update FAQs from a CSV with intent, question, answer, source, approved.

    A row whose intent and question already exist updates that FAQ's answer, source,
    and approval in place, so faq_ids (and requests that reference them) stay stable.
    Every row is validated before anything is written. When known_intents is given
    (e.g. the keys of label_mapping.json), every FAQ intent must be one of them, so a
    typo cannot create an unreachable FAQ.
    """
    with Path(csv_path).open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle, strict=True))
    if rows:
        missing = {"intent", "question", "answer", "source", "approved"} - set(rows[0])
        if missing:
            raise ValueError("FAQ CSV is missing columns: {}".format(
                ", ".join(sorted(missing))))
    if known_intents is not None:
        unknown = sorted({row["intent"] for row in rows} - set(known_intents))
        if unknown:
            raise ValueError("FAQ intents not in the label mapping: {}".format(
                ", ".join(unknown)))
    approved_values = {"1": True, "true": True, "yes": True,
                       "0": False, "false": False, "no": False}
    faqs = []
    for number, row in enumerate(rows, start=1):
        approved = approved_values.get(str(row["approved"]).strip().lower())
        if approved is None:
            raise ValueError("FAQ CSV record {} has an invalid approved value: {!r}".format(
                number, row["approved"]))
        faqs.append((require_text("intent", row["intent"]),
                     require_text("question", row["question"]),
                     require_text("answer", row["answer"]),
                     require_text("source", row["source"]), int(approved)))
    ids = []
    with closing(connect(db_path)) as connection, connection:
        for intent, question, answer, source, approved in faqs:
            match = connection.execute(
                "SELECT faq_id FROM faqs WHERE intent = ? AND question = ? "
                "ORDER BY faq_id LIMIT 1", (intent, question)).fetchone()
            if match:
                connection.execute(
                    "UPDATE faqs SET answer = ?, source = ?, approved = ? WHERE faq_id = ?",
                    (answer, source, approved, match["faq_id"]))
                ids.append(match["faq_id"])
            else:
                cursor = connection.execute(
                    "INSERT INTO faqs (intent, question, answer, source, approved, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)", (intent, question, answer, source, approved, now()))
                ids.append(cursor.lastrowid)
    return ids


def get_faq_by_intent(intent, db_path=DB_PATH):
    """Return the earliest approved FAQ for an intent as a dict, or None."""
    with closing(connect(db_path)) as connection:
        row = connection.execute(
            "SELECT * FROM faqs WHERE intent = ? AND approved = 1 "
            "ORDER BY faq_id LIMIT 1", (intent,)).fetchone()
    return dict(row) if row else None


def save_request(query, predicted_intent, confidence_score, faq_id, system_response,
                 status, true_intent=None, model_name=None, model_runtime_ms=None,
                 ollama_runtime_ms=None, is_test_case=False, db_path=DB_PATH,
                 review_reason=None, ollama_draft=None):
    """Log one routed request and return its request_id.

    model_runtime_ms is classifier prediction time only; ollama_runtime_ms is the
    local LLM response time, kept separate. is_test_case marks hand-entered scores used to exercise the routing code, so
    they can be kept out of reported model results. review_reason says why a request
    was sent to review (or why Ollama wording was skipped); ollama_draft keeps a reply
    the Ollama step wrote but did not send, so the reviewer can see it.
    """
    if status not in REQUEST_STATUSES:
        raise ValueError("Unknown request status: {!r}".format(status))
    values = (require_text("query", query), blank_to_none(predicted_intent),
              confidence_score, faq_id, system_response, status,
              blank_to_none(true_intent), blank_to_none(model_name),
              model_runtime_ms, ollama_runtime_ms, blank_to_none(review_reason),
              blank_to_none(ollama_draft), int(bool(is_test_case)), now())
    with closing(connect(db_path)) as connection, connection:
        cursor = connection.execute(
            "INSERT INTO requests (query, predicted_intent, confidence_score, faq_id, "
            "system_response, status, true_intent, model_name, model_runtime_ms, "
            "ollama_runtime_ms, review_reason, ollama_draft, is_test_case, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", values)
        return cursor.lastrowid


def get_pending_reviews(db_path=DB_PATH):
    """Return pending requests, oldest first, with the retrieved FAQ as evidence."""
    with closing(connect(db_path)) as connection:
        rows = connection.execute(
            "SELECT r.*, f.question AS faq_question, f.answer AS faq_answer, "
            "f.source AS faq_source FROM requests r "
            "LEFT JOIN faqs f ON f.faq_id = r.faq_id "
            "WHERE r.status = 'pending_review' ORDER BY r.request_id").fetchall()
    return [dict(row) for row in rows]


def save_review(request_id, decision, final_response=None, corrected_intent=None,
                reviewer_notes=None, db_path=DB_PATH):
    """Record a reviewer decision and set the request status to match it.

    approved: the retrieved FAQ answer becomes the final response.
    edited: the reviewer supplies the final response (and optionally the intent).
    rejected: no answer is sent; final_response may hold a referral message.
    """
    if decision not in REVIEW_DECISIONS:
        raise ValueError("Unknown review decision: {!r}".format(decision))
    final_response = blank_to_none(final_response)
    with closing(connect(db_path)) as connection, connection:
        request = connection.execute(
            "SELECT r.status, f.answer AS faq_answer FROM requests r "
            "LEFT JOIN faqs f ON f.faq_id = r.faq_id WHERE r.request_id = ?",
            (request_id,)).fetchone()
        if request is None:
            raise ValueError("Request {} does not exist.".format(request_id))
        if request["status"] != "pending_review":
            raise ValueError("Request {} is {}, not pending_review.".format(
                request_id, request["status"]))
        if decision == "approved":
            if request["faq_answer"] is None:
                raise ValueError("Request {} has no FAQ answer to approve; "
                                 "edit or reject it instead.".format(request_id))
            final_response = request["faq_answer"]
        elif decision == "edited" and final_response is None:
            raise ValueError("An edited review needs a final response.")
        cursor = connection.execute(
            "INSERT INTO reviews (request_id, decision, corrected_intent, final_response, "
            "reviewer_notes, reviewed_at) VALUES (?, ?, ?, ?, ?, ?)",
            (request_id, decision, blank_to_none(corrected_intent), final_response,
             blank_to_none(reviewer_notes), now()))
        connection.execute("UPDATE requests SET status = ? WHERE request_id = ?",
                           (decision, request_id))
        return cursor.lastrowid


def get_evaluation_records(include_test_cases=False, db_path=DB_PATH):
    """Return every request joined with its review (if any), oldest first."""
    sql = ("SELECT r.*, v.review_id, v.decision, v.corrected_intent, v.final_response, "
           "v.reviewer_notes, v.reviewed_at FROM requests r "
           "LEFT JOIN reviews v ON v.request_id = r.request_id")
    if not include_test_cases:
        sql += " WHERE r.is_test_case = 0"
    with closing(connect(db_path)) as connection:
        rows = connection.execute(sql + " ORDER BY r.request_id").fetchall()
    return [dict(row) for row in rows]
