"""Minimal Streamlit screen for reviewing pending BankAssist requests.

Run from the repository root:
    streamlit run review_app.py
Set BANKASSIST_DB_PATH to review a database other than bankassist.db.
"""

import os

import streamlit as st

import database
import routing


DB_PATH = os.environ.get("BANKASSIST_DB_PATH", str(database.DB_PATH))

st.set_page_config(page_title="BankAssist review", layout="centered")
st.title("BankAssist human review")
st.caption("Database: {}".format(DB_PATH))
database.initialize_database(DB_PATH)

if "message" in st.session_state:
    st.success(st.session_state.pop("message"))

pending = database.get_pending_reviews(DB_PATH)
if not pending:
    st.info("No requests are waiting for review.")
    st.stop()

choice = st.selectbox(
    "Pending request ({} waiting)".format(len(pending)), pending,
    format_func=lambda r: "#{}: {}".format(r["request_id"], r["query"][:70]))

if choice["is_test_case"]:
    st.warning("Test case: the prediction and confidence score were entered by hand "
               "to exercise routing. They are not model output.")
st.subheader("Customer query")
st.write(choice["query"])
left, right = st.columns(2)
left.metric("Predicted intent", choice["predicted_intent"] or "None")
right.metric("Confidence score", "{:.2f}".format(choice["confidence_score"])
             if choice["confidence_score"] is not None else "None")

st.subheader("Approved FAQ for this intent (held back, not sent)")
if choice["faq_id"] is None:
    st.write("No approved FAQ exists for the predicted intent.")
else:
    st.markdown("**Q:** {}".format(choice["faq_question"]))
    st.markdown("**A:** {}".format(choice["faq_answer"]))
    st.caption("FAQ {} · source: {}".format(choice["faq_id"], choice["faq_source"]))
reason = routing.explain_review(choice)
if reason:
    st.info("Why this needs review: {}".format(reason))
if choice.get("ollama_draft"):
    st.subheader("Ollama draft (not sent)")
    st.write(choice["ollama_draft"])
    st.caption("Written by the local model from the FAQ above, then held back by a safety "
               "check. Edit starts from the approved FAQ text, not this draft.")
st.subheader("Current system response")
st.write(choice["system_response"])

st.subheader("Decision")
key = choice["request_id"]
decision = st.radio("Decision", ["approved", "edited", "rejected"], horizontal=True,
                    key="decision-{}".format(key),
                    format_func={"approved": "Approve", "edited": "Edit",
                                 "rejected": "Reject"}.get)
corrected_intent = final_response = None
if decision == "edited":
    corrected_intent = st.text_input("Corrected intent (leave blank to keep the prediction)",
                                     key="intent-{}".format(key))
    final_response = st.text_area("Final response", value=choice["faq_answer"] or "",
                                  key="response-{}".format(key))
elif decision == "rejected":
    final_response = st.text_area("Message to customer (optional)",
                                  key="reject-{}".format(key))
notes = st.text_area("Reviewer notes", key="notes-{}".format(key))

if st.button("Submit decision", type="primary"):
    try:
        database.save_review(key, decision, final_response, corrected_intent, notes, DB_PATH)
    except ValueError as error:
        st.error(str(error))
    else:
        st.session_state["message"] = "Request #{} saved as {}.".format(key, decision)
        st.rerun()
