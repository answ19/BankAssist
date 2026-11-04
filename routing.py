"""Route a classifier prediction to an approved FAQ answer or to human review.

This module does not classify anything. It takes the shared classifier's output
(predicted intent and confidence score) and decides what the customer is shown.
"""

import math
import os

import database


# Default 0.90 until the team validates a threshold on the validation split.
# Override with the BANKASSIST_CONFIDENCE_THRESHOLD environment variable or the
# threshold argument of route_query.
CONFIDENCE_THRESHOLD = float(os.environ.get("BANKASSIST_CONFIDENCE_THRESHOLD", "0.90"))

ESCALATION_MESSAGE = ("We could not confirm an approved answer for this question. "
                      "It has been sent to a member of staff for review.")


def check_probability(name, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("{} must be a number, got {!r}.".format(name, value))
    if math.isnan(value) or not 0 <= value <= 1:
        raise ValueError("{} must be between 0 and 1, got {!r}.".format(name, value))
    return float(value)


def route_query(query, predicted_intent, confidence_score, threshold=None,
                true_intent=None, model_name=None, model_runtime_ms=None,
                ollama_runtime_ms=None, is_test_case=False, db_path=database.DB_PATH):
    """Save the request and return it in the shared prediction-output format.

    The request is answered automatically only when the confidence reaches the
    threshold and an approved FAQ exists for the predicted intent. Otherwise it
    is saved as pending_review; any FAQ found is kept as evidence for the reviewer
    but is not shown to the customer.

    The confidence score measures confidence in the predicted intent, not the
    correctness of any answer text.
    """
    threshold = check_probability(
        "threshold", CONFIDENCE_THRESHOLD if threshold is None else threshold)
    confidence_score = check_probability("confidence_score", confidence_score)
    faq = database.get_faq_by_intent(predicted_intent, db_path) if predicted_intent else None
    if faq is not None and confidence_score >= threshold:
        status, system_response = "answered", faq["answer"]
    else:
        status, system_response = "pending_review", ESCALATION_MESSAGE
    faq_id = faq["faq_id"] if faq else None
    request_id = database.save_request(
        query, predicted_intent, confidence_score, faq_id, system_response, status,
        true_intent=true_intent, model_name=model_name,
        model_runtime_ms=model_runtime_ms, ollama_runtime_ms=ollama_runtime_ms,
        is_test_case=is_test_case, db_path=db_path)
    return {
        "request_id": request_id,
        "query": query,
        "predicted_intent": predicted_intent,
        "confidence_score": confidence_score,
        "faq_id": faq_id,
        "faq_answer": faq["answer"] if faq else None,
        "system_response": system_response,
        "status": status,
    }
