"""End-to-end BankAssist flow: classify a query, route it, and log it in SQLite.

    python pipeline.py "My card hasn't arrived yet"
"""

import sys

import classifier
import database
import routing


def answer_query(query, model, threshold=None, true_intent=None, db_path=database.DB_PATH):
    """Classify with the shared classifier, then route and save the request."""
    prediction = classifier.predict(query, model)
    result = routing.route_query(
        query, prediction["predicted_intent"], prediction["confidence_score"],
        threshold=threshold, true_intent=true_intent, model_name=prediction["model_name"],
        model_runtime_ms=prediction["model_runtime_ms"], db_path=db_path)
    result["top_intents"] = prediction["top_intents"]
    return result


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        print('Usage: python pipeline.py "customer question"', file=sys.stderr)
        return 1
    database.initialize_database()
    result = answer_query(" ".join(argv), classifier.load_model())
    print("Status: {status}\nPredicted intent: {predicted_intent} "
          "(confidence {confidence_score:.2f})\nResponse: {system_response}".format(**result))
    print("Top intents: {}".format(", ".join(
        "{} {:.2f}".format(intent, score) for intent, score in result["top_intents"])))
    return 0


if __name__ == "__main__":
    sys.exit(main())
