"""End-to-end BankAssist flow: classify a query, route it, and log it in SQLite.

    python pipeline.py "My card hasn't arrived yet"
    python pipeline.py --ollama "My card hasn't arrived yet"   # Ollama writes the reply
"""

import sys

import classifier
import database
import routing


def answer_query(query, model, threshold=None, true_intent=None, db_path=database.DB_PATH,
                 use_ollama=None):
    """Classify with the shared classifier, then route and save the request."""
    prediction = classifier.predict(query, model)
    result = routing.route_query(
        query, prediction["predicted_intent"], prediction["confidence_score"],
        threshold=threshold, true_intent=true_intent, model_name=prediction["model_name"],
        model_runtime_ms=prediction["model_runtime_ms"], db_path=db_path,
        use_ollama=use_ollama)
    result["top_intents"] = prediction["top_intents"]
    return result


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    use_ollama = "--ollama" in argv
    argv = [arg for arg in argv if arg != "--ollama"]
    if not argv:
        print('Usage: python pipeline.py [--ollama] "customer question"', file=sys.stderr)
        return 1
    database.initialize_database()
    result = answer_query(" ".join(argv), classifier.load_model(),
                          use_ollama=True if use_ollama else None)
    print("Status: {status}\nPredicted intent: {predicted_intent} "
          "(confidence {confidence_score:.2f})\nResponse: {system_response}".format(**result))
    print("Top intents: {}".format(", ".join(
        "{} {:.2f}".format(top["intent"], top["probability"])
        for top in result["top_intents"])))
    if result.get("review_reason"):
        print("Note: {}".format(result["review_reason"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
