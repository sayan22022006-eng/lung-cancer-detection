import sys
import joblib
from pathlib import Path

MODEL_FILE = Path("models/text/tfidf_logistic_model.joblib")
VECTORIZER_FILE = Path("models/text/tfidf_vectorizer.joblib")
THRESHOLD_FILE = Path("models/text/decision_threshold.txt")

TARGET_LABEL = "parenchymal_lesion"


def load_components():
    model = joblib.load(MODEL_FILE)
    vectorizer = joblib.load(VECTORIZER_FILE)

    with open(THRESHOLD_FILE, "r") as f:
        threshold = float(f.read().strip())

    return model, vectorizer, threshold


def predict_text(text):
    model, vectorizer, threshold = load_components()

    text = text.strip()

    if not text:
        raise ValueError("Text cannot be empty.")

    features = vectorizer.transform([text])

    probability = float(model.predict_proba(features)[0][1])

    prediction = (
        "PARENCHYMAL_LESION"
        if probability >= threshold
        else "NO_PARENCHYMAL_LESION"
    )

    return {
        "prediction": prediction,
        "probability": probability,
        "threshold": threshold
    }


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage:")
        print('python src/text_classifier/predict_text.py "your report text"')
        sys.exit(1)

    text = " ".join(sys.argv[1:])

    result = predict_text(text)

    print("\n==============================")
    print("TEXT CLASSIFIER RESULT")
    print("==============================")
    print(f"Prediction : {result['prediction']}")
    print(f"Probability: {result['probability']:.4f}")
    print(f"Threshold  : {result['threshold']:.2f}")
