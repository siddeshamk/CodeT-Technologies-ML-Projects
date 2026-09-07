from pathlib import Path
import joblib

MODEL_FILE = Path(__file__).parent / "model" / "spam_classifier.joblib"


def load_model():
    if not MODEL_FILE.exists():
        raise FileNotFoundError(
            "Model not found. Run train_model.py first."
        )
    return joblib.load(MODEL_FILE)


def predict_message(message, model):
    label = model.predict([message])[0]
    probabilities = model.predict_proba([message])[0]
    confidence = max(probabilities)
    return label, confidence


if __name__ == "__main__":
    model = load_model()
    print("Spam Email Classifier")
    print("Type 'exit' to stop.\n")

    while True:
        message = input("Enter a message: ").strip()
        if message.lower() == "exit":
            break

        label, confidence = predict_message(message, model)
        result = "SPAM" if label == "spam" else "NOT SPAM"
        print(f"Prediction: {result} | Confidence: {confidence:.2%}\n")
