import io
import zipfile
from pathlib import Path
from urllib.request import urlopen

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

DATA_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "model"
DATA_FILE = DATA_DIR / "SMSSpamCollection"
MODEL_FILE = MODEL_DIR / "spam_classifier.joblib"


def download_dataset():
    DATA_DIR.mkdir(exist_ok=True)

    if DATA_FILE.exists():
        return

    print("Downloading UCI SMS Spam Collection...")
    with urlopen(DATA_URL, timeout=30) as response:
        archive_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        archive.extract("SMSSpamCollection", DATA_DIR)


def load_dataset():
    download_dataset()
    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "message"],
        encoding="utf-8",
    )
    return df


def train():
    df = load_dataset()

    X_train, X_test, y_train, y_test = train_test_split(
        df["message"],
        df["label"],
        test_size=0.2,
        random_state=42,
        stratify=df["label"],
    )

    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            lowercase=True,
            stop_words="english",
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("classifier", MultinomialNB()),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    print(f"\nTest accuracy: {accuracy:.4f}\n")
    print(classification_report(y_test, predictions))

    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(pipeline, MODEL_FILE)
    print(f"Saved model to: {MODEL_FILE}")


if __name__ == "__main__":
    train()
