"""
Run predictions with the trained sentiment classifier.

The model is a scikit-learn LogisticRegression classifier trained on
TF-IDF style features (5911 input features, classes: negative / neutral /
positive). It expects numeric feature vectors, not raw text -- so a
matching vectorizer (e.g. TfidfVectorizer / CountVectorizer) that was
fit on the same training vocabulary is required to turn text into the
5911-length input the model expects.

Usage:
    python src/predict.py "I really love this product!"

If you have the vectorizer that was used during training, save it as
model/vectorizer.pkl and this script will load it automatically. If it's
not present, the script raises a clear error instead of failing silently
or trying to load an untrusted vocabulary file.
"""

import sys
import pickle
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "sentiment_model.pkl"
VECTORIZER_PATH = Path(__file__).resolve().parent.parent / "model" / "vectorizer.pkl"


def load_model(path: Path = MODEL_PATH):
    """Load the trained classifier.

    Security note: pickle files can execute arbitrary code when loaded.
    Only load model/vectorizer files you trust (e.g. ones you trained
    yourself or that come from this repository's own releases).
    """
    with open(path, "rb") as f:
        return pickle.load(f)


def load_vectorizer(path: Path = VECTORIZER_PATH):
    if not path.exists():
        raise FileNotFoundError(
            f"No vectorizer found at {path}.\n"
            "This model expects pre-vectorized numeric input (5911 features). "
            "Add the TfidfVectorizer/CountVectorizer used during training as "
            "model/vectorizer.pkl, or adapt this script to your own "
            "text-to-feature pipeline."
        )
    with open(path, "rb") as f:
        return pickle.load(f)


def predict(text: str) -> str:
    model = load_model()
    vectorizer = load_vectorizer()
    features = vectorizer.transform([text])
    prediction = model.predict(features)[0]
    return prediction


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print('Usage: python src/predict.py "your text here"')
        sys.exit(1)

    input_text = " ".join(sys.argv[1:])
    try:
        result = predict(input_text)
        print(f"Text: {input_text}")
        print(f"Predicted sentiment: {result}")
    except FileNotFoundError as e:
        print(f"Error: {e}")
        sys.exit(1)
