"""train_model.py

Train a simple intent classifier (TF-IDF + Logistic Regression).
Saves a joblib pipeline to models/intent_pipeline.joblib and writes metadata.

This script exposes a `train()` function so it can be imported and used
by the FastAPI app for in-process retraining in demo scenarios.
"""

import json
from pathlib import Path
from datetime import datetime
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

ROOT = Path(".")
DATA_PATH = ROOT / "data" / "intent_data.csv"
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "intent_pipeline.joblib"
MODEL_META = MODEL_DIR / "model_metadata.json"
TRAIN_LOG = ROOT / "logs" / "training_log.csv"
TRAIN_LOG.parent.mkdir(exist_ok=True)


def train(save=True):
    """Train the intent model and save artifacts.

    Returns:
        dict: metadata {version, timestamp, accuracy}
    """
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH} — run prepare_data.py first."
        )

    df = pd.read_csv(DATA_PATH)
    # Very small dataset: keep train/test split but expect low scores.
    X = df["text"].astype(str)
    y = df["intent"].astype(str)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42
    )

    # Pipeline: TF-IDF + logistic regression (simple, interpretable)
    pipe = Pipeline(
        [
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=2000)),
            ("clf", LogisticRegression(max_iter=200)),
        ]
    )

    pipe.fit(X_train, y_train)
    preds = pipe.predict(X_test)
    acc = accuracy_score(y_test, preds)

    metadata = {
        "version": datetime.utcnow().strftime("%Y%m%d%H%M%S"),
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "accuracy_test": float(acc),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
    }

    if save:
        joblib.dump(pipe, MODEL_PATH)
        with MODEL_META.open("w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)
        # append a small line to train log
        with TRAIN_LOG.open("a", encoding="utf-8") as f:
            f.write(
                f"{metadata['timestamp_utc']},{metadata['version']},{metadata['accuracy_test']},{metadata['n_train']},{metadata['n_test']}\n"
            )

    print("Trained model. Test accuracy:", metadata["accuracy_test"])
    print("Saved model to:", MODEL_PATH.resolve())
    return metadata


if __name__ == "__main__":
    train()
