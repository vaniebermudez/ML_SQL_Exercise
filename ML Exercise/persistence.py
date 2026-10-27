"""
models/persistence.py
---------------------
Save and load fitted models using joblib.
"""

import joblib
from pathlib import Path

from uplift_model.config import MODEL_DIR


def save_model(model, name: str) -> Path:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    path = MODEL_DIR / f"{name}.pkl"
    joblib.dump(model, path)
    print(f"[save] Model saved → {path}")
    return path


def load_model(name: str):
    path = MODEL_DIR / f"{name}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"No saved model at {path}")
    model = joblib.load(path)
    print(f"[load] Model loaded ← {path}")
    return model
