"""Loads a trained sign classifier and turns features into class probabilities."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from .config import MODEL_PATH
from .features import FEATURE_VERSION


class ModelNotFoundError(FileNotFoundError):
    pass


def _single_threaded(estimator) -> None:
    """Thread pools cost more than they save when predicting one frame at a time."""
    steps = getattr(estimator, "steps", None)
    for _, step in steps or [(None, estimator)]:
        if hasattr(step, "n_jobs"):
            step.n_jobs = 1


class SignClassifier:
    def __init__(self, bundle: dict, path: Path | None = None):
        if bundle.get("feature_version") != FEATURE_VERSION:
            raise ValueError("This model was trained with a different feature version - retrain it: "
                             "python -m gesturebridge.train")
        self.bundle = bundle
        self.path = path
        self.model = bundle["pipeline"]
        self.classes: list[str] = list(bundle["classes"])
        self.model_name: str = bundle.get("model_name", "model")
        self.mirror_left: bool = bundle.get("mirror_left", True)
        _single_threaded(self.model)

    @classmethod
    def load(cls, path: Path = MODEL_PATH) -> "SignClassifier":
        path = Path(path)
        if not path.exists():
            raise ModelNotFoundError(
                f"No trained model at {path}. Collect data (python -m gesturebridge.collect) "
                "and train (python -m gesturebridge.train) first.")
        return cls(joblib.load(path), path)

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(np.asarray(features, dtype=np.float64).reshape(1, -1))[0]

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.predict_proba(features)
        best = int(np.argmax(probs))
        return self.classes[best], float(probs[best])

    @property
    def description(self) -> str:
        info = self.bundle
        return (f"{self.model_name.upper()} - {len(self.classes)} classes, trained on "
                f"{info.get('n_samples', '?')} samples ({info.get('trained_at', '?')})")
