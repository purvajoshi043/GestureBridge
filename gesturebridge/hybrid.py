"""Hybrid recogniser: your trained signs + all instant signs at the same time.

* Signs you recorded and trained (words, letters, numbers or your own new words) are
  recognised by the trained model. If you recorded your own handshape for an existing
  sign (e.g. YOU), it REPLACES the built-in rule for that sign.
* Every other sign is still recognised by the instant handshape rules (58 signs).
* A trained model only knows the signs it was trained on, so for any hand it will name
  one of them. A novelty check stops that: the trained model only wins when the hand
  really looks like your recordings of that sign (nearest-neighbour distance in the
  standardised feature space, compared with how far your own recordings are from each
  other). Otherwise the instant rules decide.

The same machinery checks a new recording: if the recorded handshape is already
recognised as ANOTHER sign of the same mode, the app warns "this sign already exists".
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pandas as pd

from .handshape_rules import HANDSHAPES, MODES, RuleClassifier, display_label, points_from_features

NOVELTY_PER_CLASS = 300      # stored reference frames per sign
TOP_TRAINED = 3              # novelty is checked for the trained model's best few signs


def mode_of(label: str) -> str:
    if label.startswith("LETTER_"):
        return "letters"
    if label.startswith("DIGIT_"):
        return "numbers"
    return "words"


def _ramp(value: float, start: float, end: float) -> float:
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def _min_dist(a: np.ndarray, b: np.ndarray, chunk: int = 512) -> np.ndarray:
    """For every row of a: the distance to its nearest row of b."""
    out = np.empty(len(a))
    for i in range(0, len(a), chunk):
        d = ((a[i:i + chunk, None, :] - b[None, :, :]) ** 2).sum(-1)
        out[i:i + chunk] = np.sqrt(d.min(1))
    return out


def build_novelty(X: np.ndarray, y: np.ndarray, groups: np.ndarray, seed: int = 0) -> dict:
    """Reference frames and an acceptance distance per sign (from unaugmented training features)."""
    X, y, groups = np.asarray(X, dtype=np.float64), np.asarray(y), np.asarray(groups)
    mean = X.mean(0)
    scale = X.std(0)
    scale[scale < 1e-6] = 1.0
    Z = (X - mean) / scale
    rng = np.random.default_rng(seed)
    classes = {}
    for label in sorted(set(y)):
        idx = np.flatnonzero(y == label)
        Zc, gc = Z[idx], groups[idx]
        recordings = sorted(set(gc))
        if len(recordings) >= 2:
            # how far a frame is from the nearest frame of ANOTHER recording of the same sign
            dists = np.concatenate([_min_dist(Zc[gc == g], Zc[gc != g]) for g in recordings])
            threshold = float(np.percentile(dists, 95))
        else:   # one recording only: neighbouring frames are near-duplicates, so be generous
            d = ((Zc[:, None, :] - Zc[None, :, :]) ** 2).sum(-1)
            np.fill_diagonal(d, np.inf)
            threshold = float(np.percentile(np.sqrt(d.min(1)), 95)) * 3.0
        keep = rng.choice(len(Zc), size=min(len(Zc), NOVELTY_PER_CLASS), replace=False)
        classes[str(label)] = {"X": Zc[keep].astype(np.float32), "threshold": max(threshold, 1e-3)}
    return {"mean": mean.astype(np.float64), "scale": scale.astype(np.float64), "classes": classes}


def novelty_from_dataset(df: pd.DataFrame, classes) -> dict | None:
    """For models trained before the novelty data was stored in the bundle."""
    from .dataset import feature_matrix
    df = df[df["label"].isin(list(classes))]
    if df.empty:
        return None
    X, y, groups = feature_matrix(df, augment_copies=0)
    return build_novelty(X, y, groups)


class HybridClassifier:
    """Same interface as the other classifiers (classes, predict_proba, predict, description)."""

    model_name = "hybrid"
    mirror_left = True

    def __init__(self, trained=None, mode: str = "words", novelty: dict | None = None,
                 exclude: set[str] | frozenset = frozenset()):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.trained = trained
        self.novelty = novelty
        self.rules = RuleClassifier(mode)
        self.last_facts: dict[str, float] = {}
        trained_classes = list(trained.classes) if trained is not None else []
        self.trained_labels = [c for c in trained_classes
                               if c != "NOTHING" and mode_of(c) == mode and c not in exclude]
        # your own recordings of a built-in sign replace its rule
        self.overridden = {c for c in trained_classes if c in HANDSHAPES[mode]}
        self.instant_labels = [label for label in HANDSHAPES[mode]
                               if label not in self.overridden and label not in exclude]
        self.classes = sorted(set(self.instant_labels) | set(self.trained_labels) | {"NOTHING"})
        self._index = {c: i for i, c in enumerate(self.classes)}
        self._trained_index = {c: i for i, c in enumerate(trained_classes)}

    @property
    def description(self) -> str:
        mine = ", ".join(display_label(c) for c in self.trained_labels) or "none in this mode"
        return (f"HYBRID ({self.mode}) - your trained signs ({mine}) + {len(self.instant_labels)} "
                f"instant signs")

    def closeness(self, z: np.ndarray, label: str) -> float:
        """1 = looks like your recordings of `label`, 0 = clearly different."""
        if self.novelty is None:
            return 1.0
        ref = self.novelty["classes"].get(label)
        if ref is None:
            return 0.0
        d = float(np.sqrt(((ref["X"] - z) ** 2).sum(1).min()))
        return _ramp(d, 2.0 * ref["threshold"], ref["threshold"])

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float64)
        scores = self.rules.scores(points_from_features(features))
        self.last_facts = self.rules.last_facts
        instant = {label: scores[label] for label in self.instant_labels}
        instant["NOTHING"] = max(0.05, 1.0 - max(instant.values(), default=0.0))
        strength = {k: v ** 2 + 1e-4 for k, v in instant.items()}
        total = sum(strength.values())
        q = {k: v / total for k, v in strength.items()}

        evidence: dict[str, float] = {}
        if self.trained_labels:
            p = self.trained.predict_proba(features)
            z = (features - self.novelty["mean"]) / self.novelty["scale"] if self.novelty else features
            best = sorted(self.trained_labels, key=lambda c: -p[self._trained_index[c]])[:TOP_TRAINED]
            evidence = {c: float(p[self._trained_index[c]]) * self.closeness(z, c) for c in best}
        trust = max(evidence.values(), default=0.0)

        out = np.full(len(self.classes), 1e-6)
        for label, value in evidence.items():
            out[self._index[label]] += value
        for label, value in q.items():
            out[self._index[label]] += (1.0 - trust) * value
        return out / out.sum()

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.predict_proba(features)
        best = int(np.argmax(probs))
        return self.classes[best], float(probs[best])


def find_conflict(features: np.ndarray, label: str, trained=None, novelty: dict | None = None,
                  min_share: float = 0.5, min_prob: float = 0.6) -> tuple[str, float] | None:
    """Is this recording of `label` already the handshape of ANOTHER sign in the same mode?

    Returns (other sign, share of frames) or None. Signs in other modes may share a
    handshape (V = CLEAR in words, V in letters, 2 in numbers), like real sign languages.
    """
    if label == "NOTHING" or len(features) == 0:
        return None
    checker = HybridClassifier(trained, mode_of(label), novelty, exclude={label})
    votes = Counter()
    for row in features:
        other, prob = checker.predict(row)
        if other != "NOTHING" and prob >= min_prob:
            votes[other] += 1
    if not votes:
        return None
    other, count = votes.most_common(1)[0]
    share = count / len(features)
    return (other, share) if share >= min_share else None
