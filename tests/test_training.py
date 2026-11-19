import json

import numpy as np

from gesturebridge.dataset import feature_matrix, load_dataset
from gesturebridge.recognizer import SignClassifier
from gesturebridge.train import burst_holdout_split, train
from tests.synthetic import make_dataset


def test_burst_holdout_never_splits_a_burst(tmp_path):
    df = make_dataset(shapes=["OPEN", "FIST"], signers=("a", "b"), bursts_per_shape=3, frames_per_burst=10)
    train_idx, test_idx, single = burst_holdout_split(df, 0.25, seed=0)
    assert single == []
    train_bursts = set(df.iloc[train_idx]["burst"])
    test_bursts = set(df.iloc[test_idx]["burst"])
    assert train_bursts and test_bursts and not train_bursts & test_bursts
    assert set(df.iloc[test_idx]["label"]) == {"OPEN", "FIST"}


def test_training_end_to_end(tmp_path):
    data = tmp_path / "landmarks.csv"
    make_dataset(shapes=["OPEN", "FIST", "POINT", "THUMB_UP", "THUMB_DOWN"], signers=("a", "b"),
                 bursts_per_shape=2, frames_per_burst=25).to_csv(data, index=False)
    doc = train(data, models_dir=tmp_path / "models", reports_dir=tmp_path / "reports", augment=1, folds=3,
                log=lambda *_: None)

    assert doc["best_model"] in ("rf", "svm", "mlp")
    assert all(m["accuracy"] > 0.9 for m in doc["models"].values())
    assert set(doc["loso"]) == {"rf", "svm", "mlp"}
    assert json.loads((tmp_path / "reports" / "metrics.json").read_text(encoding="utf-8"))["labels"]
    for name in ("model_comparison", "confusion_svm", "per_class_f1", "leave_one_signer_out"):
        assert (tmp_path / "reports" / "figures" / f"{name}.png").exists()

    classifier = SignClassifier.load(tmp_path / "models" / "sign_classifier.joblib")
    X, y, _ = feature_matrix(load_dataset(data))
    probs = classifier.predict_proba(X[0])
    assert np.isclose(probs.sum(), 1.0) and classifier.classes[int(np.argmax(probs))] == y[0]


def test_signers_who_recorded_different_signs(tmp_path):
    """Regression: training crashed when one person had recorded only one sign."""
    data = tmp_path / "landmarks.csv"
    full = make_dataset(shapes=["OPEN", "FIST", "POINT"], signers=("a",), bursts_per_shape=2, frames_per_burst=20)
    partial = make_dataset(shapes=["OPEN"], signers=("b",), bursts_per_shape=1, frames_per_burst=15, seed=1)
    tiny = make_dataset(shapes=["THUMB_UP"], signers=("b",), bursts_per_shape=1, frames_per_burst=4, seed=2)
    import pandas as pd
    pd.concat([full, partial, tiny]).to_csv(data, index=False)
    doc = train(data, models_dir=tmp_path / "models", reports_dir=tmp_path / "reports", augment=1, folds=3,
                log=lambda *_: None)
    assert "THUMB_UP" not in doc["labels"]                      # 4 samples: skipped, not a crash
    assert set(doc["loso"]["svm"]) == {"b"}                     # "a" cannot be held out (b has one sign)
    assert (tmp_path / "models" / "sign_classifier.joblib").exists()
