"""Train and compare sign classifiers: Random Forest vs SVM vs MLP.

    python -m gesturebridge.train                    # uses data/raw/landmarks.csv
    python -m gesturebridge.train --augment 3 --tune # more augmentation + grid search

Evaluation design (important for an honest accuracy):
* Burst-level hold-out: every recording burst (≈100 near-identical consecutive
  frames) goes entirely to train OR test. A random frame split would put
  almost-duplicate frames on both sides and inflate the accuracy.
* Cross-validation on the training part with the same grouping.
* Leave-one-signer-out (if >= 2 signers): train on everybody else, test on a
  person the model has never seen - the most realistic test of a new user.
* Augmentation (small rotations, hand turns, landmark noise) only on training data.

Outputs: models/sign_classifier.joblib (best model, retrained on all data),
models/model_<name>.joblib, reports/metrics.json, reports/training_summary.md,
reports/figures/*.png
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold, StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import plots
from .config import METRICS_PATH, MODEL_PATH, MODELS_DIR, RAW_DATA_PATH, REPORTS_DIR
from .dataset import feature_matrix, group_ids, load_dataset, summary
from .features import FEATURE_NAMES, FEATURE_VERSION
from .recognizer import _single_threaded

MODEL_KEYS = ("rf", "svm", "mlp")
MIN_SAMPLES_PER_SIGN = 10
REPORTED_PARAMS = ("n_estimators", "max_depth", "min_samples_leaf", "calibratedclassifiercv__estimator__C",
                   "calibratedclassifiercv__estimator__gamma", "mlpclassifier__hidden_layer_sizes",
                   "mlpclassifier__alpha")

PARAM_GRIDS = {
    "rf": {"n_estimators": [100, 200], "max_depth": [None, 20], "min_samples_leaf": [1, 2]},
    "svm": {"calibratedclassifiercv__estimator__C": [1, 10, 100],
            "calibratedclassifiercv__estimator__gamma": ["scale", 0.01]},
    "mlp": {"mlpclassifier__hidden_layer_sizes": [(128, 64), (256, 128)],
            "mlpclassifier__alpha": [1e-4, 1e-3]},
}


def build_models(seed: int = 42) -> dict:
    return {
        "rf": RandomForestClassifier(n_estimators=100, class_weight="balanced", n_jobs=-1, random_state=seed),
        # Platt-scaled SVM: CalibratedClassifierCV turns SVM scores into probabilities,
        # which the stabiliser and the beam-search decoder need.
        "svm": make_pipeline(StandardScaler(), CalibratedClassifierCV(
            SVC(kernel="rbf", C=10, gamma="scale", class_weight="balanced", random_state=seed),
            method="sigmoid", ensemble=False, cv=5)),
        "mlp": make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(128, 64), alpha=1e-3,
                                                             max_iter=500, early_stopping=True,
                                                             n_iter_no_change=20, random_state=seed)),
    }


def burst_holdout_split(df: pd.DataFrame, test_size: float, seed: int) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Per label, hold out ~test_size of its recording bursts. Labels with a single burst
    are split in time inside the burst (first 80% train, last 20% test) and reported."""
    rng = np.random.default_rng(seed)
    groups = group_ids(df)
    test_mask = np.zeros(len(df), dtype=bool)
    single_burst = []
    for label in sorted(df["label"].unique()):
        idx = np.flatnonzero(df["label"].to_numpy() == label)
        label_groups = sorted(set(groups[idx]))
        if len(label_groups) >= 2:
            rng.shuffle(label_groups)
            n_test = min(len(label_groups) - 1, max(1, round(len(label_groups) * test_size)))
            test_mask[idx] = np.isin(groups[idx], label_groups[:n_test])
        else:
            single_burst.append(label)
            order = idx[np.argsort(df["timestamp"].to_numpy()[idx], kind="stable")]
            cut = int(round(len(order) * (1 - test_size)))
            test_mask[order[cut:]] = True
    return np.flatnonzero(~test_mask), np.flatnonzero(test_mask), single_burst


def cv_splitter(y: np.ndarray, groups: np.ndarray, folds: int, seed: int):
    per_label = pd.Series(groups).groupby(y).nunique()
    n = int(min(folds, per_label.min()))
    if n >= 2:
        return StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=seed), groups, "grouped by burst"
    counts = pd.Series(y).value_counts().min()
    return StratifiedKFold(n_splits=int(max(2, min(folds, counts))), shuffle=True, random_state=seed), None, \
        "stratified by frame (record >= 2 bursts per sign for grouped CV)"


def prediction_latency_ms(model, X: np.ndarray, repeats: int = 200) -> float:
    """Median time for one live-frame prediction (single-threaded, as in the app)."""
    timed = copy.deepcopy(model)
    _single_threaded(timed)
    rows = X[np.random.default_rng(0).integers(0, len(X), size=min(repeats, len(X)))]
    timed.predict_proba(rows[:1])  # warm-up
    times = []
    for row in rows:
        started = time.perf_counter()
        timed.predict_proba(row.reshape(1, -1))
        times.append((time.perf_counter() - started) * 1000)
    return float(np.median(times))


def evaluate(model, X: np.ndarray, y: np.ndarray, labels: list[str]) -> dict:
    predicted = model.predict(X)
    precision, recall, f1, _ = precision_recall_fscore_support(y, predicted, labels=labels, average="macro",
                                                               zero_division=0)
    matrix = confusion_matrix(y, predicted, labels=labels)
    with np.errstate(invalid="ignore", divide="ignore"):
        normalised = np.nan_to_num(matrix / matrix.sum(axis=1, keepdims=True))
    report = classification_report(y, predicted, labels=labels, output_dict=True, zero_division=0)
    confusions = [
        {"true": labels[i], "predicted": labels[j], "rate": round(float(normalised[i, j]), 4),
         "count": int(matrix[i, j])}
        for i in range(len(labels)) for j in range(len(labels)) if i != j and matrix[i, j] > 0
    ]
    confusions.sort(key=lambda c: c["rate"], reverse=True)
    return {
        "accuracy": float(accuracy_score(y, predicted)),
        "macro_precision": float(precision), "macro_recall": float(recall), "macro_f1": float(f1),
        "per_class": {label: {"precision": round(report[label]["precision"], 4),
                              "recall": round(report[label]["recall"], 4),
                              "f1": round(report[label]["f1-score"], 4),
                              "support": int(report[label]["support"])} for label in labels},
        "confusion": {"labels": labels, "matrix": matrix.tolist(), "normalised": normalised.round(4).tolist()},
        "top_confusions": confusions[:10],
        "_predicted": predicted,
    }


def make_bundle(model, name: str, df: pd.DataFrame, metrics: dict) -> dict:
    return {
        "pipeline": model,
        "classes": [str(c) for c in model.classes_],
        "model_name": name,
        "feature_version": FEATURE_VERSION,
        "feature_names": FEATURE_NAMES,
        "mirror_left": True,
        "trained_at": time.strftime("%Y-%m-%d %H:%M"),
        "n_samples": int(len(df)),
        "signers": sorted(df["signer"].unique().tolist()),
        "test_metrics": {k: metrics[k] for k in ("accuracy", "macro_f1")},
    }


def group_accuracy(df_test: pd.DataFrame, y_true: np.ndarray, predicted: np.ndarray, column: str) -> dict:
    result = {}
    for value in sorted(df_test[column].unique()):
        mask = df_test[column].to_numpy() == value
        result[str(value)] = {"accuracy": round(float(np.mean(y_true[mask] == predicted[mask])), 4),
                              "samples": int(mask.sum())}
    return result


def train(data_path: Path = RAW_DATA_PATH, models_dir: Path = MODELS_DIR, reports_dir: Path = REPORTS_DIR,
          model_keys=MODEL_KEYS, augment: int = 2, test_size: float = 0.2, folds: int = 5,
          tune: bool = False, loso: bool = True, seed: int = 42, log=print) -> dict:
    started_all = time.perf_counter()
    models_dir, reports_dir = Path(models_dir), Path(reports_dir)
    figures_dir = reports_dir / "figures"
    df = load_dataset(data_path)
    counts = df["label"].value_counts()
    too_few = sorted(counts[counts < MIN_SAMPLES_PER_SIGN].index)
    if too_few:
        log(f"  skipping {too_few}: fewer than {MIN_SAMPLES_PER_SIGN} samples - record them again")
        df = df[~df["label"].isin(too_few)].reset_index(drop=True)
    stats = summary(df)
    labels = sorted(df["label"].unique())
    if len(labels) < 2:
        raise ValueError("Need at least two different signs to train a classifier.")
    log(f"Dataset: {stats['samples']} samples, {len(labels)} signs, signers: {stats['signers']}")
    small = {k: v for k, v in stats["labels"].items() if v < 50}
    if small:
        log(f"  warning: few samples (<50) for {small} - record more for reliable results")

    train_idx, test_idx, single_burst = burst_holdout_split(df, test_size, seed)
    df_train, df_test = df.iloc[train_idx].reset_index(drop=True), df.iloc[test_idx].reset_index(drop=True)
    if single_burst:
        log(f"  warning: only one recording burst for {single_burst}; their test frames come from the same "
            "burst, so their accuracy is optimistic. Record a second burst for each sign.")
    log(f"Split: {len(df_train)} train / {len(df_test)} test samples (whole bursts held out)")

    X_train_raw, y_train_raw, g_train = feature_matrix(df_train, augment_copies=0)
    X_train, y_train, _ = feature_matrix(df_train, augment_copies=augment, seed=seed)
    X_test, y_test, _ = feature_matrix(df_test, augment_copies=0)
    log(f"Features: {X_train_raw.shape[1]} per sample; training set with augmentation: {len(X_train)}")

    splitter, cv_groups, cv_mode = cv_splitter(y_train_raw, g_train, folds, seed)
    candidates = build_models(seed)
    smallest = int(pd.Series(y_train_raw).value_counts().min())
    if smallest < 10:  # Platt scaling cross-validates inside the training data
        candidates["svm"].set_params(calibratedclassifiercv__cv=2)
    results: dict[str, dict] = {}
    fitted: dict[str, object] = {}
    for key in model_keys:
        model = candidates[key]
        name = plots.MODEL_NAMES[key]
        if tune:
            log(f"[{name}] grid search ...")
            search = GridSearchCV(model, PARAM_GRIDS[key], scoring="f1_macro", cv=splitter, n_jobs=-1)
            search.fit(X_train_raw, y_train_raw, groups=cv_groups)
            model = clone(model).set_params(**search.best_params_)
            log(f"  best params: {search.best_params_}")
        log(f"[{name}] {splitter.get_n_splits()}-fold cross-validation ({cv_mode}) ...")
        cv = cross_validate(clone(model), X_train_raw, y_train_raw, groups=cv_groups, cv=splitter,
                            scoring={"accuracy": "accuracy", "macro_f1": "f1_macro"}, n_jobs=-1)
        t0 = time.perf_counter()
        model = clone(model).fit(X_train, y_train)
        train_seconds = time.perf_counter() - t0
        metrics = evaluate(model, X_test, y_test, labels)
        metrics.update({
            "name": name,
            "cv": {"accuracy_mean": float(cv["test_accuracy"].mean()), "accuracy_std": float(cv["test_accuracy"].std()),
                   "macro_f1_mean": float(cv["test_macro_f1"].mean()), "macro_f1_std": float(cv["test_macro_f1"].std()),
                   "folds": int(splitter.get_n_splits()), "mode": cv_mode},
            "train_seconds": round(train_seconds, 2),
            "latency_ms": prediction_latency_ms(model, X_test),
            "params": {k: str(v) for k, v in model.get_params().items() if k in REPORTED_PARAMS},
        })
        results[key], fitted[key] = metrics, model
        log(f"  test accuracy {metrics['accuracy']:.3f}  macro-F1 {metrics['macro_f1']:.3f}  "
            f"CV macro-F1 {metrics['cv']['macro_f1_mean']:.3f}±{metrics['cv']['macro_f1_std']:.3f}  "
            f"{metrics['latency_ms']:.2f} ms/frame")

    best = max(results, key=lambda k: (round(results[k]["macro_f1"], 3), -results[k]["latency_ms"]))
    log(f"Best model: {results[best]['name']} (macro-F1 {results[best]['macro_f1']:.3f})")

    groups = {}
    predicted_best = results[best]["_predicted"]
    for column in ("condition", "handedness", "signer"):
        if df_test[column].nunique() >= 1:
            groups[column] = group_accuracy(df_test, y_test, predicted_best, column)

    loso_results: dict[str, dict[str, dict]] = {}
    signers = sorted(df["signer"].unique())
    if loso and len(signers) >= 2:
        log(f"Leave-one-signer-out over {len(signers)} signers ...")
        for key in model_keys:
            loso_results[key] = {}
            for signer in signers:
                held = df["signer"] == signer
                df_in, df_out = df[~held].reset_index(drop=True), df[held].reset_index(drop=True)
                # people often record different subsets of signs: test only the signs the others recorded
                df_out = df_out[df_out["label"].isin(set(df_in["label"]))].reset_index(drop=True)
                if df_in["label"].nunique() < 2 or df_out.empty:
                    if key == model_keys[0]:
                        log(f"  skipped {signer}: the other signers did not record enough of the same signs")
                    continue
                Xi, yi, _ = feature_matrix(df_in, augment_copies=augment, seed=seed)
                Xo, yo, _ = feature_matrix(df_out, augment_copies=0)
                model = clone(fitted[key]).fit(Xi, yi)
                pred = model.predict(Xo)
                _, _, f1, _ = precision_recall_fscore_support(yo, pred, average="macro", zero_division=0)
                loso_results[key][signer] = {"accuracy": round(float(np.mean(pred == yo)), 4),
                                             "macro_f1": round(float(f1), 4), "samples": int(len(yo))}
            if loso_results[key]:
                mean_acc = np.mean([v["accuracy"] for v in loso_results[key].values()])
                log(f"  {plots.MODEL_NAMES[key]}: mean accuracy on unseen signers {mean_acc:.3f}")
        loso_results = {k: v for k, v in loso_results.items() if v}

    importance = []
    if "rf" in fitted:
        order = np.argsort(fitted["rf"].feature_importances_)[::-1][:20]
        importance = [{"feature": FEATURE_NAMES[i], "importance": round(float(fitted["rf"].feature_importances_[i]), 5)}
                      for i in order]

    # Final models: retrain on ALL data (the held-out evaluation above is what we report).
    log("Retraining on all data for deployment ...")
    X_all, y_all, _ = feature_matrix(df, augment_copies=augment, seed=seed)
    models_dir.mkdir(parents=True, exist_ok=True)
    saved = {}
    for key in model_keys:
        final = clone(fitted[key]).fit(X_all, y_all)
        path = models_dir / f"model_{key}.joblib"
        joblib.dump(make_bundle(final, key, df, results[key]), path)
        saved[key] = str(path)
        if key == best:
            joblib.dump(make_bundle(final, key, df, results[key]), models_dir / MODEL_PATH.name)

    figures = {
        "comparison": plots.model_comparison(
            {k: {"accuracy": r["accuracy"], "macro_f1": r["macro_f1"], "cv_macro_f1": r["cv"]["macro_f1_mean"]}
             for k, r in results.items()}, figures_dir / "model_comparison.png"),
        "latency": plots.latency_chart({k: r["latency_ms"] for k, r in results.items()},
                                       figures_dir / "prediction_latency.png"),
        "per_class": plots.horizontal_bars(
            {label: v["f1"] for label, v in results[best]["per_class"].items()},
            f"F1 score per sign - {results[best]['name']}", "F1 score (held-out bursts)",
            figures_dir / "per_class_f1.png", colour=plots.MODEL_COLOURS[best]),
    }
    for key, r in results.items():
        figures[f"confusion_{key}"] = plots.confusion_heatmap(
            np.array(r["confusion"]["normalised"]), labels, f"Confusion matrix - {r['name']} (row %)",
            figures_dir / f"confusion_{key}.png")
    if loso_results:
        figures["loso"] = plots.grouped_by_signer(
            {k: {s: v["accuracy"] for s, v in per.items()} for k, per in loso_results.items()},
            figures_dir / "leave_one_signer_out.png")
    if importance:
        figures["importance"] = plots.horizontal_bars(
            {item["feature"]: item["importance"] for item in importance},
            "Top 20 features (Random Forest importance)", "Mean decrease in impurity",
            figures_dir / "feature_importance.png", percent=False)

    for r in results.values():
        r.pop("_predicted", None)
    metrics_doc = {
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_path": str(data_path),
        "dataset": stats,
        "labels": labels,
        "split": {"strategy": "burst-level hold-out", "test_size": test_size, "train_samples": len(df_train),
                  "test_samples": len(df_test), "single_burst_labels": single_burst, "seed": seed},
        "augment_copies": augment,
        "tuned": tune,
        "models": results,
        "best_model": best,
        "groups": groups,
        "loso": loso_results,
        "feature_importance": importance,
        "figures": {k: str(Path(v).relative_to(reports_dir)) if Path(v).is_relative_to(reports_dir) else v
                    for k, v in figures.items()},
        "saved_models": saved,
        "seconds": round(time.perf_counter() - started_all, 1),
    }
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / METRICS_PATH.name).write_text(json.dumps(metrics_doc, indent=2), encoding="utf-8")
    (reports_dir / "training_summary.md").write_text(summary_markdown(metrics_doc), encoding="utf-8")
    log(f"Saved best model to {models_dir / MODEL_PATH.name} and the report to {reports_dir}")
    return metrics_doc


def summary_markdown(doc: dict) -> str:
    lines = [f"# Training summary ({doc['created']})", "",
             f"* Samples: {doc['dataset']['samples']} - signs: {len(doc['labels'])} - signers: "
             f"{', '.join(doc['dataset']['signers'])}",
             f"* Split: {doc['split']['strategy']} ({doc['split']['train_samples']} train / "
             f"{doc['split']['test_samples']} test), augmentation x{doc['augment_copies']}",
             f"* Best model: **{doc['models'][doc['best_model']]['name']}**", "",
             "| Model | Test accuracy | Test macro-F1 | CV macro-F1 (mean ± sd) | ms / frame | Train s |",
             "|---|---|---|---|---|---|"]
    for r in doc["models"].values():
        lines.append(f"| {r['name']} | {r['accuracy']:.3f} | {r['macro_f1']:.3f} | "
                     f"{r['cv']['macro_f1_mean']:.3f} ± {r['cv']['macro_f1_std']:.3f} | "
                     f"{r['latency_ms']:.2f} | {r['train_seconds']} |")
    best = doc["models"][doc["best_model"]]
    lines += ["", f"## Per-sign results ({best['name']})", "", "| Sign | Precision | Recall | F1 | Test samples |",
              "|---|---|---|---|---|"]
    for label, v in best["per_class"].items():
        lines.append(f"| {label} | {v['precision']:.2f} | {v['recall']:.2f} | {v['f1']:.2f} | {v['support']} |")
    if best["top_confusions"]:
        lines += ["", "## Most confused signs", "", "| True sign | Predicted as | Rate |", "|---|---|---|"]
        lines += [f"| {c['true']} | {c['predicted']} | {c['rate']:.1%} |" for c in best["top_confusions"][:8]]
    if doc["loso"]:
        lines += ["", "## Leave-one-signer-out accuracy", "", "| Model | " + " | ".join(
            next(iter(doc["loso"].values())).keys()) + " |", "|---|" + "---|" * len(next(iter(doc["loso"].values())))]
        for key, per in doc["loso"].items():
            lines.append(f"| {doc['models'][key]['name']} | " + " | ".join(f"{v['accuracy']:.3f}" for v in per.values()) + " |")
    for column, values in doc["groups"].items():
        lines += ["", f"## Accuracy by {column} (held-out test, best model)", "", f"| {column} | Accuracy | Samples |",
                  "|---|---|---|"]
        lines += [f"| {k} | {v['accuracy']:.3f} | {v['samples']} |" for k, v in values.items()]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Train and compare RF / SVM / MLP sign classifiers.")
    parser.add_argument("--data", type=Path, default=RAW_DATA_PATH)
    parser.add_argument("--models", nargs="+", choices=MODEL_KEYS, default=list(MODEL_KEYS))
    parser.add_argument("--augment", type=int, default=2, help="augmented copies per training sample")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--tune", action="store_true", help="grid-search hyper-parameters (slower)")
    parser.add_argument("--no-loso", action="store_true", help="skip leave-one-signer-out evaluation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        train(args.data, model_keys=args.models, augment=args.augment, test_size=args.test_size,
              folds=args.folds, tune=args.tune, loso=not args.no_loso, seed=args.seed)
    except (FileNotFoundError, ValueError) as exc:
        sys.exit(f"Error: {exc}")


if __name__ == "__main__":
    main()
