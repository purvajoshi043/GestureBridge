"""Landmark dataset: CSV storage, loading, augmentation and feature matrices.

Only landmark numbers are stored - never camera images - which keeps the
dataset small and protects the privacy of the people who recorded it.
Raw landmarks (not features) are saved, so feature engineering can change
later without re-recording.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH
from .features import NUM_LANDMARKS, extract_features, to_pixel_space

LANDMARK_COLUMNS = [f"{axis}{i}" for i in range(NUM_LANDMARKS) for axis in "xyz"]
META_COLUMNS = ["label", "signer", "session", "burst", "condition", "handedness",
                "handedness_score", "img_w", "img_h", "timestamp"]
COLUMNS = META_COLUMNS + LANDMARK_COLUMNS


def make_row(label: str, signer: str, session: str, burst: str, condition: str, hand, timestamp: float) -> dict:
    row = {
        "label": label, "signer": signer, "session": session, "burst": burst, "condition": condition,
        "handedness": hand.handedness, "handedness_score": round(hand.score, 4),
        "img_w": hand.width, "img_h": hand.height, "timestamp": round(timestamp, 3),
    }
    row.update({col: round(float(v), 6) for col, v in zip(LANDMARK_COLUMNS, hand.landmarks.ravel())})
    return row


def append_rows(rows: list[dict], path: Path = RAW_DATA_PATH) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if new_file:
            writer.writeheader()
        writer.writerows(rows)


def load_dataset(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No dataset at {path}. Record signs first: python -m gesturebridge.collect --signer <name>")
    df = pd.read_csv(path, dtype={"label": str, "signer": str, "session": str, "burst": str, "condition": str})
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"{path} is missing columns: {missing[:5]}...")
    df = df.dropna(subset=LANDMARK_COLUMNS + ["label"]).reset_index(drop=True)
    for col in ("signer", "session", "burst", "condition", "handedness"):
        df[col] = df[col].fillna("unknown").astype(str)
    df["label"] = df["label"].str.strip().str.upper()
    return df


def remove_burst(burst: str, path: Path = RAW_DATA_PATH) -> int:
    """Delete every sample of one recording burst (undo). Returns rows removed."""
    df = pd.read_csv(path, dtype=str)
    keep = df[df["burst"] != burst]
    keep.to_csv(path, index=False)
    return len(df) - len(keep)


def points_array(df: pd.DataFrame) -> np.ndarray:
    """(n, 21, 3) landmarks in pixel space."""
    raw = df[LANDMARK_COLUMNS].to_numpy(dtype=np.float64).reshape(-1, NUM_LANDMARKS, 3)
    widths = df["img_w"].to_numpy(dtype=np.float64)
    heights = df["img_h"].to_numpy(dtype=np.float64)
    return np.stack([to_pixel_space(r, w, h) for r, w, h in zip(raw, widths, heights)])


def group_ids(df: pd.DataFrame) -> np.ndarray:
    """One id per recording burst: frames from one burst are near-duplicates and must
    never be split across train and test (that would leak and inflate accuracy)."""
    return (df["signer"] + "|" + df["session"] + "|" + df["burst"]).to_numpy()


def augment_points(points: np.ndarray, rng: np.random.Generator, rotation_deg: float = 12.0,
                   yaw_deg: float = 15.0, jitter: float = 0.02) -> np.ndarray:
    """Small random in-plane rotation, a slight turn of the hand and landmark noise."""
    p = points - points[0]
    roll = np.radians(rng.uniform(-rotation_deg, rotation_deg))
    yaw = np.radians(rng.uniform(-yaw_deg, yaw_deg))
    rz = np.array([[np.cos(roll), -np.sin(roll), 0], [np.sin(roll), np.cos(roll), 0], [0, 0, 1]])
    ry = np.array([[np.cos(yaw), 0, np.sin(yaw)], [0, 1, 0], [-np.sin(yaw), 0, np.cos(yaw)]])
    p = p @ (ry @ rz).T
    size = np.linalg.norm(p[9]) or 1.0
    p += rng.normal(0.0, jitter * size, p.shape)
    return p + points[0]


def feature_matrix(df: pd.DataFrame, mirror_left: bool = True, augment_copies: int = 0,
                   seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """X (features), y (labels), groups (burst ids); augmented copies are appended."""
    points = points_array(df)
    hands = df["handedness"].to_numpy()
    labels = df["label"].to_numpy()
    groups = group_ids(df)
    X = [extract_features(p, h, mirror_left) for p, h in zip(points, hands)]
    y, g = list(labels), list(groups)
    rng = np.random.default_rng(seed)
    for _ in range(augment_copies):
        X += [extract_features(augment_points(p, rng), h, mirror_left) for p, h in zip(points, hands)]
        y += list(labels)
        g += list(groups)
    return np.asarray(X), np.asarray(y), np.asarray(g)


def summary(df: pd.DataFrame) -> dict:
    return {
        "samples": int(len(df)),
        "labels": df["label"].value_counts().sort_index().to_dict(),
        "signers": df["signer"].value_counts().to_dict(),
        "conditions": df["condition"].value_counts().to_dict(),
        "handedness": df["handedness"].value_counts().to_dict(),
        "bursts_per_label": df.groupby("label")["burst"].nunique().to_dict(),
    }
