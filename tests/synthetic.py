"""Synthetic hand landmarks for automated tests ONLY.

A tiny forward-kinematics hand model produces MediaPipe-like landmarks for a
few simple handshapes, so the training and agent code can be tested without a
camera. It is not a substitute for real recordings: a model trained on this
data will not recognise real hands.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from gesturebridge.dataset import COLUMNS, LANDMARK_COLUMNS

# Base of each finger (palm facing the camera, fingers pointing up = -y), palm length = 1.
BASES = {"thumb": (-0.28, -0.18), "index": (-0.24, -0.95), "middle": (0.0, -1.0),
         "ring": (0.21, -0.93), "pinky": (0.40, -0.80)}
SEGMENTS = {"thumb": (0.40, 0.32, 0.27), "index": (0.45, 0.27, 0.20), "middle": (0.50, 0.30, 0.22),
            "ring": (0.47, 0.28, 0.20), "pinky": (0.36, 0.22, 0.18)}
SPREAD_DEG = {"thumb": -50, "index": -8, "middle": 0, "ring": 7, "pinky": 16}
MAX_FLEX_DEG = (75, 95, 65)
ORDER = ("thumb", "index", "middle", "ring", "pinky")

# curl per finger (thumb, index, middle, ring, pinky) and in-plane roll (degrees)
SHAPES = {
    "OPEN":       ((0.0, 0.0, 0.0, 0.0, 0.0), 0),
    "FIST":       ((0.9, 1.0, 1.0, 1.0, 1.0), 0),
    "POINT":      ((0.9, 0.0, 1.0, 1.0, 1.0), 0),
    "VICTORY":    ((0.9, 0.0, 0.0, 1.0, 1.0), 0),
    "THUMB_UP":   ((0.0, 1.0, 1.0, 1.0, 1.0), 0),
    "THUMB_DOWN": ((0.0, 1.0, 1.0, 1.0, 1.0), 180),
    "PINKY":      ((0.9, 1.0, 1.0, 1.0, 0.0), 0),
}


def _rotation(roll: float, yaw: float, pitch: float) -> np.ndarray:
    r, y, p = np.radians([roll, yaw, pitch])
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    ry = np.array([[np.cos(y), 0, np.sin(y)], [0, 1, 0], [-np.sin(y), 0, np.cos(y)]])
    rx = np.array([[1, 0, 0], [0, np.cos(p), -np.sin(p)], [0, np.sin(p), np.cos(p)]])
    return rz @ ry @ rx


def hand_points(curls, roll=0.0, yaw=0.0, pitch=0.0, lengths=None) -> np.ndarray:
    """21 x 3 points in palm units, wrist at the origin."""
    lengths = lengths or {}
    points = np.zeros((21, 3))
    towards_camera = np.array([0.0, 0.0, -1.0])
    for f, finger in enumerate(ORDER):
        base = np.array([*BASES[finger], 0.0])
        angle = np.radians(SPREAD_DEG[finger])
        direction = np.array([np.sin(angle), -np.cos(angle), 0.0])
        bend_axis = np.array([1.0, 0.0, 0.0]) if finger == "thumb" else towards_camera
        joint = base.copy()
        points[1 + 4 * f] = joint
        flex = 0.0
        for s, length in enumerate(SEGMENTS[finger]):
            flex += np.radians(MAX_FLEX_DEG[s]) * curls[f]
            d = np.cos(flex) * direction + np.sin(flex) * bend_axis
            joint = joint + d * length * lengths.get(finger, 1.0)
            points[2 + 4 * f + s] = joint
    return points @ _rotation(roll, yaw, pitch).T


def sample_landmarks(shape: str, rng: np.random.Generator, width=640, height=480, lengths=None,
                     left=False, drift=None) -> np.ndarray:
    curls, roll = SHAPES[shape]
    curls = np.clip(np.array(curls) + rng.normal(0, 0.06, 5), 0, 1.1)
    roll = roll + rng.normal(0, 8)
    pts = hand_points(curls, roll, rng.normal(0, 12), rng.normal(0, 8), lengths)
    size = rng.uniform(70, 140) if drift is None else drift["size"]
    centre = (np.array([rng.uniform(180, 460), rng.uniform(200, 330)]) if drift is None else drift["centre"])
    px = pts * size
    px[:, :2] += centre
    px += rng.normal(0, 1.2, px.shape)
    if left:
        px[:, 0] = width - px[:, 0]
    return np.column_stack([px[:, 0] / width, px[:, 1] / height, px[:, 2] / width])


def make_dataset(shapes=None, signers=("asha", "ravi", "meera"), bursts_per_shape=2, frames_per_burst=40,
                 seed=0, left_handed=()) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    shapes = list(shapes or SHAPES)
    rows = []
    t = 1_000_000.0
    for signer in signers:
        lengths = {f: rng.uniform(0.92, 1.08) for f in ORDER}
        left = signer in left_handed
        for shape in shapes:
            for b in range(bursts_per_shape):
                drift = {"size": rng.uniform(70, 140), "centre": np.array([rng.uniform(200, 440), rng.uniform(200, 320)])}
                condition = "bright" if b % 2 == 0 else "dim"
                for _ in range(frames_per_burst):
                    drift["centre"] = drift["centre"] + rng.normal(0, 2, 2)
                    lm = sample_landmarks(shape, rng, lengths=lengths, left=left, drift=drift)
                    row = {"label": shape, "signer": signer, "session": f"{signer}-s1", "burst": f"{signer}-{shape}-{b}",
                           "condition": condition, "handedness": "Left" if left else "Right",
                           "handedness_score": 0.98, "img_w": 640, "img_h": 480, "timestamp": t}
                    row.update(dict(zip(LANDMARK_COLUMNS, lm.ravel())))
                    rows.append(row)
                    t += 0.05
    return pd.DataFrame(rows, columns=COLUMNS)
