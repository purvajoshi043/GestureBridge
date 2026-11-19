"""Feature engineering: 21 hand landmarks -> 106 numbers that describe a handshape.

Steps
1. Pixel space   - MediaPipe x is a fraction of the image width and y of the height;
                   converting to pixels makes angles and distances correct for 640x480.
2. Normalisation - translate so the wrist is the origin, scale by palm length
                   (wrist -> middle-finger knuckle). The sign then looks the same
                   near or far from the camera and anywhere in the frame.
3. Handedness    - left hands are mirrored into a right-hand frame, so one model
                   serves left- and right-handed signers.
   Rotation is deliberately NOT removed: thumbs-up (HELP) and thumbs-down (NO)
   differ only in orientation.

Features (106)
    60  normalised x, y, z of landmarks 1-20 (the wrist is always 0)
    15  bend angle at the 3 joints of each finger (0 = straight, 1 = fully folded back)
    16  distances: 10 fingertip pairs, 5 fingertips to wrist, thumb tip to index knuckle
     5  finger extension ratio (1 = straight finger)
     6  palm normal (which way the palm faces) + hand pointing direction
     4  spread angles between neighbouring fingers
"""

from __future__ import annotations

from itertools import combinations

import numpy as np

FEATURE_VERSION = 1
NUM_LANDMARKS = 21
WRIST, INDEX_MCP, MIDDLE_MCP, PINKY_MCP = 0, 5, 9, 17
FINGERS = {
    "thumb": (1, 2, 3, 4),
    "index": (5, 6, 7, 8),
    "middle": (9, 10, 11, 12),
    "ring": (13, 14, 15, 16),
    "pinky": (17, 18, 19, 20),
}
TIPS = (4, 8, 12, 16, 20)
PALM = (0, 5, 9, 13, 17)
EPS = 1e-6


def to_pixel_space(landmarks, width: int, height: int) -> np.ndarray:
    points = np.asarray(landmarks, dtype=np.float64).reshape(NUM_LANDMARKS, 3).copy()
    points[:, 0] *= width
    points[:, 1] *= height
    points[:, 2] *= width  # MediaPipe z uses roughly the same scale as x
    return points


def palm_centre_and_size(points: np.ndarray) -> tuple[np.ndarray, float]:
    """Palm centre (pixels) and palm length, used by the stabiliser to measure hand speed."""
    centre = points[list(PALM), :2].mean(axis=0)
    return centre, float(np.linalg.norm(points[MIDDLE_MCP, :2] - points[WRIST, :2]))


def normalise_hand(points: np.ndarray, handedness: str = "Right", mirror_left: bool = True) -> np.ndarray:
    p = np.asarray(points, dtype=np.float64) - points[WRIST]
    if mirror_left and str(handedness).lower().startswith("l"):
        p[:, 0] = -p[:, 0]
    scale = np.linalg.norm(p[MIDDLE_MCP])
    if scale < EPS:
        scale = np.linalg.norm(p, axis=1).max()
    return p / max(scale, EPS)


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > EPS else np.zeros_like(v)


def _angle(u: np.ndarray, v: np.ndarray) -> float:
    cos = np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v) + EPS)
    return float(np.arccos(np.clip(cos, -1.0, 1.0)))


def joint_angles(p: np.ndarray) -> list[float]:
    values = []
    for chain in FINGERS.values():
        joints = (WRIST,) + chain
        for a, b, c in zip(joints, joints[1:], joints[2:]):
            values.append(_angle(p[b] - p[a], p[c] - p[b]) / np.pi)
    return values


def tip_distances(p: np.ndarray) -> list[float]:
    values = [float(np.linalg.norm(p[a] - p[b])) for a, b in combinations(TIPS, 2)]
    values += [float(np.linalg.norm(p[t] - p[WRIST])) for t in TIPS]
    values.append(float(np.linalg.norm(p[4] - p[INDEX_MCP])))
    return values


def finger_extension(p: np.ndarray) -> list[float]:
    values = []
    for a, b, c, d in FINGERS.values():
        path = np.linalg.norm(p[b] - p[a]) + np.linalg.norm(p[c] - p[b]) + np.linalg.norm(p[d] - p[c])
        values.append(float(np.linalg.norm(p[d] - p[a]) / (path + EPS)))
    return values


def orientation(p: np.ndarray) -> list[float]:
    normal = _unit(np.cross(p[INDEX_MCP] - p[WRIST], p[PINKY_MCP] - p[WRIST]))
    pointing = _unit(p[MIDDLE_MCP] - p[WRIST])
    return list(normal) + list(pointing)


def finger_spread(p: np.ndarray) -> list[float]:
    directions = [p[tip] - p[base] for base, *_, tip in FINGERS.values()]
    return [_angle(u, v) / np.pi for u, v in zip(directions, directions[1:])]


def extract_features(points: np.ndarray, handedness: str = "Right", mirror_left: bool = True) -> np.ndarray:
    """Features from pixel-space landmarks of shape (21, 3)."""
    p = normalise_hand(points, handedness, mirror_left)
    vector = np.concatenate([
        p[1:].ravel(),
        joint_angles(p),
        tip_distances(p),
        finger_extension(p),
        orientation(p),
        finger_spread(p),
    ])
    return np.nan_to_num(vector.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)


def features_from_landmarks(landmarks, width: int, height: int, handedness: str = "Right",
                            mirror_left: bool = True) -> np.ndarray:
    return extract_features(to_pixel_space(landmarks, width, height), handedness, mirror_left)


def _feature_names() -> list[str]:
    names = [f"{axis}{i}" for i in range(1, NUM_LANDMARKS) for axis in "xyz"]
    for finger in FINGERS:
        names += [f"bend_{finger}_{joint}" for joint in ("base", "mid", "tip")]
    tip_names = [f for f in FINGERS]
    names += [f"dist_{a}_{b}" for a, b in combinations(tip_names, 2)]
    names += [f"dist_{f}_wrist" for f in tip_names]
    names.append("dist_thumb_index_base")
    names += [f"extension_{f}" for f in FINGERS]
    names += ["palm_normal_x", "palm_normal_y", "palm_normal_z", "pointing_x", "pointing_y", "pointing_z"]
    names += [f"spread_{a}_{b}" for a, b in zip(tip_names, tip_names[1:])]
    return names


FEATURE_NAMES = _feature_names()
NUM_FEATURES = len(FEATURE_NAMES)
