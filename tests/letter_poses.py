"""Hand-built key-poses for the fingerspelling letters (tests ONLY).

Each pose is made from the synthetic hand model in tests/synthetic.py: finger curls,
a rotation, and an explicit thumb-tip position (the simple thumb model cannot place
the thumb precisely). They check that every letter's rule fires on its own textbook
pose and not on its look-alikes. Real hands are noisier - see docs/SIGN_GUIDE.md.
"""

from __future__ import annotations

import numpy as np

from tests.synthetic import hand_points

STRAIGHT, CURLED, BENT = 0.0, 1.0, 0.45


def _thumb_to(points: np.ndarray, tip) -> np.ndarray:
    """Put the thumb (landmarks 2-4) on a straight line from its base (1) to `tip`."""
    p = points.copy()
    base, tip = p[1], np.asarray(tip, dtype=float)
    for k, share in ((2, 0.40), (3, 0.72), (4, 1.0)):
        p[k] = base + (tip - base) * share
    return p


def _rotate(points: np.ndarray, roll_deg: float) -> np.ndarray:
    r = np.radians(roll_deg)
    rz = np.array([[np.cos(r), -np.sin(r), 0], [np.sin(r), np.cos(r), 0], [0, 0, 1]])
    return points @ rz.T


def _mid(p, a, b):
    return (p[a] + p[b]) / 2


def pose(letter: str) -> np.ndarray:
    """21 x 3 landmarks (wrist at 0, palm length about 1, palm facing the camera)."""
    i, m, r, k = STRAIGHT, CURLED, CURLED, CURLED     # default: only the index straight
    curls = {
        "A": (CURLED,) * 4, "S": (CURLED,) * 4, "T": (CURLED,) * 4, "N": (CURLED,) * 4, "M": (CURLED,) * 4,
        "E": (CURLED,) * 4,
        "B": (STRAIGHT,) * 4,
        "D": (STRAIGHT, CURLED, CURLED, CURLED), "G": (STRAIGHT, CURLED, CURLED, CURLED),
        "L": (STRAIGHT, CURLED, CURLED, CURLED), "Q": (STRAIGHT, CURLED, CURLED, CURLED),
        "Z": (STRAIGHT, CURLED, CURLED, CURLED), "X": (BENT, CURLED, CURLED, CURLED),
        "H": (STRAIGHT, STRAIGHT, CURLED, CURLED), "U": (STRAIGHT, STRAIGHT, CURLED, CURLED),
        "V": (STRAIGHT, STRAIGHT, CURLED, CURLED), "K": (STRAIGHT, STRAIGHT, CURLED, CURLED),
        "P": (STRAIGHT, STRAIGHT, CURLED, CURLED),
        "W": (STRAIGHT, STRAIGHT, STRAIGHT, CURLED),
        "I": (CURLED, CURLED, CURLED, STRAIGHT), "J": (CURLED, CURLED, CURLED, STRAIGHT),
        "Y": (CURLED, CURLED, CURLED, STRAIGHT),
    }[letter]
    i, m, r, k = curls
    p = hand_points((0.0, i, m, r, k))
    if letter in ("U", "H", "P", "K"):                 # index and middle side by side
        p[6:9] = p[10:13] + (p[5] - p[9]) * 0.35
    if letter in ("V", "K", "P"):                      # spread the index away from the middle finger
        a = -0.35 if letter == "V" else -0.6
        turn = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
        for j in (6, 7, 8):
            p[j] = p[5] + (p[j] - p[5]) @ turn.T
    out = np.array([-0.85, -0.55, 0.0])                # thumb held out to the side
    down = np.array([0.0, 0.05, 0.0])
    thumb = {
        "A": p[6] + np.array([-0.14, 0.05, 0.0]),     # up along the side of the index finger
        "S": p[10] + down,                             # across the front of the middle finger
        "T": _mid(p, 6, 10), "N": _mid(p, 10, 14), "M": _mid(p, 14, 18),
        "E": (p[8] + p[12]) / 2 + np.array([0.0, 0.10, 0.0]),
        "B": (p[5] + p[6]) / 2 + np.array([0.08, 0.0, 0.0]),   # folded in front of the index finger
        "D": p[12], "X": p[10],
        "G": out, "L": out, "Q": out, "Y": out,
        "Z": p[10], "I": p[10], "J": p[10],
        "H": p[14], "U": p[14], "V": p[14], "W": p[18],
        "K": (p[6] + p[10]) / 2 + np.array([0.0, 0.12, 0.0]), "P": (p[6] + p[10]) / 2 + np.array([0.0, 0.12, 0.0]),
    }[letter]
    p = _thumb_to(p, thumb)
    roll = {"G": 90, "H": 90, "Z": 90, "J": 90, "P": 180, "Q": 180}.get(letter, 0)
    return _rotate(p, roll)


# C, F, O and R are left out: their shapes (curved, pinched, crossed fingers) need more than
# this simple hand model can draw.
LETTERS = ("A", "B", "D", "E", "G", "H", "I", "J", "K", "L", "M", "N", "P", "Q", "S", "T", "U", "V", "W",
           "X", "Y", "Z")
NEW_LETTERS = ("J", "M", "N", "P", "Q", "T", "Z")
