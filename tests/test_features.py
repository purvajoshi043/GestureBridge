import numpy as np

from gesturebridge.features import FEATURE_NAMES, NUM_FEATURES, extract_features, features_from_landmarks
from tests.synthetic import SHAPES, hand_points


def _pixels(points, scale=100.0, offset=(320.0, 260.0)):
    p = points * scale
    p[:, :2] += offset
    return p


def test_feature_vector_size():
    f = extract_features(_pixels(hand_points(SHAPES["OPEN"][0])))
    assert f.shape == (NUM_FEATURES,) == (len(FEATURE_NAMES),) == (106,)
    assert np.isfinite(f).all()


def test_position_and_size_invariance():
    pts = hand_points(SHAPES["VICTORY"][0])
    a = extract_features(_pixels(pts, 80, (200, 300)))
    b = extract_features(_pixels(pts, 140, (450, 180)))
    assert np.allclose(a, b, atol=1e-9)


def test_left_hand_is_mirrored_into_right_hand_frame():
    right = _pixels(hand_points(SHAPES["POINT"][0]))
    left = right.copy()
    left[:, 0] = 640 - left[:, 0]
    assert np.allclose(extract_features(right, "Right"), extract_features(left, "Left"), atol=1e-9)


def test_orientation_is_kept():
    up = extract_features(_pixels(hand_points(SHAPES["THUMB_UP"][0], roll=0)))
    down = extract_features(_pixels(hand_points(SHAPES["THUMB_DOWN"][0], roll=180)))
    assert np.linalg.norm(up - down) > 1.0


def test_aspect_ratio_is_corrected():
    landmarks = np.random.default_rng(0).uniform(0.3, 0.7, (21, 3))
    wide = features_from_landmarks(landmarks, 640, 480)
    square = features_from_landmarks(landmarks, 480, 480)
    assert not np.allclose(wide, square)


def test_degenerate_hand_gives_finite_features():
    assert np.isfinite(extract_features(np.zeros((21, 3)))).all()
