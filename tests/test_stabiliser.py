import numpy as np

from gesturebridge.config import RecognitionSettings
from gesturebridge.stabiliser import PredictionStabiliser

CLASSES = ["HUNGRY", "I", "NOTHING", "WANT"]
FPS = 30


def probs(label: str, p: float = 0.9) -> np.ndarray:
    v = np.full(len(CLASSES), (1 - p) / (len(CLASSES) - 1))
    v[CLASSES.index(label)] = p
    return v


def run(stab, frames, start=0.0, centre=(320, 240), speed_px=0.0):
    """Feed (label, p) or None frames; returns the events and the next time."""
    events, t = [], start
    x = np.array(centre, dtype=float)
    for frame in frames:
        x = x + [speed_px, 0]
        out = stab.update(None if frame is None else probs(*frame), t, x, 100.0)
        if out.event:
            events.append((out.event, out.event_label))
        t += 1 / FPS
    return events, t


def make(**overrides):
    return PredictionStabiliser(CLASSES, RecognitionSettings(**overrides))


def test_accepts_after_hold_not_before():
    stab = make(hold_seconds=0.6)
    events, _ = run(stab, [("I", 0.95)] * 15)          # 0.5 s
    assert events == []
    events, _ = run(stab, [("I", 0.95)] * 10, start=0.5)
    assert events == [("accepted", "I")]


def test_holding_a_sign_does_not_repeat_it():
    events, _ = run(make(), [("I", 0.95)] * 150)       # 5 s
    assert events == [("accepted", "I")]


def test_same_sign_again_after_release():
    stab = make()
    events, t = run(stab, [("I", 0.95)] * 40 + [None] * 25 + [("I", 0.95)] * 40)
    assert events == [("accepted", "I"), ("accepted", "I")]


def test_flicker_while_holding_does_not_repeat_the_sign():
    """The live bug from 3 Nov: HELLO was added three times while being held."""
    held = [("I", 0.95)] * 30 + ([("I", 0.95)] * 25 + [("NOTHING", 0.95)] * 6) * 6   # ~7 s, 0.2 s dropouts
    events, _ = run(make(window=1), held)
    assert events == [("accepted", "I")]


def test_two_different_signs():
    events, _ = run(make(), [("I", 0.95)] * 40 + [("HUNGRY", 0.95)] * 50)
    assert events == [("accepted", "I"), ("accepted", "HUNGRY")]


def test_low_confidence_asks_to_repeat():
    events, _ = run(make(threshold=0.7, uncertain_threshold=0.4), [("WANT", 0.55)] * 60)
    assert events == [("uncertain", "WANT")]


def test_nothing_class_is_ignored():
    events, _ = run(make(), [("NOTHING", 0.99)] * 90)
    assert events == []


def test_moving_hand_is_not_read():
    events, _ = run(make(motion_gate=True, motion_threshold=1.0), [("I", 0.95)] * 90, speed_px=20.0)  # 6 hand-lengths/s
    assert events == []
    events, _ = run(make(motion_gate=False), [("I", 0.95)] * 90, speed_px=20.0)
    assert events == [("accepted", "I")]
