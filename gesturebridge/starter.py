"""Starter mode: a live demo before your own sign model is trained.

MediaPipe's pretrained Gesture Recognizer knows 7 fixed hand gestures. They are
mapped to GestureBridge signs below, and the recogniser replaces both the hand
tracker and the classifier - the rest of the pipeline (stabiliser, beam search,
rules, LLM, speech) runs unchanged.

This model was NOT trained by us. Switch to your own model (recorded with
`python -m gesturebridge.collect`, trained with `python -m gesturebridge.train`)
as soon as you have one.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from .config import MODELS_DIR
from .hand_tracker import HandObservation, ensure_hand_model

STARTER_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/gesture_recognizer/"
                     "gesture_recognizer/float16/latest/gesture_recognizer.task")
STARTER_MODEL_PATH = MODELS_DIR / "gesture_recognizer.task"

# MediaPipe gesture -> GestureBridge sign (the same handshapes as docs/SIGN_GUIDE.md where possible)
GESTURE_TO_SIGN = {
    "Open_Palm": "HELLO",
    "Pointing_Up": "I",
    "Thumb_Up": "HELP",
    "Thumb_Down": "NO",
    "Closed_Fist": "YES",
    "ILoveYou": "THANK_YOU",
    "Victory": "SPEAK",
    "None": "NOTHING",
}
GESTURE_HELP = [
    ("✋", "Open palm", "HELLO"),
    ("☝️", "Index finger up", "I"),
    ("👍", "Thumbs up", "HELP"),
    ("👎", "Thumbs down", "NO"),
    ("✊", "Closed fist", "YES"),
    ("🤟", "Thumb + index + little finger", "THANK YOU"),
    ("✌️", "V sign", "SPEAK - finish and speak the sentence"),
]


class GestureTracker:
    """Drop-in replacement for HandTracker that also remembers the recognised gesture."""

    def __init__(self, min_detection_confidence: float = 0.6, min_tracking_confidence: float = 0.5):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        options = vision.GestureRecognizerOptions(
            base_options=mp_python.BaseOptions(
                model_asset_path=str(ensure_hand_model(STARTER_MODEL_PATH, STARTER_MODEL_URL))),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._recognizer = vision.GestureRecognizer.create_from_options(options)
        self._last_ts = -1
        self.last_gesture: tuple[str, float] | None = None

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[HandObservation]:
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        ts = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
        ts = max(ts, self._last_ts + 1)
        self._last_ts = ts
        result = self._recognizer.recognize_for_video(
            self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb), ts)
        self.last_gesture = None
        if not result.hand_landmarks:
            return []
        if result.gestures and result.gestures[0]:
            top = result.gestures[0][0]
            self.last_gesture = (top.category_name, float(top.score))
        landmarks = np.array([[lm.x, lm.y, lm.z] for lm in result.hand_landmarks[0]], dtype=np.float64)
        category = result.handedness[0][0]
        return [HandObservation(landmarks, category.category_name, float(category.score), width, height)]

    def close(self) -> None:
        self._recognizer.close()


class StarterClassifier:
    """Turns the recogniser's top gesture into class probabilities for the stabiliser."""

    model_name = "starter"
    mirror_left = True

    def __init__(self, tracker: GestureTracker):
        self.tracker = tracker
        self.classes = sorted(set(GESTURE_TO_SIGN.values()))

    @property
    def description(self) -> str:
        return "STARTER - MediaPipe's pretrained gesture model (7 gestures), not trained by us"

    def predict_proba(self, features) -> np.ndarray:
        name, score = self.tracker.last_gesture or ("None", 1.0)
        label = GESTURE_TO_SIGN.get(name, "NOTHING")
        # The canned model's scores are conservative (~0.7-0.9 for clear gestures);
        # the square root spreads them over the stabiliser's acceptance range.
        p = float(np.sqrt(np.clip(score, 0.0, 1.0)))
        probs = np.full(len(self.classes), (1.0 - p) / (len(self.classes) - 1))
        probs[self.classes.index(label)] = p
        return probs

    def predict(self, features) -> tuple[str, float]:
        probs = self.predict_proba(features)
        best = int(np.argmax(probs))
        return self.classes[best], float(probs[best])


def use_starter(agent) -> None:
    """Switch a GestureBridgeAgent to starter mode."""
    if agent.tracker is not None:
        agent.tracker.close()
    r = agent.settings.recognition
    agent.tracker = GestureTracker(r.min_detection_confidence, r.min_tracking_confidence)
    agent.set_classifier(StarterClassifier(agent.tracker))


def use_trained(agent, classifier) -> None:
    """Switch back to the normal hand tracker + a trained classifier."""
    if isinstance(agent.tracker, GestureTracker):
        agent.tracker.close()
        agent.tracker = None
    agent.set_classifier(classifier)
