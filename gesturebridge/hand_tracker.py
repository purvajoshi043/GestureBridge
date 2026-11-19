"""Hand landmark detection with MediaPipe Tasks (HandLandmarker) + drawing helpers.

MediaPipe is a pretrained model that only returns 21 (x, y, z) landmarks per
hand. Everything after that - features, classifier, reasoning - is ours.
"""

from __future__ import annotations

import os
import threading
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .config import HAND_MODEL_PATH
from .features import TIPS, palm_centre_and_size, to_pixel_space

HAND_MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
                  "hand_landmarker/float16/latest/hand_landmarker.task")

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),            # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),            # index
    (5, 9), (9, 10), (10, 11), (11, 12),       # middle
    (9, 13), (13, 14), (14, 15), (15, 16),     # ring
    (13, 17), (17, 18), (18, 19), (19, 20),    # little finger
    (0, 17),                                   # palm base
]


@dataclass
class HandObservation:
    landmarks: np.ndarray   # (21, 3) normalised image coordinates from MediaPipe
    handedness: str         # "Left" / "Right"
    score: float
    width: int
    height: int

    @property
    def points(self) -> np.ndarray:
        return to_pixel_space(self.landmarks, self.width, self.height)

    def palm(self) -> tuple[np.ndarray, float]:
        return palm_centre_and_size(self.points)

    def bbox(self, pad: int = 20) -> tuple[int, int, int, int]:
        xy = self.landmarks[:, :2] * [self.width, self.height]
        x1, y1 = np.maximum(xy.min(axis=0) - pad, 0).astype(int)
        x2, y2 = np.minimum(xy.max(axis=0) + pad, [self.width - 1, self.height - 1]).astype(int)
        return int(x1), int(y1), int(x2), int(y2)


def ensure_hand_model(path: Path = HAND_MODEL_PATH, url: str = HAND_MODEL_URL) -> Path:
    """Download MediaPipe's hand_landmarker.task (~7.8 MB) on first use."""
    path = Path(path)
    if path.exists() and path.stat().st_size > 1_000_000:
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".download")
    print(f"Downloading the MediaPipe hand model to {path} ...")
    urllib.request.urlretrieve(url, tmp)
    tmp.replace(path)
    return path


class HandTracker:
    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.6,
                 min_presence_confidence: float = 0.5, min_tracking_confidence: float = 0.5,
                 video_mode: bool = True, model_path: Path = HAND_MODEL_PATH):
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision

        self._mp = mp
        self.video_mode = video_mode
        options = vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(ensure_hand_model(model_path))),
            running_mode=vision.RunningMode.VIDEO if video_mode else vision.RunningMode.IMAGE,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection_confidence,
            min_hand_presence_confidence=min_presence_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)
        self._last_ts = -1

    def process(self, frame_bgr: np.ndarray, timestamp_ms: int | None = None) -> list[HandObservation]:
        height, width = frame_bgr.shape[:2]
        rgb = np.ascontiguousarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        if self.video_mode:
            # VIDEO mode needs strictly increasing timestamps.
            ts = int(time.monotonic() * 1000) if timestamp_ms is None else int(timestamp_ms)
            ts = max(ts, self._last_ts + 1)
            self._last_ts = ts
            result = self._landmarker.detect_for_video(image, ts)
        else:
            result = self._landmarker.detect(image)

        hands = []
        for landmarks, handedness in zip(result.hand_landmarks, result.handedness):
            array = np.array([[lm.x, lm.y, lm.z] for lm in landmarks], dtype=np.float64)
            category = handedness[0]
            hands.append(HandObservation(array, category.category_name, float(category.score), width, height))
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> "HandTracker":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def draw_hand(frame: np.ndarray, hand: HandObservation, colour=(90, 220, 120), label: str | None = None) -> None:
    pts = (hand.landmarks[:, :2] * [hand.width, hand.height]).astype(int)
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, tuple(pts[a]), tuple(pts[b]), colour, 2, cv2.LINE_AA)
    for i, (x, y) in enumerate(pts):
        cv2.circle(frame, (int(x), int(y)), 5 if i in TIPS else 3, (255, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(frame, (int(x), int(y)), 5 if i in TIPS else 3, colour, 1, cv2.LINE_AA)
    if label:
        x1, y1, x2, _ = hand.bbox()
        draw_label(frame, label, (x1, max(y1 - 10, 22)), colour)


def draw_label(frame: np.ndarray, text: str, origin: tuple[int, int], colour=(90, 220, 120),
               scale: float = 0.6) -> None:
    (w, h), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2)
    x, y = origin
    cv2.rectangle(frame, (x, y - h - base - 4), (x + w + 8, y + 2), colour, -1)
    cv2.putText(frame, text, (x + 4, y - base), cv2.FONT_HERSHEY_SIMPLEX, scale, (20, 20, 20), 2, cv2.LINE_AA)


BLACK_FRAME_HINT = "Camera image is black - open the privacy shutter / press the camera key, close other camera apps"


def is_black(frame: np.ndarray) -> bool:
    """A closed privacy shutter or a disabled camera delivers (almost) black frames."""
    return float(frame[::8, ::8].mean()) < 4.0


class LatestFrameCamera:
    """Reads the camera in a background thread and keeps only the newest frame.

    OpenCV queues frames; if processing a frame takes longer than the camera's frame
    interval, the queue grows and the video lags behind reality. Here old frames are
    simply dropped, so what you see is always "now".
    """

    def __init__(self, cap: cv2.VideoCapture):
        self._cap = cap
        self._lock = threading.Lock()
        self._fresh = threading.Event()
        self._frame: np.ndarray | None = None
        self._ok = False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        while self._running:
            ok, frame = self._cap.read()
            with self._lock:
                self._ok, self._frame = ok, frame
            self._fresh.set()
            if not ok:
                time.sleep(0.05)

    def read(self, timeout: float = 3.0) -> tuple[bool, np.ndarray | None]:
        """The newest frame not returned before (waits for one if necessary)."""
        if not self._fresh.wait(timeout):
            return False, None
        with self._lock:
            self._fresh.clear()
            return self._ok, self._frame

    def isOpened(self) -> bool:  # noqa: N802 - same name as cv2.VideoCapture
        return self._running and self._cap.isOpened()

    def getBackendName(self) -> str:  # noqa: N802
        return self._cap.getBackendName()

    def release(self) -> None:
        self._running = False
        self._thread.join(timeout=1.0)
        self._cap.release()


class FakeCapture:
    """Testing aid: plays the images in a folder like a 30 FPS webcam (each one for 2 seconds).
    Enabled with the environment variable GESTUREBRIDGE_FAKE_CAMERA=<folder>."""

    def __init__(self, folder: str, width: int, height: int):
        self.frames = []
        for path in sorted(Path(folder).glob("*.jpg")) + sorted(Path(folder).glob("*.png")):
            image = cv2.imread(str(path))
            if image is not None:
                scale = min(width / image.shape[1], height / image.shape[0])
                image = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)))
                canvas = np.full((height, width, 3), 40, dtype=np.uint8)
                y, x = (height - image.shape[0]) // 2, (width - image.shape[1]) // 2
                canvas[y:y + image.shape[0], x:x + image.shape[1]] = image
                self.frames.append(canvas)
        self.n = 0
        self.opened = bool(self.frames)

    def read(self):
        time.sleep(1 / 30)
        self.n += 1
        return True, self.frames[(self.n // 60) % len(self.frames)].copy()

    def isOpened(self):  # noqa: N802
        return self.opened

    def getBackendName(self):  # noqa: N802
        return "fake"

    def release(self):
        self.opened = False


def open_camera(index: int = 0, width: int = 640, height: int = 480) -> LatestFrameCamera:
    """Open a webcam; on Windows DirectShow starts much faster than the default backend."""
    if os.environ.get("GESTUREBRIDGE_FAKE_CAMERA"):
        return LatestFrameCamera(FakeCapture(os.environ["GESTUREBRIDGE_FAKE_CAMERA"], width, height))
    backends = [cv2.CAP_DSHOW, cv2.CAP_ANY] if hasattr(cv2, "CAP_DSHOW") else [cv2.CAP_ANY]
    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            return LatestFrameCamera(cap)
        cap.release()
    raise RuntimeError(f"Could not open camera {index}. Is another app using it? Try another --camera index.")
