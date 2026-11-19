"""Turning noisy per-frame predictions into reliable sign decisions.

A single frame is unreliable (motion blur, half-formed handshapes while moving
between signs). The stabiliser reasons over time:

* Smoothing     average the class probabilities of the last `window` frames
                (approximate inference: the evidence of many frames is combined).
* Hold          the same sign must stay on top for `hold_seconds`.
* Stillness     hand speed (palm lengths per second) must be low: signs are
                read while the hand is held, not while it travels between signs.
* Threshold     accept only if the smoothed probability >= `threshold`.
* Uncertainty   a new sign held still for `hold_seconds` with a probability only
                between `uncertain_threshold` and `threshold` -> an "uncertain"
                event (the UI asks to repeat the sign).
* Cooldown      no new sign for `cooldown_seconds` after an accepted one.
* Release       the same sign can only be accepted again after the hand has shown
                something else (or left the frame) for `release_seconds`, so a one-frame
                flicker while a sign is held never adds it twice.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np

from .config import RecognitionSettings

MIN_STREAK_FRAMES = 3


@dataclass
class StabiliserOutput:
    hand_present: bool
    top: list[tuple[str, float]] = field(default_factory=list)   # smoothed top-k
    label: str | None = None
    confidence: float = 0.0
    margin: float = 0.0
    entropy: float = 0.0          # 0 = certain, 1 = completely unsure
    hold_progress: float = 0.0
    still: bool = True
    speed: float = 0.0
    cooling_down: bool = False
    event: str | None = None      # "accepted" | "uncertain" | None
    event_label: str | None = None


class PredictionStabiliser:
    def __init__(self, classes: list[str], settings: RecognitionSettings | None = None,
                 ignore_labels: tuple[str, ...] = ("NOTHING",)):
        self.classes = list(classes)
        self.settings = settings or RecognitionSettings()
        self.ignore_labels = set(ignore_labels)
        self.reset()

    def reset(self) -> None:
        self._window: deque = deque(maxlen=max(1, int(self.settings.window)))
        self._streak_label: str | None = None
        self._streak_frames = 0
        self._streak_start = 0.0
        self._band_start: float | None = None   # when the sign entered the "unsure" band
        self._uncertain_sent = False
        self._last_uncertain = -1e9
        self._cooldown_until = -1e9
        self._blocked_label: str | None = None
        self._release_start: float | None = None
        self._prev_centre: np.ndarray | None = None
        self._prev_t = 0.0
        self._speed = 0.0

    def configure(self, settings: RecognitionSettings) -> None:
        resize = int(settings.window) != self._window.maxlen
        self.settings = settings
        if resize:
            self._window = deque(self._window, maxlen=max(1, int(settings.window)))

    def _count_release(self, label: str | None, t: float) -> None:
        if self._blocked_label is None:
            return
        if label == self._blocked_label:
            self._release_start = None
            return
        if self._release_start is None:
            self._release_start = t
        if t - self._release_start >= self.settings.release_seconds:
            self._blocked_label = None

    def update(self, probs: np.ndarray | None, t: float, centre: np.ndarray | None = None,
               size: float | None = None) -> StabiliserOutput:
        s = self.settings
        if probs is None:  # no hand in view
            self._window.clear()
            self._streak_label, self._streak_frames = None, 0
            self._prev_centre, self._speed = None, 0.0
            self._count_release(None, t)
            return StabiliserOutput(hand_present=False, cooling_down=t < self._cooldown_until)

        self._window.append(np.asarray(probs, dtype=np.float64))
        avg = np.mean(self._window, axis=0)
        order = np.argsort(avg)[::-1]
        top = [(self.classes[i], float(avg[i])) for i in order[: max(1, s.top_k)]]
        label, confidence = top[0]
        margin = confidence - (top[1][1] if len(top) > 1 else 0.0)
        nonzero = avg[avg > 0]
        entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(len(avg))) if len(avg) > 1 else 0.0

        # Hand speed in palm-lengths per second, smoothed with an exponential moving average.
        if centre is not None and size and self._prev_centre is not None and t > self._prev_t:
            speed = float(np.linalg.norm(np.asarray(centre) - self._prev_centre) / size / (t - self._prev_t))
            self._speed = 0.5 * self._speed + 0.5 * speed
        if centre is not None:
            self._prev_centre, self._prev_t = np.asarray(centre, dtype=np.float64), t
        still = (not s.motion_gate) or self._speed <= s.motion_threshold

        if label != self._streak_label:
            self._streak_label, self._streak_frames, self._streak_start = label, 1, t
            self._uncertain_sent = False
            self._band_start = None
        else:
            self._streak_frames += 1
        if not still:
            self._streak_start = t  # the hold timer only runs while the hand is still
        in_band = s.uncertain_threshold <= confidence < s.threshold
        if not in_band or not still:
            self._band_start = None
        elif self._band_start is None:
            self._band_start = t

        self._count_release(label, t)
        held = t - self._streak_start
        cooling = t < self._cooldown_until
        ignored = label in self.ignore_labels
        progress = 0.0 if ignored or not still or label == self._blocked_label \
            else min(1.0, held / s.hold_seconds) if s.hold_seconds > 0 else 1.0

        out = StabiliserOutput(True, top, label, confidence, margin, entropy, progress, still,
                               self._speed, cooling)
        ready = (not ignored and still and not cooling and self._streak_frames >= MIN_STREAK_FRAMES
                 and held >= s.hold_seconds)
        if not ready:
            return out

        if confidence >= s.threshold and label != self._blocked_label:
            out.event, out.event_label = "accepted", label
            self._cooldown_until = t + s.cooldown_seconds
            self._blocked_label, self._release_start = label, None
            self._streak_start = t
            out.hold_progress = 0.0
        elif (in_band and label != self._blocked_label and self._band_start is not None
              and t - self._band_start >= s.hold_seconds and not self._uncertain_sent
              and t - self._last_uncertain > 2.5):
            out.event, out.event_label = "uncertain", label
            self._uncertain_sent = True
            self._last_uncertain = t
        return out
