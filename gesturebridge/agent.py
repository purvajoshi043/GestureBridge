"""GestureBridge as an intelligent agent.

PEAS description
    Performance  sign accuracy, sentence correctness, response time (sign -> speech)
    Environment  one signer in front of a webcam, varying light/background/distance
    Actuators    screen (skeleton, words, sentence, rule trace) and speaker (speech)
    Sensors      webcam frames (-> 21 hand landmarks per frame)

Agent type: model-based agent. It keeps internal state that the current frame
alone does not show - the smoothing window, the hold timer and the word buffer -
and acts on it: SENSE (perceive) -> THINK (classify, stabilise, decode, infer,
refine) -> ACT (display, speak).
"""

from __future__ import annotations

import functools
import threading
import time
import uuid
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import Settings
from .features import extract_features, palm_centre_and_size
from .hand_tracker import BLACK_FRAME_HINT, HandObservation, HandTracker, draw_hand, draw_label, is_black
from .handshape_rules import spelled_character
from .history import HistoryLog
from .pipeline import LanguagePipeline, TranslationResult
from .recognizer import SignClassifier
from .signs import SignVocabulary
from .speech.tts import Speaker, SpeechResult
from .stabiliser import PredictionStabiliser, StabiliserOutput


SPELLING_COMMIT_SECONDS = 1.5


def _locked(method):
    """The camera thread and the web page both change the word buffer: one at a time."""
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        with self.lock:
            return method(self, *args, **kwargs)
    return wrapper


@dataclass
class BufferEntry:
    gloss: str
    prob: float
    candidates: list[tuple[str, float]]
    t: float
    source: str = "sign"          # sign | manual


@dataclass
class Translation:
    id: str
    created: float
    source: str                    # gesture | button | simulator
    entries: list[BufferEntry]
    result: TranslationResult | None = None
    status: str = "pending"        # pending | done | error
    error: str = ""
    speech: SpeechResult | None = None
    audio: tuple[bytes, str] | None = None     # speech for the browser to play (data, mime type)
    audio_error: str = ""
    latency_ms: dict = field(default_factory=dict)
    feedback: str = ""
    tester: str = ""
    done: threading.Event = field(default_factory=threading.Event)


@dataclass
class FrameResult:
    frame: np.ndarray | None
    hand: HandObservation | None
    state: StabiliserOutput
    events: list[tuple[str, str]]
    translation: Translation | None = None


class GestureBridgeAgent:
    def __init__(self, settings: Settings, classifier: SignClassifier | None = None,
                 pipeline: LanguagePipeline | None = None, speaker: Speaker | None = None,
                 history: HistoryLog | None = None, vocabulary: SignVocabulary | None = None,
                 tracker: HandTracker | None = None):
        self.settings = settings
        self.vocab = vocabulary or SignVocabulary.load()
        self.classifier = classifier
        self.pipeline = pipeline or LanguagePipeline(settings.language)
        self.speaker = speaker
        self.history = history
        self.tracker = tracker
        self.stabiliser = (PredictionStabiliser(classifier.classes, settings.recognition, self.vocab.ignored)
                           if classifier else None)
        self.buffer: list[BufferEntry] = []
        self.messages: deque = deque(maxlen=40)
        self.translations: list[Translation] = []
        self.last_state = StabiliserOutput(hand_present=False)
        self.black_frames = 0
        self.lock = threading.RLock()
        self.spelling: list[str] = []          # fingerspelled letters / digits of the current word
        self._last_hand_t = 0.0
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gesturebridge")

    # -- configuration -------------------------------------------------------------
    def configure(self, settings: Settings) -> None:
        self.settings = settings
        if self.stabiliser:
            self.stabiliser.configure(settings.recognition)
        self.pipeline.configure(settings.language)
        if self.speaker:
            self.speaker.rate = int(settings.speech.rate)
            self.speaker.volume = float(settings.speech.volume)
            self.speaker.enabled = bool(settings.speech.enabled)

    def set_classifier(self, classifier: SignClassifier | None) -> None:
        """Swap the sign model (e.g. after retraining or choosing RF/SVM/MLP in the UI)."""
        self.classifier = classifier
        self.stabiliser = (PredictionStabiliser(classifier.classes, self.settings.recognition, self.vocab.ignored)
                           if classifier else None)

    def ensure_tracker(self) -> HandTracker:
        if self.tracker is None:
            r = self.settings.recognition
            self.tracker = HandTracker(max_hands=r.max_hands, min_detection_confidence=r.min_detection_confidence,
                                       min_tracking_confidence=r.min_tracking_confidence)
        return self.tracker

    @property
    def latest(self) -> Translation | None:
        return self.translations[-1] if self.translations else None

    # -- SENSE -----------------------------------------------------------------------
    def perceive(self, frame_bgr: np.ndarray) -> tuple[np.ndarray, HandObservation | None]:
        frame = cv2.flip(frame_bgr, 1) if self.settings.camera.mirror else frame_bgr
        hands = self.ensure_tracker().process(frame)
        return frame, (hands[0] if hands else None)

    def step(self, frame_bgr: np.ndarray, t: float | None = None, draw: bool = True) -> FrameResult:
        frame, hand = self.perceive(frame_bgr)
        with self.lock:
            result = self.process(hand, time.monotonic() if t is None else t)
        self.black_frames = self.black_frames + 1 if is_black(frame) else 0
        if draw:
            self.annotate(frame, hand, result.state)
            if self.black_frames >= 15:
                draw_label(frame, BLACK_FRAME_HINT[:46], (10, 30), (60, 200, 255), scale=0.5)
                draw_label(frame, BLACK_FRAME_HINT[46:].strip(" -"), (10, 58), (60, 200, 255), scale=0.5)
        result.frame = frame
        return result

    # -- THINK -----------------------------------------------------------------------
    def process(self, hand: HandObservation | None, t: float) -> FrameResult:
        events: list[tuple[str, str]] = []
        translation = None
        if self.classifier is None or self.stabiliser is None:
            state = StabiliserOutput(hand_present=hand is not None)
        elif hand is None:
            state = self.stabiliser.update(None, t)
        else:
            points = hand.points
            features = extract_features(points, hand.handedness, self.classifier.mirror_left)
            probs = self.classifier.predict_proba(features)
            centre, size = palm_centre_and_size(points)
            state = self.stabiliser.update(probs, t, centre, size)

        if hand is not None:
            self._last_hand_t = t
        elif self.spelling and t - self._last_hand_t > SPELLING_COMMIT_SECONDS:
            self.commit_spelling(events)     # taking the hand away finishes a spelled word

        if state.event == "accepted":
            translation = self._on_sign(state, t, events)
        elif state.event == "uncertain":
            options = " or ".join(self.vocab.display(label) for label, _ in state.top[:2])
            self._note(f"Not sure - {options}? Please repeat the sign more clearly.", "warning", events)
            if self.settings.speech.voice_prompt_on_uncertain and self.speaker:
                self.speaker.speak("Please repeat the sign.")
        self.last_state = state
        return FrameResult(None, hand, state, events, translation)

    def _on_sign(self, state: StabiliserOutput, t: float, events: list) -> Translation | None:
        label = state.event_label
        char = spelled_character(label)
        if char is not None:
            self.spelling.append(char)
            self._note(f"Spelled {char} → {''.join(self.spelling)}", "success", events)
            return None
        if self.vocab.kind(label) == "control":
            if label == "SPEAK":
                return self.finalize(source="gesture")
            if label == "CLEAR":
                self.clear()
                self._note("Cleared the sentence (CLEAR sign).", "info", events)
            elif label == "DELETE":
                removed = self.delete_last()
                self._note(f"Deleted {self.vocab.display(removed)}." if removed else "Nothing to delete.",
                           "info", events)
            return None
        candidates = self._word_candidates(state.top) or [(label, state.confidence)]
        self.buffer.append(BufferEntry(label, state.confidence, candidates, t))
        self._note(f"Added {self.vocab.display(label)} ({state.confidence:.0%}).", "success", events)
        return None

    def _word_candidates(self, top: list[tuple[str, float]]) -> list[tuple[str, float]]:
        """Top-k word alternatives (controls and NOTHING removed), renormalised for the decoder."""
        words = [(label, p) for label, p in top if self.vocab.is_word(label)]
        total = sum(p for _, p in words)
        return [(label, p / total) for label, p in words] if total > 0 else []

    # -- ACT -------------------------------------------------------------------------
    @_locked
    def add_gloss(self, gloss: str, source: str = "manual") -> None:
        gloss = gloss.strip().upper()
        if gloss:
            self.buffer.append(BufferEntry(gloss, 1.0, [(gloss, 1.0)], time.monotonic(), source))

    @_locked
    def commit_spelling(self, events: list | None = None) -> str | None:
        """Turn the spelled letters/digits into one word of the sentence (e.g. N-A-N-C-Y -> NANCY)."""
        if not self.spelling:
            return None
        word = "".join(self.spelling)
        self.spelling.clear()
        self.buffer.append(BufferEntry(word, 1.0, [(word, 1.0)], time.monotonic(), "spelled"))
        self._note(f"Added the spelled word {word}.", "success", events)
        return word

    @_locked
    def delete_last(self) -> str | None:
        if self.spelling:
            return self.spelling.pop()
        return self.buffer.pop().gloss if self.buffer else None

    @_locked
    def clear(self) -> None:
        self.spelling.clear()
        self.buffer.clear()
        if self.stabiliser:
            self.stabiliser.reset()

    @_locked
    def finalize(self, source: str = "button") -> Translation | None:
        """Translate the word buffer: decode + rules now, LLM + speech in the background."""
        self.commit_spelling()
        words = [e for e in self.buffer if e.gloss != "QUESTION"]
        if not words:
            self._note("Nothing to translate yet - sign some words first.", "warning")
            return None
        started = time.time()
        translation = Translation(uuid.uuid4().hex[:8], started, source, list(self.buffer))
        self.buffer.clear()
        translation.result = self.pipeline.prepare([e.candidates for e in translation.entries])
        translation.latency_ms["draft_ready"] = (time.time() - started) * 1000
        self.translations = self.translations[-19:] + [translation]
        self._executor.submit(self._complete, translation, started)
        return translation

    def _complete(self, translation: Translation, started: float) -> None:
        try:
            result = self.pipeline.polish(translation.result)
            translation.latency_ms["text_ready"] = (time.time() - started) * 1000
            if self.speaker and self.settings.speech.enabled and self.settings.speech.output == "browser":
                data, info = self.speaker.synthesize(result.spoken_text, result.spoken_language)
                if data:
                    translation.audio = (data, info)
                    translation.latency_ms["speech_start"] = (time.time() - started) * 1000
                else:
                    translation.audio_error = info
            elif self.speaker and self.settings.speech.enabled:
                translation.speech = self.speaker.speak(result.spoken_text, result.spoken_language)
                speech_start = None
                if translation.speech.ok and translation.speech.request_id:
                    speech_start = self.speaker.wait_for(translation.speech.request_id, "started", timeout=5)
                speech_start = speech_start or (translation.speech.requested_at if translation.speech.ok else None)
                if speech_start:
                    translation.latency_ms["speech_start"] = (speech_start - started) * 1000
            translation.status = "done"
        except Exception as exc:  # never let a background failure kill the live loop
            translation.status = "error"
            translation.error = f"{type(exc).__name__}: {exc}"
        finally:
            if self.history:
                self.history.record(translation)
            translation.done.set()

    def speak_again(self, translation: Translation | None = None) -> SpeechResult | None:
        translation = translation or self.latest
        if not (translation and translation.result and self.speaker):
            return None
        return self.speaker.speak(translation.result.spoken_text, translation.result.spoken_language)

    # -- display helpers ---------------------------------------------------------------
    def _note(self, text: str, level: str = "info", events: list | None = None) -> None:
        self.messages.append((time.time(), text, level))
        if events is not None:
            events.append((text, level))

    def annotate(self, frame: np.ndarray, hand: HandObservation | None, state: StabiliserOutput) -> None:
        if hand is None:
            return
        label = None
        colour = (170, 170, 170)
        if state.label and state.label not in self.vocab.ignored:
            label = f"{self.vocab.display(state.label)} {state.confidence:.0%}"
            confident = state.confidence >= self.settings.recognition.threshold
            colour = (90, 220, 120) if confident else (40, 200, 240)
            if not state.still:
                label += " (moving)"
                colour = (170, 170, 170)
        draw_hand(frame, hand, colour, label)
        if state.hold_progress > 0:
            x1, _, x2, y2 = hand.bbox()
            width = int((x2 - x1) * state.hold_progress)
            cv2.rectangle(frame, (x1, y2 + 6), (x2, y2 + 14), (60, 60, 60), -1)
            cv2.rectangle(frame, (x1, y2 + 6), (x1 + width, y2 + 14), colour, -1)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
        if self.tracker:
            self.tracker.close()
            self.tracker = None


def overlay_text(frame: np.ndarray, lines: list[str], origin=(10, 28)) -> None:
    x, y = origin
    for line in lines:
        draw_label(frame, line, (x, y), (240, 240, 240), scale=0.55)
        y += 28
