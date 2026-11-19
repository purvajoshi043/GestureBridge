"""Desktop version of the live translator (OpenCV window) - a robust backup for the demo.

    python -m gesturebridge.desktop
    python -m gesturebridge.desktop --no-llm --camera 1

Keys
    SPACE / ENTER  translate and speak        BACKSPACE  delete last word
    C              clear words                ?  or  /    mark as a question
    R              repeat the last sentence   L          LLM on/off
    B              beam search on/off         H          English / Hindi
    T              speech on/off              ESC        quit
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time

import cv2
import numpy as np

from .agent import GestureBridgeAgent
from .config import Settings
from .hand_tracker import open_camera
from .history import HistoryLog
from .recognizer import ModelNotFoundError, SignClassifier
from .speech.tts import Speaker
from .handshape_rules import HANDSHAPES, RuleClassifier
from .starter import GESTURE_HELP, use_starter

PANEL_W = 420
CANVAS_H = 620


def _put(img, text, x, y, scale=0.5, colour=(230, 230, 230), bold=False):
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 2 if bold else 1, cv2.LINE_AA)


def _wrap(img, text, x, y, width=44, scale=0.5, colour=(230, 230, 230), bold=False, step=22) -> int:
    for line in textwrap.wrap(text, width) or [""]:
        _put(img, line, x, y, scale, colour, bold)
        y += step
    return y


def draw_panel(agent: GestureBridgeAgent, fps: float, height: int) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), 30, dtype=np.uint8)
    s, state = agent.settings, agent.last_state
    y = 28
    model = agent.classifier.model_name.upper() if agent.classifier else "no model"
    _put(panel, f"GestureBridge   {fps:4.1f} FPS   model: {model}", 12, y, 0.48, (170, 170, 170))
    flags = (f"LLM {'on' if s.language.use_llm else 'off'} | beam {'on' if s.language.use_beam else 'off'} | "
             f"{'Hindi' if s.language.output_language == 'hi' else 'English'} | speech {'on' if s.speech.enabled else 'off'}")
    _put(panel, flags, 12, y + 20, 0.45, (170, 170, 170))

    y = 82
    _put(panel, "SIGN", 12, y, 0.45, (120, 200, 255))
    if not state.hand_present:
        _put(panel, "show your hand", 12, y + 30, 0.7, (150, 150, 150))
    elif state.label:
        _put(panel, f"{agent.vocab.display(state.label)}", 12, y + 32, 0.9, (255, 255, 255), bold=True)
        _put(panel, f"{state.confidence:.0%}{'  moving' if not state.still else ''}", 300, y + 32, 0.6, (200, 200, 200))
        bar_w = PANEL_W - 24
        cv2.rectangle(panel, (12, y + 44), (12 + bar_w, y + 52), (70, 70, 70), -1)
        cv2.rectangle(panel, (12, y + 44), (12 + int(bar_w * state.hold_progress), y + 52), (90, 220, 120), -1)
        alt = "   ".join(f"{agent.vocab.display(l)} {p:.0%}" for l, p in state.top[1:3])
        _put(panel, f"also: {alt}", 12, y + 72, 0.42, (160, 160, 160))

    y = 190
    _put(panel, "WORDS", 12, y, 0.45, (120, 200, 255))
    words = " ".join(agent.vocab.display(e.gloss) for e in agent.buffer) or "-"
    y = _wrap(panel, words, 12, y + 26, 34, 0.6, (255, 255, 255), bold=True, step=24)

    y = max(y + 10, 270)
    _put(panel, "SENTENCE", 12, y, 0.45, (120, 200, 255))
    latest = agent.latest
    if latest and latest.result:
        result = latest.result
        if latest.status == "pending":
            y = _wrap(panel, result.rule_sentence + "  (refining...)", 12, y + 26, 34, 0.6, (230, 230, 230), step=24)
        else:
            y = _wrap(panel, result.final_sentence, 12, y + 26, 30, 0.7, (255, 255, 255), bold=True, step=28)
            if result.refine:
                y = _wrap(panel, f"rules: {result.rule_sentence}  |  LLM: {result.refine.status}", 12, y, 52, 0.4,
                          (160, 160, 160), step=18)
            if result.hindi and result.hindi.text:
                _put(panel, "(spoken in Hindi - see console)", 12, y, 0.42, (160, 160, 160))
                y += 18
        trace = [step.rule_id for step in result.derivation.trace]
        if trace:
            y = _wrap(panel, "rules fired: " + " ".join(trace), 12, y + 4, 52, 0.4, (140, 180, 140), step=18)

    y = height - 150
    for _, text, level in list(agent.messages)[-3:]:
        colour = (90, 200, 255) if level == "warning" else (170, 220, 170)
        y = _wrap(panel, text, 12, y, 52, 0.42, colour, step=18)

    y = height - 66
    for line in ("SPACE speak  BKSP delete  C clear  ? question", "R repeat  L llm  B beam  H hindi  T speech  ESC quit"):
        _put(panel, line, 12, y, 0.42, (150, 150, 150))
        y += 20
    return panel


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings.load()
    parser = argparse.ArgumentParser(description="GestureBridge desktop translator")
    parser.add_argument("--camera", type=int, default=settings.camera.index)
    parser.add_argument("--model", default=str(settings.resolved_model_path))
    parser.add_argument("--no-llm", action="store_true")
    parser.add_argument("--hindi", action="store_true")
    parser.add_argument("--starter", action="store_true", help="use MediaPipe's pretrained gestures (no training)")
    args = parser.parse_args(argv)
    settings.camera.index = args.camera
    settings.language.use_llm = settings.language.use_llm and not args.no_llm
    if args.hindi:
        settings.language.output_language = "hi"

    speaker = Speaker(settings.speech.rate, settings.speech.volume, settings.speech.enabled)
    speaker.warm_up()
    agent = GestureBridgeAgent(settings, None, speaker=speaker, history=HistoryLog())
    try:
        if args.starter:
            raise ModelNotFoundError("starter mode requested")
        classifier = SignClassifier.load(args.model)
        agent.set_classifier(classifier)
    except ModelNotFoundError:
        if args.starter:
            print("STARTER mode: " + ", ".join(f"{name} = {word}" for _, name, word in GESTURE_HELP))
            use_starter(agent)
        else:
            print("No trained model - INSTANT mode (19 signs from our handshape rules): "
                  + ", ".join(f"{label} = {how}" for label, (_, how) in HANDSHAPES.items()))
            agent.set_classifier(RuleClassifier())
        classifier = agent.classifier
    agent.pipeline.refiner.client.warm_up()
    print(f"Model: {classifier.description}")
    print(f"LLM: {agent.pipeline.refiner.availability()[1]}")

    print("Starting the hand tracker ...")
    agent.ensure_tracker()
    cap = open_camera(settings.camera.index, settings.camera.width, settings.camera.height)
    window = "GestureBridge"
    cv2.namedWindow(window)
    fps, frames, tick = 0.0, 0, time.time()
    printed: set[str] = set()
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not available.")
                break
            result = agent.step(frame)
            frames += 1
            if time.time() - tick >= 1.0:
                fps, frames, tick = frames / (time.time() - tick), 0, time.time()

            latest = agent.latest
            if latest and latest.status != "pending" and latest.id not in printed and latest.result:
                printed.add(latest.id)
                r = latest.result
                print(f"[{' '.join(r.glosses)}] -> {r.final_sentence}"
                      + (f"  |  {r.hindi.text}" if r.hindi and r.hindi.text else "")
                      + (f"  ({latest.latency_ms.get('speech_start', 0):.0f} ms to speech)" if latest.latency_ms.get("speech_start") else ""))

            view = result.frame
            canvas = np.full((max(CANVAS_H, view.shape[0]), view.shape[1] + PANEL_W, 3), 24, dtype=np.uint8)
            canvas[:view.shape[0], :view.shape[1]] = view
            canvas[:, view.shape[1]:] = draw_panel(agent, fps, canvas.shape[0])
            cv2.imshow(window, canvas)

            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            char = key & 0xFF
            lang = settings.language
            if char == 27:
                break
            if char in (32, 13):
                agent.finalize(source="button")
            elif char == 8:
                agent.delete_last()
            elif char == ord("c"):
                agent.clear()
            elif char in (ord("?"), ord("/")):
                agent.add_gloss("QUESTION")
            elif char == ord("r"):
                agent.speak_again()
            elif char == ord("l"):
                lang.use_llm = not lang.use_llm
            elif char == ord("b"):
                lang.use_beam = not lang.use_beam
            elif char == ord("h"):
                lang.output_language = "en" if lang.output_language == "hi" else "hi"
            elif char == ord("t"):
                settings.speech.enabled = not settings.speech.enabled
            agent.configure(settings)
    finally:
        cap.release()
        agent.shutdown()
        speaker.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
