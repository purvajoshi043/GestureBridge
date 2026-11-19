"""One-command setup check - run it before every demo.

    python -m gesturebridge.check            (add --no-camera to skip the webcam test)

Checks packages, the MediaPipe hand model, the webcam (frames, brightness, FPS),
the trained sign model, the dataset, Ollama + the LLM, text-to-speech and the
internet connection (only needed for Hindi speech and reverse mode).
"""

from __future__ import annotations

import argparse
import importlib
import sys
import time

import numpy as np

from .config import RAW_DATA_PATH, Settings

OK, WARN, FAIL = "OK  ", "WARN", "FAIL"
results: list[tuple[str, str, str]] = []


def report(status: str, item: str, detail: str) -> None:
    results.append((status, item, detail))
    print(f"[{status}] {item}: {detail}")


def check_packages() -> None:
    for module, pip_name in (("mediapipe", "mediapipe"), ("cv2", "opencv-contrib-python"), ("sklearn", "scikit-learn"),
                             ("streamlit", "streamlit"), ("pyttsx3", "pyttsx3"), ("requests", "requests"),
                             ("gtts", "gTTS"), ("langchain_ollama", "langchain-ollama"),
                             ("speech_recognition", "SpeechRecognition")):
        try:
            version = getattr(importlib.import_module(module), "__version__", "")
            report(OK, f"package {pip_name}", version or "installed")
        except ImportError:
            optional = pip_name in ("gTTS", "langchain-ollama", "SpeechRecognition")
            report(WARN if optional else FAIL, f"package {pip_name}", f"missing: pip install {pip_name}")


def check_camera(settings: Settings, seconds: float = 3.0) -> None:
    from .hand_tracker import HandTracker, is_black, open_camera
    try:
        started = time.time()
        tracker = HandTracker()
        report(OK, "MediaPipe hand tracker", f"ready in {time.time() - started:.1f} s")
    except Exception as exc:
        report(FAIL, "MediaPipe hand tracker", f"{type(exc).__name__}: {exc}")
        return
    try:
        cap = open_camera(settings.camera.index, settings.camera.width, settings.camera.height)
    except RuntimeError as exc:
        report(FAIL, "camera", str(exc))
        tracker.close()
        return
    frames, black, hands, times = 0, 0, 0, []
    end = time.time() + seconds
    while time.time() < end:
        ok, frame = cap.read()
        if not ok:
            continue
        frames += 1
        black += is_black(frame)
        t0 = time.perf_counter()
        hands += bool(tracker.process(frame))
        times.append((time.perf_counter() - t0) * 1000)
    cap.release()
    tracker.close()
    fps = frames / seconds
    if frames == 0:
        report(FAIL, "camera", "opened but delivered no frames - close other camera apps and try again")
    elif black == frames:
        report(FAIL, "camera", "only black frames - open the privacy shutter or press the camera key (F-key "
                               "with a camera icon), close other camera apps, check Windows camera privacy settings")
    else:
        report(OK if fps >= 10 else WARN, "camera", f"{fps:.1f} FPS, {frames - black}/{frames} usable frames, "
                                                    f"hand seen in {hands} frames, MediaPipe {np.median(times):.0f} ms/frame")


def check_model(settings: Settings) -> None:
    from .recognizer import ModelNotFoundError, SignClassifier
    try:
        classifier = SignClassifier.load(settings.resolved_model_path)
        report(OK, "sign model", classifier.description)
    except ModelNotFoundError:
        report(WARN, "sign model", "not trained yet: python -m gesturebridge.collect --signer <name>, "
                                   "then python -m gesturebridge.train")
    except ValueError as exc:
        report(FAIL, "sign model", str(exc))


def check_dataset() -> None:
    if not RAW_DATA_PATH.exists():
        report(WARN, "dataset", f"{RAW_DATA_PATH} does not exist yet")
        return
    from .dataset import load_dataset, summary
    stats = summary(load_dataset())
    few = [label for label, n in stats["labels"].items() if n < 150]
    detail = f"{stats['samples']} samples, {len(stats['labels'])} signs, signers {list(stats['signers'])}"
    report(WARN if few else OK, "dataset", detail + (f"; fewer than 150 samples for {few}" if few else ""))


def check_llm(settings: Settings) -> None:
    from .language.llm import SentenceRefiner
    s = settings.language
    refiner = SentenceRefiner(s.llm_model, s.ollama_url, 60, s.llm_backend, s.prompt_style)
    available, why = refiner.availability()
    if not available:
        report(WARN, "LLM", why + " - the rule-based sentence will be used")
        return
    refiner.refine(["I", "HUNGRY"], "I am hungry.")  # loads the model into memory
    refiner._cache.clear()
    result = refiner.refine(["I", "HUNGRY"], "I am hungry.")
    report(OK if result.status in ("ok", "unchanged") else WARN, "LLM",
           f"{why}: 'I am hungry.' -> '{result.text}' ({result.status}, {result.latency_ms:.0f} ms)")


def check_tts() -> None:
    from .speech.tts import Speaker
    speaker = Speaker(volume=0.0)
    speaker.warm_up()
    ready = speaker.wait_for("warmup", "done", timeout=20)
    result = speaker.speak("Test.")
    started = speaker.wait_for(result.request_id, "started", timeout=10) if result.request_id else None
    speaker.close()
    if ready and started:
        report(OK, "text-to-speech", f"{result.backend}, speech starts after {(started - result.requested_at) * 1000:.0f} ms")
    elif result.ok:
        report(WARN, "text-to-speech", f"using fallback {result.backend}")
    else:
        report(FAIL, "text-to-speech", result.message or "no speech engine")


def check_internet() -> None:
    import requests
    try:
        requests.head("https://translate.google.com", timeout=4)
        report(OK, "internet", "available (Hindi speech and reverse mode work)")
    except requests.RequestException:
        report(WARN, "internet", "offline - English speech still works; Hindi speech and reverse mode need internet")


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="Check that everything GestureBridge needs works.")
    parser.add_argument("--no-camera", action="store_true")
    args = parser.parse_args(argv)
    settings = Settings.load()
    print(f"Python {sys.version.split()[0]} at {sys.executable}\n")
    check_packages()
    if not args.no_camera:
        check_camera(settings)
    check_model(settings)
    check_dataset()
    check_llm(settings)
    check_tts()
    check_internet()
    failed = [item for status, item, _ in results if status == FAIL]
    print("\n" + ("All essential checks passed." if not failed else f"Fix these first: {', '.join(failed)}"))


if __name__ == "__main__":
    main()
