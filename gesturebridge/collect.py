"""Data collection tool: record hand-landmark samples for each sign.

    python -m gesturebridge.collect --signer nancy
    python -m gesturebridge.collect --signer purva --condition dim --samples 120

Keys
    N / Right arrow     next sign          P / Left arrow    previous sign
    SPACE               record a burst (2 s countdown, then `--samples` frames;
                        SPACE again stops early)
    U                   undo the last burst
    C                   change the lighting/condition tag
    Q / ESC             quit

Only the 21 landmark coordinates are saved (data/raw/landmarks.csv) - never images.
Tips: record every sign in at least 2-3 bursts, move the hand a little
(distance, angle, position) during a burst, and record every team member.
"""

from __future__ import annotations

import argparse
import sys
import textwrap
import time

import cv2
import numpy as np
import pandas as pd

from .config import RAW_DATA_PATH, Settings
from .dataset import append_rows, make_row, remove_burst
from .hand_tracker import BLACK_FRAME_HINT, HandTracker, draw_hand, draw_label, is_black, open_camera
from .signs import SignVocabulary

CONDITIONS = ["bright", "dim", "backlit", "outdoor", "plain-background", "busy-background"]
PANEL_W = 360
CANVAS_H = 620  # taller than a 480p frame so the sign list, key-pose and help all fit
KEY_LEFT, KEY_RIGHT = {2424832, 65361, 81}, {2555904, 65363, 83}


def existing_counts(signer: str) -> tuple[dict, dict]:
    if not RAW_DATA_PATH.exists():
        return {}, {}
    df = pd.read_csv(RAW_DATA_PATH, usecols=["label", "signer"], dtype=str)
    return df[df["signer"] == signer]["label"].value_counts().to_dict(), df["label"].value_counts().to_dict()


def _text(img: np.ndarray, text: str, x: int, y: int, colour, right: bool = False, scale: float = 0.45) -> None:
    if right:
        x -= cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)[0][0]
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, colour, 1, cv2.LINE_AA)


def draw_panel(height: int, vocab: SignVocabulary, index: int, mine: dict, total: dict, signer: str,
               condition: str, status: str) -> np.ndarray:
    panel = np.full((height, PANEL_W, 3), 32, dtype=np.uint8)
    y = 28
    cv2.putText(panel, f"Signer: {signer}   [{condition}]", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (230, 230, 230), 1, cv2.LINE_AA)
    y += 24
    _text(panel, "sign", 12, y, (150, 150, 150))
    _text(panel, "you", 240, y, (150, 150, 150), right=True)
    _text(panel, "all", 320, y, (150, 150, 150), right=True)
    for i, label in enumerate(vocab.labels):
        y += 18
        selected = i == index
        if selected:
            cv2.rectangle(panel, (6, y - 14), (PANEL_W - 6, y + 5), (70, 120, 60), -1)
        mine_n, total_n = mine.get(label, 0), total.get(label, 0)
        colour = (255, 255, 255) if selected else ((120, 200, 120) if mine_n >= 200 else (210, 210, 210))
        _text(panel, vocab.display(label), 12, y, colour)
        _text(panel, str(mine_n), 240, y, colour, right=True)
        _text(panel, str(total_n), 320, y, colour, right=True)
    sign = vocab.get(vocab.labels[index])
    y += 30
    cv2.putText(panel, "Key-pose:", (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 200, 255), 1, cv2.LINE_AA)
    for line in textwrap.wrap(sign.keypose if sign else "", 40):
        y += 18
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (230, 230, 230), 1, cv2.LINE_AA)
    y = height - 70
    for line in ("N/P: sign  SPACE: record  U: undo", "C: condition  Q: quit", status):
        cv2.putText(panel, line, (12, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)
        y += 20
    return panel


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    settings = Settings.load()
    parser = argparse.ArgumentParser(description="Record hand-landmark samples for each sign.")
    parser.add_argument("--signer", required=True, help="name of the person signing (used for per-signer evaluation)")
    parser.add_argument("--condition", default="bright", help="lighting/background tag, e.g. bright, dim")
    parser.add_argument("--samples", type=int, default=100, help="frames per recording burst")
    parser.add_argument("--label", help="start at this sign")
    parser.add_argument("--camera", type=int, default=settings.camera.index)
    args = parser.parse_args(argv)

    vocab = SignVocabulary.load()
    signer = args.signer.strip().lower().replace(" ", "_")
    condition = args.condition
    index = vocab.labels.index(args.label.upper()) if args.label and args.label.upper() in vocab.labels else 0
    session = time.strftime("%Y%m%d-%H%M%S")
    mine, total = existing_counts(signer)
    burst_no, last_burst, last_burst_label = 0, None, None
    status = "Only landmark numbers are saved - no images."

    cap = open_camera(args.camera, settings.camera.width, settings.camera.height)
    tracker = HandTracker(max_hands=1, min_detection_confidence=settings.recognition.min_detection_confidence)
    window = "GestureBridge - data collection"
    cv2.namedWindow(window)
    print(f"Recording for signer '{signer}'. Saving to {RAW_DATA_PATH}. Only landmarks are stored, no images.")

    recording, countdown_until, rows = False, 0.0, []
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame not available.")
                break
            if settings.camera.mirror:
                frame = cv2.flip(frame, 1)
            hands = tracker.process(frame)
            hand = hands[0] if hands else None
            label = vocab.labels[index]
            now = time.time()

            if hand:
                draw_hand(frame, hand, (90, 220, 120) if not recording else (60, 60, 230))
            if countdown_until and now < countdown_until:
                cv2.putText(frame, str(int(countdown_until - now) + 1), (frame.shape[1] // 2 - 30, frame.shape[0] // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 3, (60, 220, 255), 6, cv2.LINE_AA)
                draw_label(frame, f"Get ready: {vocab.display(label)}", (10, 30), (60, 220, 255))
            elif countdown_until:
                countdown_until, recording, rows = 0.0, True, []
                burst_no += 1
                last_burst = f"{signer}-{session}-{burst_no:03d}"
            if recording:
                if hand:
                    rows.append(make_row(label, signer, session, last_burst, condition, hand, now))
                progress = len(rows) / args.samples
                cv2.rectangle(frame, (10, frame.shape[0] - 30), (10 + int((frame.shape[1] - 20) * progress),
                              frame.shape[0] - 14), (60, 60, 230), -1)
                draw_label(frame, f"REC {vocab.display(label)}  {len(rows)}/{args.samples}"
                           + ("" if hand else "  (no hand!)"), (10, 30), (60, 60, 230))
                if len(rows) >= args.samples:
                    recording = False
            if not recording and rows:
                append_rows(rows)
                mine[label] = mine.get(label, 0) + len(rows)
                total[label] = total.get(label, 0) + len(rows)
                last_burst_label = label
                status = f"Saved {len(rows)} samples of {vocab.display(label)}."
                print(status)
                rows = []
            if not recording and not countdown_until and not hand:
                draw_label(frame, "Show your hand to the camera", (10, 30), (170, 170, 170))
            if is_black(frame):
                status = BLACK_FRAME_HINT[:52]

            height = max(CANVAS_H, frame.shape[0])
            canvas = np.full((height, frame.shape[1] + PANEL_W, 3), 24, dtype=np.uint8)
            canvas[:frame.shape[0], :frame.shape[1]] = frame
            canvas[:, frame.shape[1]:] = draw_panel(height, vocab, index, mine, total, signer, condition, status)
            cv2.imshow(window, canvas)
            key = cv2.waitKeyEx(1)
            if key == -1:
                continue
            char = key & 0xFF
            if char in (ord("q"), 27):
                if recording and rows:
                    recording = False
                    append_rows(rows)
                break
            if char == ord(" "):
                if recording:
                    recording = False  # stop early; rows are saved on the next frame
                elif not countdown_until:
                    countdown_until = time.time() + 2.0
            elif not recording and not countdown_until:
                if char == ord("n") or key in KEY_RIGHT:
                    index = (index + 1) % len(vocab.labels)
                elif char == ord("p") or key in KEY_LEFT:
                    index = (index - 1) % len(vocab.labels)
                elif char == ord("c"):
                    condition = CONDITIONS[(CONDITIONS.index(condition) + 1) % len(CONDITIONS)] \
                        if condition in CONDITIONS else CONDITIONS[0]
                    status = f"Condition: {condition}"
                elif char == ord("u") and last_burst:
                    removed = remove_burst(last_burst)
                    mine[last_burst_label] = max(0, mine.get(last_burst_label, 0) - removed)
                    total[last_burst_label] = max(0, total.get(last_burst_label, 0) - removed)
                    status = f"Undid last burst ({removed} samples)."
                    print(status)
                    last_burst = None
    finally:
        cap.release()
        tracker.close()
        cv2.destroyAllWindows()

    print("\nSamples for this signer:")
    for label in vocab.labels:
        print(f"  {vocab.display(label):<12} {mine.get(label, 0)}")
    print("\nNext: python -m gesturebridge.train")


if __name__ == "__main__":
    main()
