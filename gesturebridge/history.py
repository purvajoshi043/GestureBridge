"""Log of every translated sentence: the history panel, live-test feedback and latency data."""

from __future__ import annotations

import csv
import threading
import time
from pathlib import Path

import pandas as pd

from .config import HISTORY_PATH

FIELDS = [
    "id", "timestamp", "source", "signed", "decoded", "beam_changed", "rule_sentence",
    "final_sentence", "llm_status", "llm_reason", "hindi", "spoken_language",
    "latency_text_ms", "latency_speech_ms", "rules_fired", "feedback", "tester", "expected",
]


class HistoryLog:
    def __init__(self, path: Path = HISTORY_PATH):
        self.path = Path(path)
        self._lock = threading.Lock()

    def record(self, translation) -> None:
        result = translation.result
        if result is None:
            return
        row = {
            "id": translation.id,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(translation.created)),
            "source": translation.source,
            "signed": " ".join(e.gloss for e in translation.entries),
            "decoded": " ".join(result.glosses),
            "beam_changed": result.decode.changed,
            "rule_sentence": result.rule_sentence,
            "final_sentence": result.final_sentence,
            "llm_status": result.refine.status if result.refine else "",
            "llm_reason": result.refine.reason if result.refine else "",
            "hindi": result.hindi.text if result.hindi and result.hindi.text else "",
            "spoken_language": result.spoken_language,
            "latency_text_ms": round(translation.latency_ms.get("text_ready", 0.0), 1),
            "latency_speech_ms": round(translation.latency_ms.get("speech_start", 0.0), 1),
            "rules_fired": " ".join(step.rule_id for step in result.derivation.trace),
            "feedback": translation.feedback,
            "tester": translation.tester,
            "expected": "",
        }
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            new_file = not self.path.exists() or self.path.stat().st_size == 0
            with self.path.open("a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=FIELDS)
                if new_file:
                    writer.writeheader()
                writer.writerow(row)

    def load(self) -> pd.DataFrame:
        if not self.path.exists():
            return pd.DataFrame(columns=FIELDS)
        return pd.read_csv(self.path, dtype=str, keep_default_na=False)

    def set_feedback(self, entry_id: str, feedback: str, tester: str = "", expected: str = "") -> bool:
        with self._lock:
            df = self.load()
            mask = df["id"] == entry_id
            if not mask.any():
                return False
            df.loc[mask, "feedback"] = feedback
            if tester:
                df.loc[mask, "tester"] = tester
            if expected:
                df.loc[mask, "expected"] = expected
            df.to_csv(self.path, index=False)
            return True

    def clear(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()
