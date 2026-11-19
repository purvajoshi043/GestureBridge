"""English -> Hindi for the optional Hindi speech output.

1. A hand-checked phrasebook (instant, offline, reliable) covers the common sentences.
2. Otherwise the local LLM translates; the answer must be mostly Devanagari.
3. If both fail the caller keeps English and tells the user.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

import requests

from ..config import PHRASEBOOK_PATH
from .guard import expand_contractions, clean_output
from .llm import OllamaClient

_DEVANAGARI = re.compile(r"[ऀ-ॿ]")
_LETTER = re.compile(r"[^\W\d_]", re.UNICODE)


def normalise(sentence: str) -> str:
    text = expand_contractions(sentence)
    text = re.sub(r"[^a-z' ]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def devanagari_ratio(text: str) -> float:
    letters = _LETTER.findall(text)
    return len(_DEVANAGARI.findall(text)) / len(letters) if letters else 0.0


@dataclass
class HindiResult:
    text: str | None
    source: str      # phrasebook | llm | none
    reason: str = ""


class HindiTranslator:
    def __init__(self, client: OllamaClient | None = None, phrasebook_path: Path = PHRASEBOOK_PATH):
        self.client = client
        data = json.loads(Path(phrasebook_path).read_text(encoding="utf-8"))
        self.phrasebook = {normalise(k): v for k, v in data["entries"].items()}

    def lookup(self, english: str, form: str = "m") -> str | None:
        entry = self.phrasebook.get(normalise(english))
        if isinstance(entry, dict):
            return entry.get(form) or entry.get("m")
        return entry

    def translate(self, english: str, form: str = "m") -> HindiResult:
        found = self.lookup(english, form)
        if found:
            return HindiResult(found, "phrasebook")
        if self.client is None or not self.client.has_model():
            return HindiResult(None, "none", "sentence not in the phrasebook and the local LLM is not available")
        speaker = "a woman (use feminine verb forms)" if form == "f" else "a man (use masculine verb forms)"
        messages = [
            {"role": "system", "content": "Translate the user's English sentence into simple, natural Hindi "
                                          f"in Devanagari script. The speaker is {speaker}. "
                                          "Keep the meaning exactly. Reply with the Hindi sentence only."},
            {"role": "user", "content": english},
        ]
        try:
            answer = clean_output(self.client.chat(messages, temperature=0.0, max_tokens=80))
        except requests.RequestException as exc:
            return HindiResult(None, "none", f"LLM translation failed: {type(exc).__name__}")
        if devanagari_ratio(answer) < 0.6:
            return HindiResult(None, "none", "LLM did not answer in Hindi")
        return HindiResult(answer, "llm", "machine translation by the local LLM - may contain mistakes")
