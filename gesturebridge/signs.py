"""The sign vocabulary: which labels exist, what they mean and how to perform them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import SIGNS_PATH


@dataclass(frozen=True)
class Sign:
    label: str
    type: str          # word | control | none
    keypose: str
    note: str = ""


class SignVocabulary:
    def __init__(self, signs: list[Sign]):
        self.signs = signs
        self._by_label = {s.label: s for s in signs}

    @classmethod
    def load(cls, path: Path = SIGNS_PATH) -> "SignVocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        signs = [Sign(item["label"].upper(), item["type"], item["keypose"],
                      item.get("basis") or item.get("action", "")) for item in data["signs"]]
        return cls(signs)

    @property
    def labels(self) -> list[str]:
        return [s.label for s in self.signs]

    @property
    def words(self) -> list[str]:
        return [s.label for s in self.signs if s.type == "word"]

    @property
    def controls(self) -> list[str]:
        return [s.label for s in self.signs if s.type == "control"]

    @property
    def ignored(self) -> tuple[str, ...]:
        return tuple(s.label for s in self.signs if s.type == "none")

    def get(self, label: str) -> Sign | None:
        return self._by_label.get(label.upper())

    def kind(self, label: str) -> str:
        sign = self.get(label)
        if sign:
            return sign.type
        return "none" if label.upper() == "NOTHING" else "word"  # unknown labels are treated as words

    def is_word(self, label: str) -> bool:
        return self.kind(label) == "word"

    @staticmethod
    def display(label: str) -> str:
        from .handshape_rules import display_label
        return display_label(label)
