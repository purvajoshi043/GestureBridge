"""The sign vocabulary: which labels exist, what they mean and how to perform them."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .config import CUSTOM_SIGNS_PATH, SIGNS_PATH


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
    def load(cls, path: Path = SIGNS_PATH, custom_path: Path | None = CUSTOM_SIGNS_PATH) -> "SignVocabulary":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        signs = [Sign(item["label"].upper(), item["type"], item["keypose"],
                      item.get("basis") or item.get("action", "")) for item in data["signs"]]
        if custom_path is not None:       # words the team added in the app
            from .custom_signs import load_custom
            known = {s.label for s in signs}
            signs += [Sign(item["label"], "word", item.get("how", ""), "custom word")
                      for item in load_custom(custom_path) if item["label"] not in known]
        return cls(signs)

    def add(self, sign: Sign) -> None:
        if sign.label not in self._by_label:
            self.signs.append(sign)
            self._by_label[sign.label] = sign

    @property
    def custom(self) -> list[str]:
        return [s.label for s in self.signs if s.note == "custom word"]

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
