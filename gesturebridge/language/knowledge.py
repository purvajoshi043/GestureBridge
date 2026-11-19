"""Knowledge base of facts about sign glosses (the lexicon).

Each gloss maps to facts such as ``pos(I) = pronoun``, ``person(I) = 1`` or
``mass(WATER) = true``. The rule engine and the decoder reason over these
facts, so adding a new sign only needs a lexicon entry, not new code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from ..config import LEXICON_PATH


@dataclass(eq=False)
class Token:
    """One element of the rule engine's working memory."""

    id: int
    gloss: str
    text: str
    pos: str
    feats: dict = field(default_factory=dict)
    inserted: bool = False

    def get(self, attr: str):
        if attr in ("gloss", "text", "pos", "inserted"):
            return getattr(self, attr)
        return self.feats.get(attr)

    def __repr__(self) -> str:
        return f"Token({self.gloss}->{self.text!r}:{self.pos})"


def humanise(gloss: str) -> str:
    """Surface text for a gloss that is not in the lexicon (e.g. a fingerspelled name)."""
    return gloss.replace("_", " ").title()


class KnowledgeBase:
    def __init__(self, entries: dict[str, dict]):
        self.entries = {gloss.upper(): dict(facts) for gloss, facts in entries.items()}

    @classmethod
    def load(cls, path: Path = LEXICON_PATH) -> "KnowledgeBase":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(data["entries"])

    def __contains__(self, gloss: str) -> bool:
        return gloss.upper() in self.entries

    def entry(self, gloss: str) -> dict | None:
        found = self.entries.get(gloss.upper())
        return dict(found) if found else None

    def pos(self, gloss: str) -> str:
        found = self.entries.get(gloss.upper())
        return found["pos"] if found else "unknown"

    def text(self, gloss: str) -> str:
        found = self.entries.get(gloss.upper())
        return found["text"] if found else humanise(gloss)

    def make_token(self, gloss: str, token_id: int) -> Token:
        gloss = gloss.upper()
        found = self.entries.get(gloss)
        if found is None:
            return Token(token_id, gloss, humanise(gloss), "unknown")
        feats = {k: v for k, v in found.items() if k not in ("pos", "text")}
        if found["pos"] == "noun":
            feats.setdefault("person", 3)
            feats.setdefault("number", "sg")
        return Token(token_id, gloss, found["text"], found["pos"], feats)

    def describe(self, gloss: str) -> str:
        """Human-readable facts, e.g. ``pronoun(I), person=1, number=sg``."""
        found = self.entries.get(gloss.upper())
        if found is None:
            return f"unknown({gloss})"
        extras = ", ".join(f"{k}={v}" for k, v in found.items() if k not in ("pos", "text"))
        return f"{found['pos']}({gloss})" + (f", {extras}" if extras else "")

    @property
    def tags(self) -> list[str]:
        return sorted({facts["pos"] for facts in self.entries.values()} | {"unknown"})
