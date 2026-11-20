"""Words the team adds themselves: you give the word, then record your own sign for it.

A custom word is stored in ``knowledge/custom_signs.json`` and becomes part of
* the sign vocabulary (it appears under "Sign to record" and in the sign guide), and
* the knowledge base (lexicon), so the grammar rules and the beam search know what
  kind of word it is (noun, verb, describing word, ...).

If the word is already in our lexicon (e.g. EAT, DOCTOR) its known facts are reused;
otherwise simple facts are made from the chosen word type.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

from .config import CUSTOM_SIGNS_PATH

# word types the user can choose -> lexicon part of speech
WORD_TYPES = {
    "noun": "Thing / person / place (noun) - e.g. mother, school",
    "verb": "Action (verb) - e.g. eat, sleep",
    "adjective": "Describing word (adjective) - e.g. happy, tired",
    "greeting": "Greeting / phrase - e.g. good morning",
}
MAX_LENGTH = 24


def normalise_label(word: str) -> str:
    """'good morning' -> 'GOOD_MORNING'. Only letters, digits and single underscores."""
    label = re.sub(r"[^A-Z0-9]+", "_", word.strip().upper()).strip("_")
    return label[:MAX_LENGTH].strip("_")


def load_custom(path: Path = CUSTOM_SIGNS_PATH) -> list[dict]:
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return [item for item in data.get("signs", []) if item.get("label")]


def _verb_forms(text: str) -> dict:
    if re.search(r"(s|sh|ch|x|z|o)$", text):
        s3 = text + "es"
    elif re.search(r"[^aeiou]y$", text):
        s3 = text[:-1] + "ies"
    else:
        s3 = text + "s"
    ing = (text[:-1] if text.endswith("e") and not text.endswith("ee") else text) + "ing"
    return {"s3": s3, "ing": ing}


def lexicon_facts(item: dict, known: dict | None = None) -> dict:
    """Knowledge-base facts for a custom word (reuses the lexicon entry if we already know the word)."""
    if known:
        return dict(known)
    text = item.get("text") or item["label"].replace("_", " ").lower()
    pos = item.get("pos", "noun")
    facts = {"pos": pos, "text": text}
    if pos == "verb":
        facts.update({"transitive": True, **_verb_forms(text)})
    return facts


def existing_reason(label: str, vocabulary_labels, handshape_labels, lexicon=None) -> str | None:
    """Why a new word cannot be added (it already exists), or None if it is new."""
    if not label:
        return "Please type a word (letters or digits)."
    if label in ("NOTHING", "QUESTION"):
        return f"'{label}' is reserved by the app."
    if label in set(vocabulary_labels) or label in set(handshape_labels):
        return (f"The word '{label.replace('_', ' ')}' already exists. To change its sign, choose it under "
                "'Sign to record' and record your own handshape.")
    if len(label) == 1 or label.startswith(("LETTER_", "DIGIT_")):
        return "Single letters and numbers already exist in Letters and Numbers mode."
    return None


def add_custom(word: str, pos: str, how: str, vocabulary_labels, handshape_labels,
               lexicon=None, path: Path = CUSTOM_SIGNS_PATH) -> tuple[dict | None, str | None]:
    """Add a word. Returns (item, None) on success or (None, reason) if it already exists."""
    label = normalise_label(word)
    taken = [item["label"] for item in load_custom(path)]
    reason = existing_reason(label, list(vocabulary_labels) + taken, handshape_labels)
    if reason:
        return None, reason
    if pos not in WORD_TYPES:
        return None, f"Unknown word type '{pos}'."
    item = {"label": label, "text": word.strip().lower(), "pos": pos,
            "how": how.strip() or "Your own handshape (see your recordings).",
            "added": time.strftime("%Y-%m-%d %H:%M")}
    known = lexicon.entry(label) if lexicon is not None else None
    if known:                      # e.g. EAT is already in the lexicon: keep its grammar facts
        item["pos"], item["text"] = known["pos"], known["text"]
    items = load_custom(path) + [item]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"_about": "Words added in the app (Record & train tab).", "signs": items},
                               indent=2, ensure_ascii=False), encoding="utf-8")
    return item, None


def remove_custom(label: str, path: Path = CUSTOM_SIGNS_PATH) -> bool:
    items = load_custom(path)
    kept = [item for item in items if item["label"] != label]
    if len(kept) == len(items):
        return False
    Path(path).write_text(json.dumps({"_about": "Words added in the app (Record & train tab).", "signs": kept},
                                     indent=2, ensure_ascii=False), encoding="utf-8")
    return True
