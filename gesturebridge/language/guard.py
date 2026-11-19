"""Guardrails for LLM output: the LLM may polish the sentence, never change its meaning.

A refined sentence is accepted only if, compared with the rule-based draft:
  * every content word of the draft is still there (or a listed synonym),
  * no new content word appears (no hallucinated objects, places, feelings...),
  * negation is kept exactly (not/no/never... present in both or in neither),
  * the sentence type is kept (question stays a question),
  * the length stays reasonable.
Otherwise the pipeline falls back to the rule-based draft.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

CONTRACTIONS = {
    "i'm": "i am", "you're": "you are", "we're": "we are", "they're": "they are",
    "he's": "he is", "she's": "she is", "it's": "it is", "that's": "that is",
    "what's": "what is", "where's": "where is", "who's": "who is", "there's": "there is",
    "i've": "i have", "you've": "you have", "we've": "we have", "they've": "they have",
    "i'd": "i would", "you'd": "you would", "i'll": "i will", "you'll": "you will",
    "let's": "let us", "can't": "can not", "cannot": "can not", "won't": "will not",
    "shan't": "shall not",
}

NEGATIONS = {"not", "no", "never", "nothing", "nobody", "none", "neither", "nor"}

# Words that carry no new meaning: the LLM may add or remove them freely.
FUNCTION_WORDS = {
    "a", "an", "the", "some", "any", "this", "that", "these", "those",
    "to", "of", "for", "in", "on", "at", "with", "from", "by", "and", "or", "but", "so",
    "be", "do", "have", "will", "would", "can", "could", "may", "might", "shall", "should", "must",
    "please", "kindly", "it", "let",
} | NEGATIONS

# Content words that mean the same thing for this task.
SYNONYM_WORDS = [
    ("want", "like"),          # "I want water" ~ "I would like water"
    ("hello", "hi"),
    ("okay", "ok"),
    ("goodbye", "bye"),
]

IRREGULAR = {
    "am": "be", "is": "be", "are": "be", "was": "be", "were": "be", "been": "be", "being": "be",
    "go": "go", "going": "go", "goes": "go", "went": "go", "gone": "go",
    "does": "do", "did": "do", "doing": "do", "done": "do",
    "has": "have", "had": "have", "having": "have",
    "me": "i", "my": "i", "mine": "i", "myself": "i",
    "us": "we", "our": "we", "ours": "we",
    "him": "he", "his": "he", "her": "she", "hers": "she",
    "them": "they", "their": "they", "your": "you", "yours": "you",
    "ate": "eat", "eaten": "eat", "drank": "drink", "came": "come", "slept": "sleep",
    "thanks": "thank", "yes": "yes", "this": "this", "please": "please",
}

_WORD = re.compile(r"[a-z]+(?:'[a-z]+)?")
_LABEL = re.compile(r"^(sentence|output|answer|refined|final|corrected|result|english)\s*:\s*", re.I)


def expand_contractions(text: str) -> str:
    text = text.lower().replace("’", "'").replace("‘", "'")
    for short, full in CONTRACTIONS.items():
        text = re.sub(rf"\b{re.escape(short)}\b", full, text)
    return re.sub(r"n't\b", " not", text)


def stem(word: str) -> str:
    word = word.lower()
    if word in IRREGULAR:
        return IRREGULAR[word]
    for suffix, replacement in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", ""), ("s", "")):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            word = word[: -len(suffix)] + replacement
            break
    if word.endswith("e") and len(word) > 3:
        word = word[:-1]
    return word


def words(text: str) -> list[str]:
    return _WORD.findall(expand_contractions(text))


def content_stems(text: str) -> set[str]:
    stems = set()
    for w in words(text):
        s = stem(w)
        if w not in FUNCTION_WORDS and s not in FUNCTION_WORDS:
            stems.add(s)
    return stems


SYNONYMS = [{stem(w) for w in group} for group in SYNONYM_WORDS]


def has_negation(text: str) -> bool:
    return any(w in NEGATIONS for w in words(text))


def _with_synonyms(stems: set[str]) -> set[str]:
    expanded = set(stems)
    for group in SYNONYMS:  # compared as stems, like everything else
        if expanded & group:
            expanded |= group
    return expanded


def clean_output(raw: str) -> str:
    """First non-empty line, without labels, quotes or markdown."""
    for line in raw.strip().splitlines():
        line = line.strip().strip("*`").strip()
        if line:
            break
    else:
        return ""
    line = _LABEL.sub("", line).strip()
    if len(line) >= 2 and line[0] in "\"'" and line[-1] in "\"'":
        line = line[1:-1].strip()
    return re.sub(r"\s+", " ", line)


@dataclass
class GuardResult:
    ok: bool
    reason: str
    cleaned: str


def check_refinement(draft: str, raw_output: str) -> GuardResult:
    cleaned = clean_output(raw_output or "")
    if not cleaned:
        return GuardResult(False, "the LLM returned an empty answer", cleaned)
    if cleaned[-1] not in ".!?":
        cleaned += draft[-1] if draft and draft[-1] in ".!?" else "."

    draft_words, new_words = words(draft), words(cleaned)
    if len(new_words) > len(draft_words) + 4:
        return GuardResult(False, f"the LLM made the sentence much longer ({len(new_words)} words)", cleaned)
    if len(new_words) < max(1, len(draft_words) // 2):
        return GuardResult(False, "the LLM dropped too many words", cleaned)

    if has_negation(draft) != has_negation(cleaned):
        change = "removed" if has_negation(draft) else "added"
        return GuardResult(False, f"the LLM {change} a negation (meaning would flip)", cleaned)

    if draft.rstrip().endswith("?") != cleaned.endswith("?"):
        return GuardResult(False, "the LLM changed the sentence type (question vs statement)", cleaned)

    draft_content = content_stems(draft)
    new_content = content_stems(cleaned)
    missing = {w for w in draft_content if not (_with_synonyms({w}) & new_content)}
    if missing:
        return GuardResult(False, f"the LLM dropped meaning words: {_surface(missing, draft)}", cleaned)
    added = new_content - _with_synonyms(draft_content)
    if added:
        return GuardResult(False, f"the LLM added new meaning: {_surface(added, cleaned)}", cleaned)

    return GuardResult(True, "meaning preserved", cleaned)


def _surface(stems: set[str], text: str) -> str:
    """Show the words as written (e.g. 'there') rather than their stems ('ther')."""
    found = {stem(w): w for w in words(text)}
    return ", ".join(sorted(f"'{found.get(s, s)}'" for s in stems))
