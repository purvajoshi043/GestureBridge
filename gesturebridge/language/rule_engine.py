"""Forward-chaining production system that turns sign glosses into English.

The rules live in ``knowledge/grammar_rules.json`` (the knowledge), this file
is only the generic inference engine. One recognise-act cycle:

1. MATCH      every rule's IF-part is matched against working memory
              (the token list + a set of sentence facts) -> activations.
2. RESOLVE    conflict resolution picks one activation: highest salience,
              then rule order, then leftmost match. Refraction stops a rule
              from firing twice on the same tokens.
3. ACT        the THEN-part edits working memory and a trace step records
              which rule fired and why (explainable AI).

Cycles repeat until no rule can fire (a fixed point).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

from ..config import RULES_PATH
from .knowledge import KnowledgeBase, Token

MAX_CYCLES = 200
_TEMPLATE = re.compile(r"^@(\w+)\((\w+)\)$")


# ---------------------------------------------------------------------------
# Morphology helpers used by rule templates such as "@be(s)" or "@obj(p)"
# ---------------------------------------------------------------------------

def _person_number(tok: Token) -> tuple[int, str]:
    return tok.get("person") or 3, tok.get("number") or "sg"


def _be(tok: Token) -> str:
    person, number = _person_number(tok)
    if person == 1 and number == "sg":
        return "am"
    if person == 2 or number == "pl":
        return "are"
    return "is"


def _do(tok: Token) -> str:
    person, number = _person_number(tok)
    return "does" if person == 3 and number == "sg" else "do"


def _third_singular(tok: Token) -> str:
    if tok.get("s3"):
        return tok.get("s3")
    text = tok.text
    if text.endswith(("s", "sh", "ch", "x", "o")):
        return text + "es"
    return text + "s"


MORPHOLOGY = {
    "be": _be,
    "do": _do,
    "obj": lambda tok: tok.get("obj") or tok.text,
    "s3": _third_singular,
    "ing": lambda tok: tok.get("ing") or tok.text + "ing",
    "art": lambda tok: tok.get("article") or "",
    "def": lambda tok: "the" if tok.get("article") else "",
    "text": lambda tok: tok.text,
}


def expand(word: str, bindings: dict[str, Token]) -> str:
    match = _TEMPLATE.match(word)
    if not match:
        return word
    fn, ref = match.groups()
    return MORPHOLOGY[fn](bindings[ref])


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------

def _is_boundary(tok: Token) -> bool:
    return tok.pos == "punct"


def anchor_ok(tokens: list[Token], i: int, anchor: str) -> bool:
    if anchor == "first":
        return i == 0
    if anchor == "last":
        return i == len(tokens) - 1
    at_start = i == 0 or _is_boundary(tokens[i - 1])
    at_end = i == len(tokens) - 1 or _is_boundary(tokens[i + 1])
    if anchor == "start":
        return at_start
    if anchor == "end":
        return at_end
    if anchor == "only":
        return at_start and at_end
    raise ValueError(f"unknown anchor {anchor!r}")


def _value_ok(actual, expected, bindings: dict[str, Token], attr: str) -> bool:
    if isinstance(expected, dict):
        if "not" in expected:
            return not _value_ok(actual, expected["not"], bindings, attr)
        if "same_as" in expected:
            other = bindings.get(expected["same_as"])
            return other is not None and actual == other.get(attr)
        raise ValueError(f"bad constraint {expected!r}")
    if isinstance(expected, list):
        return actual in expected
    return actual == expected


def token_ok(tokens: list[Token], i: int, constraint: dict, bindings: dict[str, Token]) -> bool:
    tok = tokens[i]
    for key, expected in constraint.items():
        if key == "bind":
            continue
        if key == "at":
            if not anchor_ok(tokens, i, expected):
                return False
        elif key == "not_at":
            if anchor_ok(tokens, i, expected):
                return False
        elif not _value_ok(tok.get(key), expected, bindings, key):
            return False
    return True


@dataclass
class Rule:
    id: str
    name: str
    salience: int
    pattern: list[dict]
    actions: list[dict]
    explain: str = ""
    description: str = ""
    if_facts: list[str] = field(default_factory=list)
    unless_facts: list[str] = field(default_factory=list)
    require_any: list[dict] = field(default_factory=list)
    forbid_any: list[dict] = field(default_factory=list)
    not_preceded_by: dict | None = None
    not_followed_by: dict | None = None
    order: int = 0

    @classmethod
    def from_dict(cls, data: dict, order: int) -> "Rule":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        return cls(order=order, **known)

    def activations(self, tokens: list[Token], facts: set[str]) -> list[tuple[int, dict[str, Token]]]:
        if any(f not in facts for f in self.if_facts):
            return []
        if any(f in facts for f in self.unless_facts):
            return []
        for constraint in self.require_any:
            if not any(token_ok(tokens, i, constraint, {}) for i in range(len(tokens))):
                return []
        for constraint in self.forbid_any:
            if any(token_ok(tokens, i, constraint, {}) for i in range(len(tokens))):
                return []

        found = []
        width = len(self.pattern)
        for start in range(len(tokens) - width + 1):
            bindings: dict[str, Token] = {}
            for offset, constraint in enumerate(self.pattern):
                if not token_ok(tokens, start + offset, constraint, bindings):
                    break
                bindings[constraint.get("bind", f"_{offset}")] = tokens[start + offset]
            else:
                if self.not_preceded_by and start > 0 and token_ok(tokens, start - 1, self.not_preceded_by, bindings):
                    continue
                end = start + width
                if self.not_followed_by and end < len(tokens) and token_ok(tokens, end, self.not_followed_by, bindings):
                    continue
                found.append((start, bindings))
        return found


@dataclass
class TraceStep:
    cycle: int
    rule_id: str
    rule_name: str
    matched: str
    explanation: str
    result: str

    def __str__(self) -> str:
        return f"{self.rule_id} {self.rule_name}: {self.explanation}  ->  {self.result}"


@dataclass
class Derivation:
    """Everything the engine produced for one gloss sequence."""

    glosses: list[str]
    tokens: list[Token]
    facts: set[str]
    trace: list[TraceStep]
    sentence: str
    cycles: int
    hit_limit: bool = False


class _KeepMissing(dict):
    def __missing__(self, key: str) -> str:
        return "{" + key + "}"


class RuleEngine:
    def __init__(self, knowledge: KnowledgeBase, rules: list[Rule], max_cycles: int = MAX_CYCLES):
        self.kb = knowledge
        self.rules = sorted(rules, key=lambda r: r.order)
        self.max_cycles = max_cycles

    @classmethod
    def load(cls, knowledge: KnowledgeBase | None = None, rules_path: Path = RULES_PATH) -> "RuleEngine":
        data = json.loads(Path(rules_path).read_text(encoding="utf-8"))
        rules = [Rule.from_dict(item, order) for order, item in enumerate(data["rules"])]
        return cls(knowledge or KnowledgeBase.load(), rules)

    def rule(self, rule_id: str) -> Rule:
        return next(r for r in self.rules if r.id == rule_id)

    # -- inference -----------------------------------------------------------
    def run(self, glosses: list[str]) -> Derivation:
        tokens = [self.kb.make_token(g, i) for i, g in enumerate(glosses)]
        facts: set[str] = set()
        fired: set[tuple] = set()
        trace: list[TraceStep] = []
        next_id = len(tokens)
        cycle = 0
        hit_limit = False

        while True:
            if cycle >= self.max_cycles:
                hit_limit = True
                break
            agenda = []
            for rule in self.rules:
                for start, bindings in rule.activations(tokens, facts):
                    key = (rule.id, tuple(tok.id for tok in bindings.values()))
                    if key not in fired:
                        agenda.append((-rule.salience, rule.order, start, rule, bindings, key))
            if not agenda:
                break
            agenda.sort(key=lambda item: item[:3])
            _, _, _, rule, bindings, key = agenda[0]
            fired.add(key)
            cycle += 1

            before = {name: tok.text for name, tok in bindings.items()}
            matched = " ".join(tok.gloss if not tok.inserted else tok.text for tok in bindings.values())
            added, next_id = self._act(rule, bindings, tokens, facts, next_id)

            values = _KeepMissing(before)
            values.update({f"{name}_now": tok.text for name, tok in bindings.items()})
            values["added"] = " ".join(added)
            trace.append(TraceStep(
                cycle=cycle,
                rule_id=rule.id,
                rule_name=rule.name,
                matched=matched,
                explanation=(rule.explain or rule.description).format_map(values),
                result=realise(tokens, facts, final=False),
            ))

        return Derivation(list(glosses), tokens, facts, trace, realise(tokens, facts), cycle, hit_limit)

    def _act(self, rule: Rule, bindings: dict[str, Token], tokens: list[Token],
             facts: set[str], next_id: int) -> tuple[list[str], int]:
        added: list[str] = []
        for action in rule.actions:
            op = action["op"]
            if op in ("add_fact", "remove_fact"):
                (facts.add if op == "add_fact" else facts.discard)(action["fact"])
                continue

            if op == "reorder":
                movers = [bindings[ref] for ref in action["refs"]]
                slots = sorted(_index(tokens, tok) for tok in movers)
                for slot, tok in zip(slots, movers):
                    tokens[slot] = tok
                continue

            ref = bindings[action["ref"]]
            if op in ("insert_before", "insert_after"):
                templates = action["word"].split()
                pos_spec = action.get("pos", "word")
                pos_list = pos_spec if isinstance(pos_spec, list) else [pos_spec] * len(templates)
                new_tokens = []
                for template, pos in zip(templates, pos_list):
                    word = expand(template, bindings)
                    if not word:
                        continue
                    new_tokens.append(Token(
                        id=next_id,
                        gloss=action.get("gloss", "+" + word.upper()),
                        text=word,
                        pos=pos,
                        feats=dict(action.get("feats", {})),
                        inserted=True,
                    ))
                    next_id += 1
                    added.append(word)
                at = _index(tokens, ref) + (1 if op == "insert_after" else 0)
                tokens[at:at] = new_tokens
            elif op == "delete":
                tokens.pop(_index(tokens, ref))
            elif op == "set":
                if "text" in action:
                    ref.text = expand(action["text"], bindings)
                if "pos" in action:
                    ref.pos = action["pos"]
                ref.feats.update(action.get("feats", {}))
            elif op == "move":
                i = _index(tokens, ref)
                start, end = _clause_bounds(tokens, i)
                tokens.pop(i)
                target = {"clause_start": start, "clause_end": end, "start": 0, "end": len(tokens)}[action["to"]]
                tokens.insert(target, ref)
            else:
                raise ValueError(f"unknown action {op!r} in rule {rule.id}")
        return added, next_id


def _index(tokens: list[Token], tok: Token) -> int:
    for i, item in enumerate(tokens):
        if item is tok:
            return i
    raise ValueError(f"{tok!r} is not in working memory")


def _clause_bounds(tokens: list[Token], i: int) -> tuple[int, int]:
    start = i
    while start > 0 and not _is_boundary(tokens[start - 1]):
        start -= 1
    end = i
    while end < len(tokens) - 1 and not _is_boundary(tokens[end + 1]):
        end += 1
    return start, end


# ---------------------------------------------------------------------------
# Surface realisation: tokens -> sentence string
# ---------------------------------------------------------------------------

def realise(tokens: list[Token], facts: set[str], final: bool = True) -> str:
    parts: list[str] = []
    for tok in tokens:
        text = tok.text.strip()
        if not text:
            continue
        if tok.pos == "punct":
            if parts:
                parts[-1] += text
        else:
            parts.append(text)
    sentence = " ".join(parts)
    if not final or not sentence:
        return sentence
    sentence = sentence.rstrip(",; ")
    if sentence and sentence[-1] not in ".!?":
        sentence += "?" if "question" in facts else "."
    return capitalise(sentence)


def capitalise(sentence: str) -> str:
    chars = list(sentence)
    upper_next = True
    for i, ch in enumerate(chars):
        if upper_next and ch.isalpha():
            chars[i] = ch.upper()
            upper_next = False
        elif ch in ".!?":
            upper_next = True
    return "".join(chars)
