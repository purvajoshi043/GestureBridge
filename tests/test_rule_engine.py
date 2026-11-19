import json
import random

import pytest

from gesturebridge.config import TEST_SENTENCES_PATH
from gesturebridge.language.knowledge import KnowledgeBase
from gesturebridge.language.rule_engine import RuleEngine

CASES = json.loads(TEST_SENTENCES_PATH.read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def engine():
    return RuleEngine.load()


@pytest.mark.parametrize("case", CASES, ids=[c["glosses"] for c in CASES])
def test_test_sentences(engine, case):
    assert engine.run(case["glosses"].split()).sentence in case["expected"]


def test_trace_explains_every_step(engine):
    derivation = engine.run("WATER I WANT".split())
    assert [s.rule_id for s in derivation.trace] == ["R10", "R31"]
    for step in derivation.trace:
        assert step.explanation and "{" not in step.explanation
    assert derivation.trace[-1].result == "I want some water"


def test_negation_is_never_lost(engine):
    for glosses in ("I HUNGRY NO", "WATER NO", "I WANT FOOD NO", "HELLO NO HUNGRY NO"):
        sentence = engine.run(glosses.split()).sentence.lower()
        assert " not " in f" {sentence} " or sentence.startswith("no"), glosses


def test_unknown_gloss_passes_through(engine):
    assert engine.run(["NANCY"]).sentence == "Nancy."
    assert engine.run([]).sentence == ""


def test_terminates_on_random_input(engine):
    rng = random.Random(1)
    vocabulary = list(engine.kb.entries)
    for _ in range(500):
        derivation = engine.run([rng.choice(vocabulary) for _ in range(rng.randint(1, 6))])
        assert not derivation.hit_limit


def test_new_sign_needs_only_a_lexicon_entry():
    kb = KnowledgeBase.load()
    kb.entries["COLD"] = {"pos": "adjective", "text": "cold"}
    engine = RuleEngine.load(kb)
    assert engine.run("I COLD".split()).sentence == "I am cold."
    assert engine.run("YOU COLD QUESTION".split()).sentence == "Are you cold?"
