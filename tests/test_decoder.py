import math

import pytest

from gesturebridge.language.decoder import END, BeamSearchDecoder
from gesturebridge.pipeline import parse_gloss_text


@pytest.fixture(scope="module")
def decoder():
    return BeamSearchDecoder.load()


def test_confident_signs_are_kept(decoder):
    slots = [[("I", 0.97), ("YOU", 0.02)], [("WANT", 0.9), ("HUNGRY", 0.05)], [("WATER", 0.92), ("FOOD", 0.05)]]
    result = decoder.decode(slots)
    assert result.best == ["I", "WANT", "WATER"]
    assert not result.changed


def test_grammar_breaks_a_near_tie(decoder):
    result = decoder.decode(parse_gloss_text("I WANT:0.46/HUNGRY:0.44/FOOD:0.10"))
    assert result.greedy == ["I", "WANT"]
    assert result.best == ["I", "HUNGRY"]
    assert result.corrections() == [(1, "WANT", "HUNGRY")]


def test_beam_is_sorted_and_scored(decoder):
    result = decoder.decode(parse_gloss_text("I WANT:0.5/HUNGRY:0.5"))
    scores = [h.score for h in result.beam]
    assert scores == sorted(scores, reverse=True)
    best = result.beam[0]
    assert math.isclose(best.score, best.evidence + decoder.lm_weight * best.grammar, rel_tol=1e-9)


def test_bigram_distribution_sums_to_one(decoder):
    lm = decoder.lm
    for context in ("<s>", "pronoun", "verb"):
        total = sum(lm.prob(context, tag) for tag in lm.tags) + lm.prob(context, END)
        assert math.isclose(total, 1.0, rel_tol=1e-9)


def test_parse_gloss_text():
    assert parse_gloss_text("i want water") == [[("I", 1.0)], [("WANT", 1.0)], [("WATER", 1.0)]]
    slots = parse_gloss_text("A:0.6/B")
    assert slots[0][0] == ("A", 0.6) and slots[0][1][0] == "B" and math.isclose(slots[0][1][1], 0.4)
    assert parse_gloss_text("A/B")[0] == [("A", 0.5), ("B", 0.5)]
    assert parse_gloss_text("  ") == []
