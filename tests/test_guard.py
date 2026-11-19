import pytest

from gesturebridge.language.guard import check_refinement, clean_output, expand_contractions


@pytest.mark.parametrize("draft, output", [
    ("I am hungry.", "I'm hungry."),
    ("I am not going home.", "I'm not going home."),
    ("I do not want food.", "I don't want food."),
    ("I want some water.", "I would like some water."),
    ("Please help me.", "Help me, please."),
    ("Are you hungry?", "Are you hungry?"),
])
def test_accepts_meaning_preserving_rewrites(draft, output):
    assert check_refinement(draft, output).ok


@pytest.mark.parametrize("draft, output, reason", [
    ("I am hungry.", "I am not hungry.", "negation"),
    ("I am not hungry.", "I am hungry.", "negation"),
    ("I want some water.", "I want some cold water.", "added new meaning"),
    ("No water.", "There is no water.", "added new meaning"),
    ("I am hungry.", "Are you hungry?", "sentence type"),
    ("I want some water.", "I want some food.", "dropped"),
    ("I am going home.", "Let's go home.", "dropped"),
    ("I am hungry.", "", "empty"),
])
def test_rejects_meaning_changes(draft, output, reason):
    result = check_refinement(draft, output)
    assert not result.ok
    assert reason in result.reason


def test_clean_output():
    assert clean_output('Sentence: "I\'m hungry."\nExplanation: ...') == "I'm hungry."
    assert clean_output("  **Please help me.**  ") == "Please help me."


def test_contractions():
    assert expand_contractions("I'm sure you don't") == "i am sure you do not"
    assert expand_contractions("We can't") == "we can not"
