import numpy as np
import pytest

from gesturebridge.agent import GestureBridgeAgent
from gesturebridge.config import Settings
from gesturebridge.features import extract_features, to_pixel_space
from gesturebridge.handshape_rules import ALL_SIGNS, HANDSHAPES, MODES, RuleClassifier, display_label
from gesturebridge.language.knowledge import KnowledgeBase
from gesturebridge.signs import SignVocabulary
from tests.letter_poses import LETTERS, NEW_LETTERS, pose
from tests.synthetic import sample_landmarks


def test_at_least_fifty_signs_in_three_modes():
    assert len(ALL_SIGNS) >= 50
    assert set(HANDSHAPES) == set(MODES)
    assert display_label("LETTER_A") == "A" and display_label("DIGIT_10") == "10"


def test_every_word_sign_is_in_the_vocabulary_and_lexicon():
    vocab, kb = SignVocabulary.load(), KnowledgeBase.load()
    for label in HANDSHAPES["words"]:
        assert vocab.get(label) is not None, label
        assert label in kb or vocab.kind(label) == "control", label


@pytest.mark.parametrize("mode", MODES)
def test_probabilities_are_a_distribution(mode):
    clf = RuleClassifier(mode)
    f = extract_features(to_pixel_space(sample_landmarks("OPEN", np.random.default_rng(0)), 640, 480))
    probs = clf.predict_proba(f)
    assert np.isclose(probs.sum(), 1.0) and len(probs) == len(clf.classes) == len(HANDSHAPES[mode]) + 1


def test_thumb_out_shapes():
    rng = np.random.default_rng(3)
    for mode, shape, sign in (("words", "OPEN", "HELLO"), ("words", "THUMB_UP", "HELP"),
                              ("words", "THUMB_DOWN", "NO"), ("numbers", "OPEN", "DIGIT_5"),
                              ("numbers", "THUMB_UP", "DIGIT_10")):
        clf = RuleClassifier(mode)
        votes = [clf.predict(extract_features(to_pixel_space(sample_landmarks(shape, rng), 640, 480)))[0]
                 for _ in range(20)]
        assert votes.count(sign) >= 14, (mode, shape, votes)


def test_spelled_letters_become_one_word():
    settings = Settings()
    settings.language.use_llm = False
    agent = GestureBridgeAgent(settings, RuleClassifier("letters"))
    try:
        agent.add_gloss("MY")
        agent.add_gloss("NAME")
        agent.spelling.extend("NANCY")
        translation = agent.finalize()          # finalising also commits the spelled word
        assert translation.done.wait(10)
        assert translation.result.final_sentence == "My name is Nancy."
    finally:
        agent.shutdown()


def test_all_26_letters():
    letters = {display_label(label) for label in HANDSHAPES["letters"]}
    assert letters == set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")


@pytest.mark.parametrize("letter", LETTERS)
def test_each_letter_reads_its_own_key_pose(letter):
    scores = RuleClassifier("letters").scores(pose(letter))
    assert max(scores, key=scores.get) == f"LETTER_{letter}", sorted(scores.items(), key=lambda kv: -kv[1])[:3]


def test_new_letters_survive_landmark_noise():
    clf, rng = RuleClassifier("letters"), np.random.default_rng(5)
    for letter in NEW_LETTERS:
        hits = 0
        for _ in range(40):
            p = pose(letter) + rng.normal(0, 0.02, (21, 3))
            p -= p[0]
            scores = clf.scores(p)
            hits += max(scores, key=scores.get) == f"LETTER_{letter}"
        assert hits >= 34, (letter, hits)
