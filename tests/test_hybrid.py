import numpy as np

from gesturebridge.custom_signs import add_custom, load_custom, normalise_label
from gesturebridge.features import extract_features
from gesturebridge.handshape_rules import ALL_SIGNS
from gesturebridge.hybrid import HybridClassifier, build_novelty, find_conflict
from gesturebridge.language.knowledge import KnowledgeBase
from gesturebridge.signs import SignVocabulary
from tests.letter_poses import pose


def features(points, rng=None, noise=0.0):
    p = points + (rng.normal(0, noise, points.shape) if rng is not None else 0.0)
    p = p - p[0]
    px = p * 100.0 + np.array([320.0, 240.0, 0.0])
    return extract_features(px, "Right")


def recording(letter, n=40, seed=0):
    rng = np.random.default_rng(seed)
    return np.array([features(pose(letter), rng, 0.015) for _ in range(n)])


class StubTrained:
    """A 'trained model' that always votes for its first sign - the novelty check must keep it honest."""

    def __init__(self, classes):
        self.classes = list(classes)

    def predict_proba(self, _features):
        probs = np.full(len(self.classes), 0.02)
        probs[0] = 1.0 - 0.02 * (len(self.classes) - 1)
        return probs


def novelty_for(label, letter):
    X = np.vstack([recording(letter, seed=1), recording(letter, seed=2)])
    groups = np.array(["r1"] * 40 + ["r2"] * 40)
    return build_novelty(X, np.array([label] * 80), groups)


def test_own_handshape_replaces_the_built_in_sign_and_other_signs_still_work():
    # the signer recorded YOU with an L-shape (the built-in HOME handshape)
    trained = StubTrained(["YOU", "NOTHING"])
    hybrid = HybridClassifier(trained, "words", novelty_for("YOU", "L"))
    assert "YOU" in hybrid.overridden and "YOU" in hybrid.trained_labels and "YOU" not in hybrid.instant_labels
    assert len(hybrid.classes) == 21 + 1                       # 20 instant word signs + YOU + NOTHING
    assert hybrid.predict(features(pose("L")))[0] == "YOU"     # your recording wins
    assert hybrid.predict(features(pose("W")))[0] == "WATER"   # not like your recordings: built-in rule decides
    assert hybrid.predict(features(pose("B")))[0] == "PLEASE"


def test_trained_letters_only_appear_in_letters_mode():
    trained = StubTrained(["LETTER_A", "NOTHING"])
    novelty = novelty_for("LETTER_A", "A")
    assert "LETTER_A" not in HybridClassifier(trained, "words", novelty).classes
    letters = HybridClassifier(trained, "letters", novelty)
    assert "LETTER_A" in letters.trained_labels and len(letters.classes) == 26 + 1


def test_recording_an_existing_handshape_is_reported():
    found = find_conflict(recording("L"), "YOU")                # L-shape is already HOME
    assert found is not None and found[0] == "HOME" and found[1] > 0.5
    found = find_conflict(recording("A"), "LETTER_T")           # A-handshape is already the letter A
    assert found is not None and found[0] == "LETTER_A"
    assert find_conflict(recording("W"), "WATER") is None       # WATER's own handshape is no conflict
    assert find_conflict(recording("W"), "NOTHING") is None


def test_custom_words(tmp_path):
    path = tmp_path / "custom_signs.json"
    vocab_labels = SignVocabulary.load(custom_path=None).labels
    item, reason = add_custom("Mother", "noun", "flat hand on the chin", vocab_labels, list(ALL_SIGNS), path=path)
    assert reason is None and item["label"] == "MOTHER"
    for word in ("water", "Thank you", "a", "mother"):          # existing word, phrase, letter, added twice
        assert add_custom(word, "noun", "", vocab_labels, list(ALL_SIGNS), path=path)[0] is None
    kb = KnowledgeBase.load(custom_path=path)
    eat, _ = add_custom("eat", "noun", "", vocab_labels, list(ALL_SIGNS), kb, path=path)
    assert eat["pos"] == "verb"                                 # known lexicon facts are reused
    assert [i["label"] for i in load_custom(path)] == ["MOTHER", "EAT"]
    vocab = SignVocabulary.load(custom_path=path)
    assert "MOTHER" in vocab.words and vocab.custom == ["MOTHER", "EAT"]
    kb = KnowledgeBase.load(custom_path=path)
    assert kb.pos("MOTHER") == "noun" and kb.text("MOTHER") == "mother"
    assert normalise_label(" good  morning! ") == "GOOD_MORNING"


def test_custom_verb_gets_grammar_forms(tmp_path):
    path = tmp_path / "custom_signs.json"
    add_custom("sleep", "verb", "", [], [], path=path)
    kb = KnowledgeBase.load(custom_path=path)
    assert kb.entry("SLEEP")["s3"] == "sleeps" and kb.entry("SLEEP")["ing"] == "sleeping"
