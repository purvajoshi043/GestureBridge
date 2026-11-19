import numpy as np
import pytest

from gesturebridge.agent import GestureBridgeAgent
from gesturebridge.config import Settings
from gesturebridge.hand_tracker import HandObservation
from tests.synthetic import sample_landmarks

CLASSES = ["CLEAR", "DELETE", "HUNGRY", "I", "NOTHING", "SPEAK", "WATER"]


class ScriptedClassifier:
    """Stands in for the trained model: returns the probabilities of the sign being 'shown'."""

    classes = CLASSES
    model_name = "scripted"
    mirror_left = True

    def __init__(self):
        self.current = "NOTHING"

    def predict_proba(self, features):
        p = np.full(len(CLASSES), 0.02)
        p[CLASSES.index(self.current)] = 1 - 0.02 * (len(CLASSES) - 1)
        return p


@pytest.fixture
def agent():
    settings = Settings()
    settings.language.use_llm = False
    settings.speech.enabled = False
    agent = GestureBridgeAgent(settings, ScriptedClassifier())
    yield agent
    agent.shutdown()


def show(agent, label, seconds, t):
    rng = np.random.default_rng(0)
    landmarks = sample_landmarks("OPEN", rng)
    hand = HandObservation(landmarks, "Right", 0.99, 640, 480)
    agent.classifier.current = label
    for _ in range(int(seconds * 30)):
        agent.process(hand, t)
        t += 1 / 30
    return t


def test_signs_to_sentence(agent):
    t = show(agent, "I", 1.2, 0.0)
    t = show(agent, "NOTHING", 0.4, t)
    t = show(agent, "HUNGRY", 1.2, t)
    assert [e.gloss for e in agent.buffer] == ["I", "HUNGRY"]
    t = show(agent, "SPEAK", 1.2, t)
    translation = agent.latest
    assert translation is not None and agent.buffer == []
    assert translation.done.wait(10)
    assert translation.status == "done"
    assert translation.result.final_sentence == "I am hungry."


def test_control_signs(agent):
    t = show(agent, "I", 1.2, 0.0)
    t = show(agent, "WATER", 1.2, t)
    t = show(agent, "DELETE", 1.2, t)
    assert [e.gloss for e in agent.buffer] == ["I"]
    show(agent, "CLEAR", 1.2, t)
    assert agent.buffer == []


def test_finalize_with_empty_buffer(agent):
    assert agent.finalize() is None
    assert "Nothing to translate" in agent.messages[-1][1]


def test_manual_words_and_question(agent):
    for gloss in ("YOU", "HUNGRY", "QUESTION"):
        agent.add_gloss(gloss)
    translation = agent.finalize()
    assert translation.done.wait(10)
    assert translation.result.final_sentence == "Are you hungry?"
