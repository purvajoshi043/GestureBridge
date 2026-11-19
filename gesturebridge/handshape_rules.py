"""Instant mode: a knowledge-based handshape recogniser - 58 signs, no training needed.

Every sign is an IF-THEN rule over facts about the hand, e.g.

    IF index straight AND middle curled AND ring curled AND little curled
       AND thumb tucked AND index points sideways  THEN  GO

The facts come from the 21 landmarks and are *fuzzy* (a degree between 0 and 1,
not just true/false): "how straight is the index finger?". A rule's strength is
the minimum of its conditions (fuzzy AND); strengths are normalised into
probabilities, so the stabiliser and the beam-search decoder get ranked
candidates exactly as from a trained classifier.

Like real sign languages, one handshape can mean different things, so there are
three modes: WORDS (21 signs), LETTERS (all 26 ASL fingerspelling letters, to
spell names) and NUMBERS (0-10). J and Z are moving letters in ASL; here they are
read from a still pose (the end of the J stroke, the first stroke of Z).
"""

from __future__ import annotations

import numpy as np

FINGERS = {"i": (5, 6, 7, 8), "m": (9, 10, 11, 12), "r": (13, 14, 15, 16), "p": (17, 18, 19, 20)}
THUMB_MCP, THUMB_TIP, INDEX_MCP = 2, 4, 5
FINGER_JOINTS = (6, 7, 8, 10, 11, 12, 14, 15, 16, 18, 19, 20)   # middle joints and tips

MODES = ("words", "letters", "numbers")

# label -> (emoji, how to make it). Letters are "LETTER_A", digits "DIGIT_7".
HANDSHAPES = {
    "words": {
        "HELLO":     ("🖐️", "open hand, thumb out"),
        "PLEASE":    ("✋", "flat hand up, thumb tucked"),
        "MY":        ("🫱", "flat hand sideways, thumb tucked"),
        "YES":       ("✊", "fist"),
        "HELP":      ("👍", "thumbs up"),
        "NO":        ("👎", "thumbs down"),
        "I":         ("🤙", "only the little finger up"),
        "SORRY":     ("🤙", "thumb + little finger (Y)"),
        "YOU":       ("☝️", "index finger up"),
        "GO":        ("👉", "index finger sideways"),
        "THIRSTY":   ("👇", "index finger down"),
        "NAME":      ("✌️", "index + middle together, sideways"),
        "HOME":      ("🫰", "L-shape: thumb + index"),
        "CLEAR":     ("✌️", "V: index + middle up"),
        "WANT":      ("🤟", "thumb + index + middle"),
        "DELETE":    ("🤘", "horns: index + little finger"),
        "THANK_YOU": ("🤟", "I-love-you: thumb + index + little"),
        "WATER":     ("🖖", "W: index + middle + ring"),
        "FOOD":      ("🤌", "all fingertips touching the thumb"),
        "HUNGRY":    ("🫳", "C-shape: all fingers curved"),
        "SPEAK":     ("👌", "OK sign - speaks the sentence"),
    },
    "letters": {
        "LETTER_A": ("✊", "fist, thumb at the side"),
        "LETTER_B": ("✋", "four fingers up together, thumb tucked"),
        "LETTER_C": ("🫳", "C-shape"),
        "LETTER_D": ("☝️", "index up, others touch the thumb"),
        "LETTER_E": ("✊", "fingertips curled down onto the thumb"),
        "LETTER_F": ("👌", "thumb + index circle, others up"),
        "LETTER_G": ("👉", "index + thumb pointing sideways"),
        "LETTER_H": ("✌️", "index + middle together, sideways"),
        "LETTER_I": ("🤙", "little finger up"),
        "LETTER_J": ("🤙", "little finger up, hand turned sideways (end of the J)"),
        "LETTER_K": ("✌️", "V with the thumb touching the middle finger"),
        "LETTER_L": ("🫰", "L-shape"),
        "LETTER_M": ("✊", "fist, thumb tip between ring and little finger"),
        "LETTER_N": ("✊", "fist, thumb tip between middle and ring finger"),
        "LETTER_O": ("👌", "all fingertips touching the thumb (O)"),
        "LETTER_P": ("👇", "K / V pointing down"),
        "LETTER_Q": ("👇", "index + thumb pointing down (G down)"),
        "LETTER_R": ("🤞", "index and middle crossed"),
        "LETTER_S": ("✊", "fist, thumb across the fingers"),
        "LETTER_T": ("✊", "fist, thumb tip between index and middle finger"),
        "LETTER_U": ("✌️", "index + middle up together"),
        "LETTER_V": ("✌️", "index + middle up, spread"),
        "LETTER_W": ("🖖", "index + middle + ring up"),
        "LETTER_X": ("☝️", "index finger hooked"),
        "LETTER_Y": ("🤙", "thumb + little finger"),
        "LETTER_Z": ("👉", "index pointing sideways, thumb tucked (start of the Z)"),
    },
    "numbers": {
        "DIGIT_0":  ("👌", "all fingertips touching the thumb (O)"),
        "DIGIT_1":  ("☝️", "index up"),
        "DIGIT_2":  ("✌️", "index + middle up"),
        "DIGIT_3":  ("🤟", "thumb + index + middle"),
        "DIGIT_4":  ("✋", "four fingers up, thumb tucked"),
        "DIGIT_5":  ("🖐️", "open hand, thumb out"),
        "DIGIT_6":  ("🖖", "thumb touches the little finger"),
        "DIGIT_7":  ("🖖", "thumb touches the ring finger"),
        "DIGIT_8":  ("🖖", "thumb touches the middle finger"),
        "DIGIT_9":  ("👌", "thumb touches the index finger"),
        "DIGIT_10": ("👍", "thumbs up"),
    },
}
ALL_SIGNS = {label: info for mode in MODES for label, info in HANDSHAPES[mode].items()}


def display_label(label: str) -> str:
    """'LETTER_A' -> 'A', 'DIGIT_7' -> '7', 'THANK_YOU' -> 'THANK YOU'."""
    if label.startswith(("LETTER_", "DIGIT_")):
        return label.split("_", 1)[1]
    return label.replace("_", " ")


def spelled_character(label: str) -> str | None:
    """The character a fingerspelling/number sign adds, or None for word signs."""
    return label.split("_", 1)[1] if label.startswith(("LETTER_", "DIGIT_")) else None


def _ramp(value: float, start: float, end: float) -> float:
    """Fuzzy membership: 0 at `start`, 1 at `end`, linear in between."""
    return float(np.clip((value - start) / (end - start), 0.0, 1.0))


def _unit2(v: np.ndarray) -> np.ndarray:
    return v / max(float(np.linalg.norm(v[:2])), 1e-6)


def hand_facts(p: np.ndarray) -> dict[str, float]:
    """Fuzzy facts from normalised landmarks (wrist at 0, palm length 1, left hands mirrored)."""
    facts: dict[str, float] = {}
    for name, (mcp, pip, dip, tip) in FINGERS.items():
        path = np.linalg.norm(p[pip] - p[mcp]) + np.linalg.norm(p[dip] - p[pip]) + np.linalg.norm(p[tip] - p[dip])
        extension = np.linalg.norm(p[tip] - p[mcp]) / max(path, 1e-6)   # 1 = straight finger
        reach = np.linalg.norm(p[tip]) / max(np.linalg.norm(p[pip]), 1e-6)  # tip beyond the middle joint?
        facts[f"{name}_straight"] = min(_ramp(extension, 0.72, 0.90), _ramp(reach, 1.0, 1.25))
        facts[f"{name}_curled"] = max(_ramp(extension, 0.62, 0.45), _ramp(reach, 1.05, 0.9))
        facts[f"{name}_bent"] = max(0.0, 1.0 - facts[f"{name}_straight"] - facts[f"{name}_curled"])
        facts[f"touch_{name}"] = _ramp(np.linalg.norm(p[THUMB_TIP] - p[tip]), 0.45, 0.25)
    thumb_tip = p[THUMB_TIP]
    # thumb "out" = its tip touches none of the other fingers' joints (a tucked thumb rests on them)
    resting = min(np.linalg.norm(thumb_tip - p[k]) for k in FINGER_JOINTS)
    facts["thumb_out"] = _ramp(resting, 0.30, 0.50)
    thumb_dir = _unit2(thumb_tip - p[THUMB_MCP])
    facts["thumb_up"], facts["thumb_down"] = _ramp(-thumb_dir[1], 0.3, 0.7), _ramp(thumb_dir[1], 0.3, 0.7)
    facts["thumb_side"] = _ramp(abs(thumb_dir[0]), 0.6, 0.9)
    index_dir = _unit2(p[8] - p[INDEX_MCP])
    facts["index_up"], facts["index_down"] = _ramp(-index_dir[1], 0.4, 0.8), _ramp(index_dir[1], 0.4, 0.8)
    facts["index_side"] = _ramp(abs(index_dir[0]), 0.6, 0.9)
    hand_dir = _unit2(p[12] - p[0])
    facts["hand_up"], facts["hand_side"] = _ramp(-hand_dir[1], 0.5, 0.85), _ramp(abs(hand_dir[0]), 0.6, 0.9)
    middle_dir = _unit2(p[12] - p[9])
    spread = np.degrees(np.arccos(np.clip(np.dot(index_dir[:2], middle_dir[:2]), -1.0, 1.0)))
    facts["together"], facts["spread"] = _ramp(spread, 14, 6), _ramp(spread, 10, 20)
    # index and middle crossed: their tips swap sides compared with their knuckles
    base = p[9][0] - p[5][0]
    tips = p[12][0] - p[8][0]
    facts["crossed"] = _ramp(-np.sign(base) * tips, 0.0, 0.12)
    # where the thumb rests: beside the index finger (A) or across the fingers (S)
    facts["thumb_at_index"] = _ramp(np.linalg.norm(thumb_tip - p[10]) - np.linalg.norm(thumb_tip - p[6]), 0.0, 0.15)
    facts["thumb_on_middle_base"] = _ramp(min(np.linalg.norm(thumb_tip - p[k]) for k in (10, 11)), 0.4, 0.2)
    # where along the row of knuckles the thumb tip sits in a fist: on a finger (S) or in a gap
    # between two fingers (T, N, M). Soft nearest-anchor: each anchor gets a share of 0..1.
    anchors = {"on_index": p[6], "gap_im": (p[6] + p[10]) / 2, "on_middle": p[10], "gap_mr": (p[10] + p[14]) / 2,
               "on_ring": p[14], "gap_rp": (p[14] + p[18]) / 2, "on_little": p[18]}
    dist = {k: float(np.linalg.norm(thumb_tip - v)) for k, v in anchors.items()}
    weight = {k: np.exp(-(d / 0.05) ** 2) for k, d in dist.items()}
    total = sum(weight.values()) + 1e-9
    for k in ("gap_im", "gap_mr", "gap_rp"):
        facts[f"thumb_{k}"] = min(_ramp(weight[k] / total, 0.35, 0.6), _ramp(dist[k], 0.30, 0.15))
    little_dir = _unit2(p[20] - p[17])
    facts["little_side"] = _ramp(abs(little_dir[0]), 0.6, 0.9)
    return facts


def _rules(f: dict[str, float], mode: str) -> dict[str, float]:
    S = lambda *n: min(f[f"{x}_straight"] for x in n)       # noqa: E731
    C = lambda *n: min(f[f"{x}_curled"] for x in n)         # noqa: E731
    B = lambda *n: min(f[f"{x}_bent"] for x in n)           # noqa: E731
    T = lambda x: f[f"touch_{x}"]                           # noqa: E731
    out, tucked = f["thumb_out"], 1.0 - f["thumb_out"]
    fist = C("i", "m", "r", "p")
    flat = min(S("i", "m", "r", "p"), tucked, 1.0 - T("i"))
    pinched = min(T("i"), T("m"), T("r"), 1.0 - max(f["i_curled"], f["m_curled"]))
    one = min(S("i"), C("m", "r", "p"))
    two = min(S("i", "m"), C("r", "p"))
    if mode == "letters":
        return {
            "LETTER_A": min(fist, f["thumb_at_index"], 1.0 - f["thumb_on_middle_base"], 1.0 - f["thumb_down"]),
            "LETTER_B": min(flat, f["hand_up"]),
            "LETTER_C": min(B("i", "m", "r"), out, 1.0 - T("i")),
            "LETTER_D": min(S("i"), T("m"), 1.0 - f["m_straight"], 1.0 - f["r_straight"],
                            1.0 - f["index_side"], 1.0 - f["index_down"]),
            "LETTER_E": min(C("i", "m", "r"), T("i"), T("m"), tucked),
            "LETTER_F": min(T("i"), S("m", "r", "p")),
            "LETTER_G": min(one, out, f["index_side"]),
            "LETTER_H": min(two, f["together"], f["index_side"]),
            "LETTER_I": min(C("i", "m", "r"), S("p"), tucked, 1.0 - f["little_side"]),
            "LETTER_J": min(C("i", "m", "r"), S("p"), tucked, f["little_side"]),
            "LETTER_K": min(two, f["spread"], f["thumb_up"], 1.0 - f["thumb_down"], f["index_up"],
                            1.0 - f["crossed"], f["thumb_on_middle_base"]),
            "LETTER_L": min(one, out, f["index_up"]),
            "LETTER_M": min(fist, f["thumb_gap_rp"]),
            "LETTER_N": min(fist, f["thumb_gap_mr"]),
            "LETTER_O": min(pinched, B("i", "m")),
            "LETTER_P": min(two, f["index_down"]),
            "LETTER_Q": min(one, out, f["index_down"]),
            "LETTER_R": min(two, f["crossed"], 1.0 - f["index_down"]),
            "LETTER_S": min(fist, f["thumb_on_middle_base"], tucked,
                            1.0 - max(f["thumb_gap_im"], f["thumb_gap_mr"], f["thumb_gap_rp"])),
            "LETTER_T": min(fist, f["thumb_gap_im"]),
            "LETTER_U": min(two, f["together"], f["index_up"], tucked, 1.0 - f["crossed"]),
            "LETTER_V": min(two, f["spread"], f["index_up"], 1.0 - f["thumb_on_middle_base"], 1.0 - f["crossed"]),
            "LETTER_W": min(S("i", "m", "r"), C("p")),
            "LETTER_X": min(B("i"), C("m", "r", "p"), tucked, 1.0 - T("i")),
            "LETTER_Y": min(C("i", "m", "r"), S("p"), out),
            "LETTER_Z": min(one, tucked, f["index_side"]),
        }
    if mode == "numbers":
        return {
            "DIGIT_0": min(pinched, B("i", "m")),
            "DIGIT_1": min(one, tucked, f["index_up"]),
            "DIGIT_2": min(two, tucked),
            "DIGIT_3": min(two, out),
            "DIGIT_4": min(S("i", "m", "r", "p"), tucked, 1.0 - T("i")),
            "DIGIT_5": min(S("i", "m", "r", "p"), out, 1.0 - T("i")),
            "DIGIT_6": min(S("i", "m", "r"), T("p")),
            "DIGIT_7": min(S("i", "m", "p"), T("r")),
            "DIGIT_8": min(S("i", "r", "p"), T("m")),
            "DIGIT_9": min(T("i"), S("m", "r", "p")),
            "DIGIT_10": min(fist, out, f["thumb_up"]),
        }
    return {
        "FOOD": pinched,
        "SPEAK": min(T("i"), S("m", "r", "p")),
        "HUNGRY": min(B("i", "m", "r"), out, 1.0 - T("i")),
        "YES": min(fist, tucked),
        "HELP": min(fist, out, f["thumb_up"]),
        "NO": min(fist, out, f["thumb_down"]),
        "I": min(C("i", "m", "r"), S("p"), tucked),
        "SORRY": min(C("i", "m", "r"), S("p"), out),
        "YOU": min(one, tucked, f["index_up"]),
        "GO": min(one, tucked, f["index_side"]),
        "THIRSTY": min(one, tucked, f["index_down"]),
        "HOME": min(one, out),
        "NAME": min(two, tucked, f["together"], f["index_side"]),
        "CLEAR": min(two, tucked, 1.0 - min(f["together"], f["index_side"])),
        "WANT": min(two, out),
        "DELETE": min(S("i", "p"), C("m", "r"), tucked),
        "THANK_YOU": min(S("i", "p"), C("m", "r"), out),
        "WATER": min(S("i", "m", "r"), C("p")),
        "HELLO": min(S("i", "m", "r", "p"), out, 1.0 - T("i")),
        "PLEASE": min(flat, 1.0 - f["hand_side"]),
        "MY": min(flat, f["hand_side"]),
    }


class RuleClassifier:
    """Same interface as a trained SignClassifier, so the agent can use either."""

    model_name = "instant"
    mirror_left = True

    def __init__(self, mode: str = "words"):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        self.mode = mode
        self.classes = sorted(list(HANDSHAPES[mode]) + ["NOTHING"])
        self.last_facts: dict[str, float] = {}

    @property
    def description(self) -> str:
        return (f"INSTANT ({self.mode}) - our knowledge-based handshape rules with fuzzy logic, "
                f"{len(ALL_SIGNS)} signs in 3 modes, no training")

    def scores(self, points: np.ndarray) -> dict[str, float]:
        self.last_facts = hand_facts(points)
        scores = _rules(self.last_facts, self.mode)
        scores["NOTHING"] = max(0.05, 1.0 - max(scores.values()))
        return scores

    def predict_proba(self, features: np.ndarray) -> np.ndarray:
        # the first 60 features are the normalised x, y, z of landmarks 1-20 (the wrist is the origin)
        points = np.vstack([np.zeros(3), np.asarray(features[:60], dtype=np.float64).reshape(20, 3)])
        scores = self.scores(points)
        strength = np.array([scores[c] for c in self.classes]) ** 2 + 1e-4   # sharpen, avoid zeros
        return strength / strength.sum()

    def predict(self, features: np.ndarray) -> tuple[str, float]:
        probs = self.predict_proba(features)
        best = int(np.argmax(probs))
        return self.classes[best], float(probs[best])
