"""Beam-search decoder: choose the most plausible sign sequence under uncertainty.

For every word slot the recogniser keeps its top-k candidate signs with
probabilities, e.g. [("HUNGRY", .48), ("WANT", .44), ("FOOD", .05)].
Taking the top-1 of every slot ("greedy") can produce nonsense such as
"I WANT" + end. The decoder scores whole sequences instead:

    score(w1..wn) = sum_i log P_classifier(w_i)                      (evidence)
                  + lambda * sum_i log P_grammar(pos_i | pos_i-1)    (prior knowledge)

P_grammar is a part-of-speech bigram model estimated (with add-k smoothing)
from example sign sentences in ``knowledge/gloss_corpus.txt``. Beam search
keeps only the best `beam_width` partial sequences after each slot, a
heuristic search that is linear in sentence length instead of k^n.

lambda = 0.2 was tuned on simulated noisy versions of the corpus sentences
(not the test set): it never changes a correctly recognised sentence, yet
recovers many sentences when recognition is noisy. Larger values let the coarse
grammar prior overrule confident, correct signs.
"""

from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from ..config import CORPUS_PATH
from .knowledge import KnowledgeBase

START, END = "<s>", "</s>"

Candidates = list[list[tuple[str, float]]]


class PosBigramModel:
    def __init__(self, counts: Counter, tags: list[str], smoothing: float = 0.5):
        self.counts = counts
        self.tags = sorted(set(tags))
        self.smoothing = smoothing
        self.context_totals: Counter = Counter()
        for (prev, _), n in counts.items():
            self.context_totals[prev] += n
        self.outcomes = len(self.tags) + 1  # every tag + END

    @classmethod
    def from_corpus(cls, kb: KnowledgeBase, path: Path = CORPUS_PATH, smoothing: float = 0.5) -> "PosBigramModel":
        counts: Counter = Counter()
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tags = [START] + [kb.pos(g) for g in line.split()] + [END]
            counts.update(zip(tags, tags[1:]))
        return cls(counts, kb.tags, smoothing)

    def prob(self, prev: str, tag: str) -> float:
        k = self.smoothing
        return (self.counts[(prev, tag)] + k) / (self.context_totals[prev] + k * self.outcomes)

    def logp(self, prev: str, tag: str) -> float:
        return math.log(self.prob(prev, tag))


@dataclass
class Hypothesis:
    glosses: tuple[str, ...]
    score: float
    evidence: float      # sum of classifier log-probabilities
    grammar: float       # sum of grammar log-probabilities (before lambda)
    last_tag: str = START

    def as_dict(self) -> dict:
        return {"glosses": " ".join(self.glosses), "score": round(self.score, 3),
                "classifier_logp": round(self.evidence, 3), "grammar_logp": round(self.grammar, 3)}


@dataclass
class DecodeResult:
    best: list[str]
    greedy: list[str]
    beam: list[Hypothesis] = field(default_factory=list)
    greedy_score: Hypothesis | None = None

    @property
    def changed(self) -> bool:
        return self.best != self.greedy

    def corrections(self) -> list[tuple[int, str, str]]:
        return [(i, g, b) for i, (g, b) in enumerate(zip(self.greedy, self.best)) if g != b]


class BeamSearchDecoder:
    def __init__(self, kb: KnowledgeBase, lm: PosBigramModel, beam_width: int = 5,
                 lm_weight: float = 0.2, min_prob: float = 1e-3):
        self.kb = kb
        self.lm = lm
        self.beam_width = beam_width
        self.lm_weight = lm_weight
        self.min_prob = min_prob

    @classmethod
    def load(cls, kb: KnowledgeBase | None = None, **kwargs) -> "BeamSearchDecoder":
        kb = kb or KnowledgeBase.load()
        return cls(kb, PosBigramModel.from_corpus(kb), **kwargs)

    def _extend(self, hyp: Hypothesis, gloss: str, p: float) -> Hypothesis:
        tag = self.kb.pos(gloss)
        evidence = math.log(max(p, self.min_prob))
        grammar = self.lm.logp(hyp.last_tag, tag)
        return Hypothesis(
            glosses=hyp.glosses + (gloss,),
            score=hyp.score + evidence + self.lm_weight * grammar,
            evidence=hyp.evidence + evidence,
            grammar=hyp.grammar + grammar,
            last_tag=tag,
        )

    def _finish(self, hyp: Hypothesis) -> Hypothesis:
        grammar = self.lm.logp(hyp.last_tag, END)
        return Hypothesis(hyp.glosses, hyp.score + self.lm_weight * grammar, hyp.evidence,
                          hyp.grammar + grammar, END)

    def score_sequence(self, slots: Candidates, glosses: list[str]) -> Hypothesis:
        hyp = Hypothesis((), 0.0, 0.0, 0.0)
        for slot, gloss in zip(slots, glosses):
            hyp = self._extend(hyp, gloss, dict(slot).get(gloss, self.min_prob))
        return self._finish(hyp)

    def decode(self, slots: Candidates) -> DecodeResult:
        slots = [[(g, p) for g, p in slot if p > 0] or slot for slot in slots if slot]
        greedy = [max(slot, key=lambda c: c[1])[0] for slot in slots]
        if not slots:
            return DecodeResult([], [])

        beam = [Hypothesis((), 0.0, 0.0, 0.0)]
        for slot in slots:
            expanded = [self._extend(hyp, gloss, p) for hyp in beam for gloss, p in slot]
            expanded.sort(key=lambda h: h.score, reverse=True)
            beam = expanded[: self.beam_width]
        finished = sorted((self._finish(h) for h in beam), key=lambda h: h.score, reverse=True)
        return DecodeResult(
            best=list(finished[0].glosses),
            greedy=greedy,
            beam=finished,
            greedy_score=self.score_sequence(slots, greedy),
        )
