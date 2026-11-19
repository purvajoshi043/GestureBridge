"""Language pipeline: candidate signs -> decoded glosses -> English (-> Hindi).

    candidates --beam search--> glosses --forward chaining--> draft sentence
               --local LLM + guard--> final sentence --phrasebook/LLM--> Hindi

`prepare()` is the fast symbolic part (a few milliseconds) and `polish()` the
slow neural part (LLM), so a UI can show the draft at once and the polished
sentence when it is ready.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from .config import LanguageSettings
from .language.decoder import BeamSearchDecoder, Candidates, DecodeResult
from .language.hindi import HindiResult, HindiTranslator
from .language.knowledge import KnowledgeBase
from .language.llm import RefineResult, SentenceRefiner
from .language.rule_engine import Derivation, RuleEngine


@dataclass
class TranslationResult:
    candidates: Candidates
    decode: DecodeResult
    derivation: Derivation
    final_sentence: str
    refine: RefineResult | None = None
    hindi: HindiResult | None = None
    timings_ms: dict[str, float] = field(default_factory=dict)

    @property
    def glosses(self) -> list[str]:
        return self.decode.best

    @property
    def rule_sentence(self) -> str:
        return self.derivation.sentence

    @property
    def spoken_text(self) -> str:
        return self.hindi.text if self.hindi and self.hindi.text else self.final_sentence

    @property
    def spoken_language(self) -> str:
        return "hi" if self.hindi and self.hindi.text else "en"


def parse_gloss_text(text: str) -> Candidates:
    """Parse "I WANT:0.5/HUNGRY:0.4 WATER" into candidate slots.

    Slots are separated by spaces, alternatives by "/", probabilities are
    optional (missing ones share the remaining probability mass equally).
    """
    slots: Candidates = []
    for chunk in text.replace(",", " ").split():
        options = []
        for part in chunk.split("/"):
            if not part:
                continue
            gloss, _, prob = part.partition(":")
            options.append((gloss.strip().upper(), float(prob) if prob else None))
        if not options:
            continue
        given = sum(p for _, p in options if p is not None)
        missing = [g for g, p in options if p is None]
        share = max(0.0, 1.0 - given) / len(missing) if missing else 0.0
        slots.append([(g, p if p is not None else (share or 1.0 / len(options))) for g, p in options])
    return slots


class LanguagePipeline:
    def __init__(self, settings: LanguageSettings | None = None, kb: KnowledgeBase | None = None):
        self.settings = settings or LanguageSettings()
        self.kb = kb or KnowledgeBase.load()
        self.engine = RuleEngine.load(self.kb)
        self.decoder = BeamSearchDecoder.load(self.kb)
        self.refiner = SentenceRefiner()
        self.translator = HindiTranslator(self.refiner.client)
        self.configure(self.settings)

    def configure(self, settings: LanguageSettings) -> None:
        self.settings = settings
        self.decoder.beam_width = max(1, int(settings.beam_width))
        self.decoder.lm_weight = float(settings.lm_weight)
        self.refiner.configure(settings.llm_model, settings.ollama_url, float(settings.llm_timeout),
                               settings.llm_backend, settings.prompt_style)
        self.translator.client = self.refiner.client

    # -- fast, symbolic part -----------------------------------------------
    def prepare(self, candidates: Candidates) -> TranslationResult:
        timings = {}
        started = time.perf_counter()
        if self.settings.use_beam:
            decode = self.decoder.decode(candidates)
        else:
            greedy = [max(slot, key=lambda c: c[1])[0] for slot in candidates if slot]
            decode = DecodeResult(best=greedy, greedy=greedy)
        timings["decode"] = _ms(started)

        started = time.perf_counter()
        derivation = self.engine.run(decode.best)
        timings["rules"] = _ms(started)
        return TranslationResult(candidates, decode, derivation, derivation.sentence, timings_ms=timings)

    # -- slow, neural part ---------------------------------------------------
    def polish(self, result: TranslationResult, use_llm: bool | None = None,
               language: str | None = None) -> TranslationResult:
        use_llm = self.settings.use_llm if use_llm is None else use_llm
        language = language or self.settings.output_language

        started = time.perf_counter()
        if use_llm and result.rule_sentence:
            result.refine = self.refiner.refine(result.glosses, result.rule_sentence)
        else:
            reason = "LLM switched off" if result.rule_sentence else "empty sentence"
            result.refine = RefineResult(result.rule_sentence, None, False, "disabled", reason)
        result.final_sentence = result.refine.text
        result.timings_ms["llm"] = _ms(started)

        if language == "hi" and result.final_sentence:
            started = time.perf_counter()
            result.hindi = self.translator.translate(result.final_sentence, self.settings.hindi_form)
            if not result.hindi.text and result.final_sentence != result.rule_sentence:
                result.hindi = self.translator.translate(result.rule_sentence, self.settings.hindi_form)
            result.timings_ms["hindi"] = _ms(started)
        return result

    def translate(self, candidates: Candidates, use_llm: bool | None = None,
                  language: str | None = None) -> TranslationResult:
        return self.polish(self.prepare(candidates), use_llm, language)

    def translate_glosses(self, glosses: list[str], **kwargs) -> TranslationResult:
        return self.translate([[(g.upper(), 1.0)] for g in glosses], **kwargs)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
