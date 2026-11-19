from gesturebridge.config import LanguageSettings
from gesturebridge.language.hindi import HindiTranslator
from gesturebridge.language.llm import SentenceRefiner
from gesturebridge.pipeline import LanguagePipeline


def test_rules_only_pipeline():
    pipeline = LanguagePipeline(LanguageSettings(use_llm=False))
    result = pipeline.translate_glosses("WATER I WANT".split())
    assert result.final_sentence == "I want some water."
    assert result.refine.status == "disabled"
    assert set(result.timings_ms) >= {"decode", "rules", "llm"}


def test_unreachable_llm_falls_back_to_rules():
    refiner = SentenceRefiner(base_url="http://127.0.0.1:9", timeout=1)
    result = refiner.refine(["I", "HUNGRY"], "I am hungry.")
    assert result.status == "unavailable" and result.text == "I am hungry." and not result.used_llm


def test_rejected_llm_answer_falls_back(monkeypatch):
    refiner = SentenceRefiner()
    monkeypatch.setattr(refiner, "availability", lambda: (True, "test"))
    monkeypatch.setattr(refiner, "_ask", lambda glosses, draft: "I am not hungry at all.")
    result = refiner.refine(["I", "HUNGRY"], "I am hungry.")
    assert result.status == "rejected" and result.text == "I am hungry."


def test_accepted_llm_answer_is_used(monkeypatch):
    refiner = SentenceRefiner()
    monkeypatch.setattr(refiner, "availability", lambda: (True, "test"))
    monkeypatch.setattr(refiner, "_ask", lambda glosses, draft: "I'm hungry.")
    result = refiner.refine(["I", "HUNGRY"], "I am hungry.")
    assert result.status == "ok" and result.text == "I'm hungry." and result.used_llm


def test_hindi_phrasebook_offline_with_gender_forms():
    translator = HindiTranslator(client=None)
    assert translator.translate("I'm hungry.").text == "मुझे भूख लगी है।"
    assert translator.translate("I am going home.", "m").text == "मैं घर जा रहा हूँ।"
    assert translator.translate("I am going home.", "f").text == "मैं घर जा रही हूँ।"
    missing = translator.translate("The weather is lovely today.")
    assert missing.text is None and missing.source == "none"
