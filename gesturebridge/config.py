"""Central configuration: file locations and every tunable setting.

Settings have sensible defaults and can be overridden from ``settings.json``
(written by the Streamlit sidebar's "Save settings" button).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent

KNOWLEDGE_DIR = PROJECT_ROOT / "knowledge"
SIGNS_PATH = KNOWLEDGE_DIR / "signs.json"
LEXICON_PATH = KNOWLEDGE_DIR / "lexicon.json"
RULES_PATH = KNOWLEDGE_DIR / "grammar_rules.json"
CORPUS_PATH = KNOWLEDGE_DIR / "gloss_corpus.txt"
PHRASEBOOK_PATH = KNOWLEDGE_DIR / "hindi_phrasebook.json"
CUSTOM_SIGNS_PATH = KNOWLEDGE_DIR / "custom_signs.json"   # words the team added themselves

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "landmarks.csv"
HISTORY_PATH = DATA_DIR / "history.csv"
TEST_SENTENCES_PATH = DATA_DIR / "test_sentences.json"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "sign_classifier.joblib"
HAND_MODEL_PATH = MODELS_DIR / "hand_landmarker.task"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_PATH = REPORTS_DIR / "metrics.json"

SETTINGS_PATH = PROJECT_ROOT / "settings.json"
CACHE_DIR = PROJECT_ROOT / ".cache"          # caches/temp files stay on the project drive


@dataclass
class RecognitionSettings:
    threshold: float = 0.70             # smoothed probability needed to accept a sign
    uncertain_threshold: float = 0.40   # between this and `threshold` -> "please repeat"
    hold_seconds: float = 0.6           # a sign must stay stable this long
    window: int = 8                     # frames averaged by the stabiliser
    cooldown_seconds: float = 0.8       # pause after any accepted sign
    release_seconds: float = 0.7        # show something else this long before the same sign counts again
    motion_gate: bool = True            # only accept signs while the hand is (almost) still
    motion_threshold: float = 1.2       # hand-lengths per second that still count as "still"
    min_detection_confidence: float = 0.6
    min_tracking_confidence: float = 0.5
    max_hands: int = 1
    top_k: int = 3


@dataclass
class LanguageSettings:
    use_beam: bool = True
    beam_width: int = 5
    lm_weight: float = 0.2                # grammar-prior weight in beam search (tuned)
    use_llm: bool = True
    llm_backend: str = "ollama"          # "ollama" (REST) or "langchain"
    llm_model: str = "llama3.2:3b"
    ollama_url: str = "http://127.0.0.1:11434"
    llm_timeout: float = 6.0            # speech never waits longer for the LLM (rules are used)
    prompt_style: str = "few_shot"       # polishes most sentences; the guard catches rare meaning changes.
                                         # "few_shot_strict" changes the draft only when needed; "zero_shot" for comparison
    output_language: str = "en"          # "en" or "hi"
    hindi_form: str = "m"                # "m"/"f": grammatical gender of first-person Hindi verbs


@dataclass
class SpeechSettings:
    enabled: bool = True
    output: str = "browser"              # "browser": audio player in the page; "device": straight to the speakers
    rate: int = 165
    volume: float = 1.0
    voice_prompt_on_uncertain: bool = False   # the signer may not hear it, so the visual prompt is the default


@dataclass
class CameraSettings:
    index: int = 0
    width: int = 640
    height: int = 480
    mirror: bool = True                  # selfie view; must match between data collection and use


@dataclass
class UISettings:
    large_text: bool = False
    high_contrast: bool = False


@dataclass
class Settings:
    recognition: RecognitionSettings = field(default_factory=RecognitionSettings)
    language: LanguageSettings = field(default_factory=LanguageSettings)
    speech: SpeechSettings = field(default_factory=SpeechSettings)
    camera: CameraSettings = field(default_factory=CameraSettings)
    ui: UISettings = field(default_factory=UISettings)
    model_path: str = ""                 # empty -> MODEL_PATH

    @property
    def resolved_model_path(self) -> Path:
        return Path(self.model_path) if self.model_path else MODEL_PATH

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Settings":
        settings = cls()
        for section in fields(cls):
            value = data.get(section.name)
            if value is None:
                continue
            current = getattr(settings, section.name)
            if isinstance(value, dict) and hasattr(current, "__dataclass_fields__"):
                for key, item in value.items():
                    if key in current.__dataclass_fields__:
                        setattr(current, key, item)
            elif not isinstance(value, dict):
                setattr(settings, section.name, value)
        return settings

    @classmethod
    def load(cls, path: Path = SETTINGS_PATH) -> "Settings":
        if Path(path).exists():
            try:
                return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
            except (OSError, ValueError):
                pass
        return cls()

    def save(self, path: Path = SETTINGS_PATH) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
