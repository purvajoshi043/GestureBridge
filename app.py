"""GestureBridge - sign language to speech, with explainable AI (Streamlit UI).

Run:   streamlit run app.py      (or double-click run_app.bat)
"""

from __future__ import annotations

import base64
import html
import json
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

from gesturebridge.agent import GestureBridgeAgent, Translation
from gesturebridge.collect import CONDITIONS
from gesturebridge.config import (FIGURES_DIR, METRICS_PATH, MODEL_PATH, MODELS_DIR, RAW_DATA_PATH, REPORTS_DIR,
                                  Settings)
from gesturebridge.custom_signs import WORD_TYPES, add_custom
from gesturebridge.dataset import append_rows, feature_matrix, make_row, remove_burst
from gesturebridge.hybrid import HybridClassifier, find_conflict, novelty_from_dataset
from gesturebridge.evaluate import load_cases, same_sentence
from gesturebridge.hand_tracker import HandTracker, draw_hand, draw_label, open_camera
from gesturebridge.history import HistoryLog
from gesturebridge.language.llm import PROMPT_STYLES
from gesturebridge.live import LiveEngine
from gesturebridge.pipeline import LanguagePipeline, TranslationResult, parse_gloss_text
from gesturebridge.recognizer import SignClassifier
from gesturebridge.signs import Sign, SignVocabulary
from gesturebridge.speech.stt import transcribe
from gesturebridge.handshape_rules import ALL_SIGNS, HANDSHAPES, MODES, RuleClassifier
from gesturebridge.starter import GESTURE_HELP, StarterClassifier, use_starter, use_trained
from gesturebridge.speech.tts import Speaker
from gesturebridge.stabiliser import StabiliserOutput
from gesturebridge.train import train as train_models

st.set_page_config(page_title="GestureBridge", page_icon="🤟", layout="wide")
ss = st.session_state
STARTER_CHOICE = "Starter: MediaPipe gestures (no training needed)"
INSTANT_CHOICE = "Instant: 58 signs, our rule-based recogniser"
EMOJI = {label: emoji for label, (emoji, _) in ALL_SIGNS.items()} | {"QUESTION": "❓", "NOTHING": "✋"}
MODE_NAMES = {"words": "🗣️ Words", "letters": "🔤 Letters (spell names)", "numbers": "🔢 Numbers"}
BEST_CHOICE = "Best model (chosen by training)"
HYBRID_CHOICE = "Instant + your trained signs"


# ---------------------------------------------------------------------------
# Shared resources and session state
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def shared_speaker() -> Speaker:
    speaker = Speaker()
    speaker.warm_up()
    return speaker


@st.cache_resource(show_spinner=False)
def shared_vocabulary() -> SignVocabulary:
    return SignVocabulary.load()


@st.cache_resource(show_spinner="Loading the sign model ...")
def load_classifier(path: str, modified: float) -> SignClassifier:
    return SignClassifier.load(Path(path))


@st.cache_resource(show_spinner="Preparing your trained signs …")
def hybrid_parts(model_path: str, modified: float, data_modified: float):
    """The trained model + its reference frames (older models: rebuilt from the recordings)."""
    trained = SignClassifier.load(Path(model_path))
    novelty = trained.bundle.get("novelty")
    if novelty is None and RAW_DATA_PATH.exists():
        novelty = novelty_from_dataset(pd.read_csv(RAW_DATA_PATH), trained.classes)
    return trained, novelty


def current_trained():
    """(trained model, novelty data) or (None, None) before the first training."""
    if not MODEL_PATH.exists():
        return None, None
    data_modified = RAW_DATA_PATH.stat().st_mtime if RAW_DATA_PATH.exists() else 0.0
    return hybrid_parts(str(MODEL_PATH), MODEL_PATH.stat().st_mtime, data_modified)


def record_kind(label: str) -> str:
    if label.startswith("LETTER_"):
        return "letter"
    if label.startswith("DIGIT_"):
        return "number"
    if label in vocab.custom:
        return "your word"
    return vocab.kind(label)


def keypose_text(label: str) -> str:
    sign = vocab.get(label)
    if sign and sign.keypose:
        return sign.keypose
    if label in ALL_SIGNS:
        return ALL_SIGNS[label][1]
    return "Your own handshape."


def conflict_warning(rows: list[dict], label: str) -> str | None:
    """'This sign already exists': the new recording looks like another sign of the same mode."""
    try:
        X, _, _ = feature_matrix(pd.DataFrame(rows))
        trained, novelty = current_trained()
        found = find_conflict(X, label, trained, novelty)
    except Exception:  # the check must never lose a recording
        return None
    if not found:
        return None
    other, share = found
    mine = " (one of your trained signs)" if trained is not None and other in trained.classes else ""
    return (f"⚠️ Saved, but this handshape already exists: it is recognised as **{vocab.display(other)}**{mine} "
            f"in {share:.0%} of the frames. Each sign needs its own handshape, otherwise the two get mixed up. "
            "Press **↩ Undo last recording** and use a different handshape (see the sign guide), "
            "or keep it if you are sure.")


def model_options() -> dict[str, Path]:
    options = {}
    if MODEL_PATH.exists():
        options[BEST_CHOICE] = MODEL_PATH
    for key, name in (("svm", "SVM"), ("mlp", "MLP neural network"), ("rf", "Random Forest")):
        path = MODELS_DIR / f"model_{key}.joblib"
        if path.exists():
            options[name] = path
    return options


vocab = shared_vocabulary()
speaker = shared_speaker()
history = HistoryLog()

if "settings" not in ss:
    ss.settings = Settings.load()
    ss.running = False
    ss.camera = None
    ss.model_loaded = None
    ss.lab_text = "I HUNGRY"
    ss.lab_result = None
if "agent" not in ss:
    pipeline = LanguagePipeline(ss.settings.language)
    pipeline.refiner.client.warm_up()
    ss.agent = GestureBridgeAgent(ss.settings, None, pipeline, speaker, history, vocab)
agent: GestureBridgeAgent = ss.agent
settings: Settings = ss.settings


@st.cache_resource(show_spinner=False)
def live_engine() -> LiveEngine:
    engine = LiveEngine()
    engine.ensure_server()
    return engine


engine = live_engine()

# Camera only while the page is in use: when this browser tab goes into the background (another tab,
# minimised window) the page stops watching the video stream, so the camera switches off after a few
# seconds; when the tab is visible again the stream reconnects and the camera comes back by itself.
# The script is added to the main page (not the component's own frame) so it outlives reruns.
PAUSE_WHEN_HIDDEN_JS = """
<script>
const page = window.parent;
if (!page.__gbPauseWhenHidden) {
  page.__gbPauseWhenHidden = true;
  const code = `
    const BLANK = 'data:image/gif;base64,R0lGODlhAQABAAAAACH5BAEKAAEALAAAAAABAAEAAAICTAEAOw==';
    document.addEventListener('visibilitychange', () => {
      document.querySelectorAll("img[alt='camera']").forEach((img) => {
        if (document.hidden) {
          if (img.src.includes('/stream')) { img.dataset.stream = img.src; img.src = BLANK; }
        } else if (img.dataset.stream) {
          img.src = img.dataset.stream.split('?')[0] + '?t=' + Date.now();
          delete img.dataset.stream;
        }
      });
    });`;
  const tag = page.document.createElement('script');
  tag.textContent = code;
  page.document.head.appendChild(tag);
}
</script>
"""
if engine.running and engine.agent is not agent:   # page was reloaded while the camera runs
    engine.agent = agent


def bound(widget, label: str, section: str, field: str, **kwargs):
    """A sidebar widget whose value lives in `settings.<section>.<field>`."""
    target = getattr(settings, section)
    key = f"cfg_{section}_{field}"
    if key not in ss:
        ss[key] = getattr(target, field)
    value = widget(label, key=key, **kwargs)
    setattr(target, field, value)
    return value


def release_camera() -> None:
    if ss.camera is not None:
        ss.camera.release()
        ss.camera = None


def audio_html(data: bytes, mime: str, autoplay: bool, tag: str = "") -> str:
    """A real <audio> element with the speech embedded. Browsers allow it to start by itself
    because the user has just clicked a button on the page."""
    encoded = base64.b64encode(data).decode("ascii")
    return (f"<audio controls {'autoplay' if autoplay else ''} data-tag='{tag}' "
            f"style='width:100%;margin:2px 0 10px' src='data:{mime};base64,{encoded}'></audio>")


def sign_chart_items() -> list[tuple[str, str, str]]:
    if isinstance(agent.classifier, StarterClassifier):
        return [(icon, word, how) for icon, how, word in GESTURE_HELP]
    if isinstance(agent.classifier, HybridClassifier):
        clf = agent.classifier
        mine = [("⭐" + EMOJI.get(label, "✋"), vocab.display(label), "your recorded handshape")
                for label in clf.trained_labels]
        return mine + [(emoji, vocab.display(label), how) for label, (emoji, how) in HANDSHAPES[clf.mode].items()
                       if label in clf.instant_labels]
    if isinstance(agent.classifier, RuleClassifier):
        return [(emoji, vocab.display(label), how) for label, (emoji, how) in HANDSHAPES[agent.classifier.mode].items()]
    if agent.classifier is not None:
        return [(EMOJI.get(label, "✋"), vocab.display(label), vocab.get(label).keypose if vocab.get(label) else "")
                for label in agent.classifier.classes if label not in vocab.ignored]
    return []


def sign_chart_html(side: bool = False) -> str:
    tiles = "".join(f"<div class='gb-tile'><div class='e'>{e}</div><div class='n'>{html.escape(n)}</div>"
                    f"<div class='h'>{html.escape(h)}</div></div>" for e, n, h in sign_chart_items())
    return f"<div class='gb-grid{' side' if side else ''}'>{tiles}</div>"


def replay_last() -> None:
    """Speak the latest sentence again (browser player or laptop speaker)."""
    if agent.latest is None:
        st.toast("Nothing to repeat yet.", icon="✋")
    elif settings.speech.output == "browser":
        ss.replay = ss.get("replay", 0) + 1
    else:
        agent.speak_again()


def ensure_camera():
    """The one shared webcam (the live translator and recording use the same object)."""
    if ss.get("camera_index") != settings.camera.index:
        release_camera()
        ss.camera_index = settings.camera.index
    if ss.camera is None or not ss.camera.isOpened():
        ss.camera = open_camera(settings.camera.index, settings.camera.width, settings.camera.height)
    return ss.camera


def recorded_counts() -> pd.DataFrame:
    if not RAW_DATA_PATH.exists():
        return pd.DataFrame(columns=["label", "signer", "burst"])
    return pd.read_csv(RAW_DATA_PATH, usecols=["label", "signer", "burst"], dtype=str)


# ---------------------------------------------------------------------------
# Sidebar: settings
# ---------------------------------------------------------------------------

with st.sidebar:
    st.header("Settings")

    options = model_options()
    has_trained = BEST_CHOICE in options
    if ss.pop("select_best_model", False) and has_trained:
        ss.model_choice = HYBRID_CHOICE
    choices = ([HYBRID_CHOICE] if has_trained else []) + [INSTANT_CHOICE] + list(options) + [STARTER_CHOICE]
    if ss.get("model_choice") not in choices:
        ss.pop("model_choice", None)
    choice = st.selectbox("Sign model", choices, key="model_choice",
                          help="Instant + your trained signs = the signs you recorded and trained, plus every "
                               "other built-in sign (your own handshape replaces a built-in one). Instant = 58 "
                               "signs from our handshape rules (no training). The other trained models only know "
                               "the signs you recorded. Starter = MediaPipe's 7 pretrained gestures.")
    if choice in (INSTANT_CHOICE, HYBRID_CHOICE):
        mode = st.radio("Sign mode", MODES, key="instant_mode", format_func=MODE_NAMES.get,
                        help="Like real sign languages, one handshape means different things: words, "
                             "fingerspelled letters (to spell names) or numbers.")
        if choice == INSTANT_CHOICE:
            stamp, make = ("instant", mode), lambda: RuleClassifier(mode)
        else:
            trained, novelty = current_trained()
            stamp, make = ("hybrid", mode, id(trained)), lambda: HybridClassifier(trained, mode, novelty)
        if ss.model_loaded != stamp:
            agent.commit_spelling()
            use_trained(agent, make())
            ss.model_loaded = stamp
    elif choice == STARTER_CHOICE:
        stamp = ("starter",)
        if ss.model_loaded != stamp:
            with st.spinner("Loading the starter gesture model …"):
                use_starter(agent)
            ss.model_loaded = stamp
        if not options:
            st.caption("No trained model yet – record and train your own in the Record & train tab.")
    else:
        path = options[choice]
        stamp = (str(path), path.stat().st_mtime)
        if ss.model_loaded != stamp:
            try:
                use_trained(agent, load_classifier(*stamp))
                ss.model_loaded = stamp
            except Exception as exc:  # e.g. a model from an older feature version
                st.error(f"Could not load the model: {exc}")
                use_trained(agent, None)
    if agent.classifier:
        st.caption(agent.classifier.description)
        st.markdown(f"<div class='gb-side-title'>📖 Sign guide · {len(sign_chart_items())} signs</div>"
                    "<div class='gb-muted'>Hold a handshape still until the bar under your hand fills.</div>"
                    + sign_chart_html(side=True), unsafe_allow_html=True)
        if isinstance(agent.classifier, (RuleClassifier, HybridClassifier)) and agent.classifier.mode != "words":
            st.caption("Letters/numbers join into one word. Take your hand out of view (1.5 s) or press "
                       "Speak to add it to the sentence – e.g. Words: MY NAME, then Letters: N-A-N-C-Y.")

    with st.expander("Recognition", expanded=False):
        bound(st.slider, "Accept a sign at confidence", "recognition", "threshold", min_value=0.3, max_value=0.99,
              step=0.01, help="Smoothed probability needed before a sign is added.")
        bound(st.slider, "Ask to repeat below", "recognition", "uncertain_threshold", min_value=0.1,
              max_value=0.9, step=0.01, help="Between this and the acceptance level the app asks to repeat.")
        bound(st.slider, "Hold the sign for (s)", "recognition", "hold_seconds", min_value=0.2, max_value=2.0, step=0.1)
        bound(st.slider, "Smoothing window (frames)", "recognition", "window", min_value=1, max_value=20)
        bound(st.slider, "Pause after a sign (s)", "recognition", "cooldown_seconds", min_value=0.0, max_value=3.0, step=0.1)
        bound(st.toggle, "Only read signs while the hand is still", "recognition", "motion_gate")
        bound(st.slider, "Stillness limit (hand lengths / s)", "recognition", "motion_threshold",
              min_value=0.2, max_value=5.0, step=0.1)

    with st.expander("Language and LLM", expanded=False):
        bound(st.toggle, "Beam search over top-3 signs", "language", "use_beam")
        bound(st.slider, "Beam width", "language", "beam_width", min_value=1, max_value=10)
        bound(st.slider, "Grammar weight (lambda)", "language", "lm_weight", min_value=0.0, max_value=1.5, step=0.05)
        bound(st.toggle, "Polish with local LLM", "language", "use_llm")
        bound(st.selectbox, "LLM backend", "language", "llm_backend", options=["ollama", "langchain"])
        bound(st.text_input, "Ollama model", "language", "llm_model")
        bound(st.selectbox, "Prompt style", "language", "prompt_style", options=list(PROMPT_STYLES))
        bound(st.slider, "LLM timeout (s)", "language", "llm_timeout", min_value=2.0, max_value=60.0, step=1.0)
        bound(st.radio, "Speak in", "language", "output_language", options=["en", "hi"], horizontal=True,
              format_func=lambda x: "English" if x == "en" else "Hindi (needs internet)")
        bound(st.radio, "Hindi verb form for 'I'", "language", "hindi_form", options=["m", "f"], horizontal=True,
              format_func=lambda x: "masculine" if x == "m" else "feminine")

    with st.expander("Speech", expanded=False):
        bound(st.toggle, "Speak sentences aloud", "speech", "enabled")
        bound(st.radio, "Play speech", "speech", "output", options=["browser", "device"], horizontal=True,
              format_func=lambda x: "in the browser" if x == "browser" else "on the laptop speaker")
        bound(st.slider, "Speaking rate", "speech", "rate", min_value=100, max_value=250, step=5)
        bound(st.toggle, "Also say 'please repeat' aloud", "speech", "voice_prompt_on_uncertain",
              help="Off by default: the signer may not hear it, so the prompt is shown on screen.")

    with st.expander("Camera", expanded=False):
        bound(st.number_input, "Camera number", "camera", "index", min_value=0, max_value=9, step=1)
        bound(st.toggle, "Mirror view (selfie)", "camera", "mirror",
              help="Must match how the training data was recorded.")

    with st.expander("Accessibility", expanded=False):
        bound(st.toggle, "Large text", "ui", "large_text")
        bound(st.toggle, "High contrast", "ui", "high_contrast")

    if st.button("Save settings as default", width="stretch"):
        settings.save()
        st.toast("Settings saved to settings.json")

    llm_ok, llm_status = agent.pipeline.refiner.availability()
    st.caption(("🟢 LLM: " if llm_ok else "🟠 LLM off: ") + llm_status)

agent.configure(settings)


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

css = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600;800&display=swap');
html, body, p, li, label, input, textarea, button, h1, h2, h3, h4, td, th,
span:not([data-testid="stIconMaterial"]) {font-family: 'Poppins', sans-serif;}
[data-testid="stAppViewContainer"] {
  background: radial-gradient(1100px 600px at 5% -10%, rgba(139,92,246,.30), transparent 60%),
              radial-gradient(900px 520px at 100% 0%, rgba(6,182,212,.22), transparent 60%),
              radial-gradient(800px 600px at 50% 120%, rgba(244,114,182,.14), transparent 60%), #070b1a;}
[data-testid="stHeader"] {background: transparent;}
[data-testid="stSidebar"] {background: rgba(13,17,36,.92); border-right: 1px solid rgba(255,255,255,.06);}
.gb-hero {display: flex; align-items: center; gap: 18px; margin: -10px 0 4px;}
.gb-logo {font-size: 3.6rem; filter: drop-shadow(0 0 18px rgba(167,139,250,.7)); animation: gbwave 2.6s ease-in-out infinite;}
@keyframes gbwave {0%,100% {transform: rotate(0)} 25% {transform: rotate(-14deg)} 75% {transform: rotate(10deg)}}
.gb-title {font-size: 3.3rem; font-weight: 800; line-height: 1; letter-spacing: -.02em;
  background: linear-gradient(90deg,#a78bfa,#22d3ee,#f472b6,#a78bfa); background-size: 300% 100%;
  -webkit-background-clip: text; background-clip: text; color: transparent; animation: gbshine 7s linear infinite;}
@keyframes gbshine {to {background-position: 300% 0}}
.gb-tag {color: #aab3d6; font-size: 1.05rem; margin-top: 6px;}
.gb-pill {display: inline-block; padding: 4px 12px; margin: 10px 6px 4px 0; border-radius: 999px; font-size: .78rem;
  font-weight: 600; color: #e9e5ff; background: rgba(139,92,246,.16); border: 1px solid rgba(167,139,250,.40);}
.gb-card {background: linear-gradient(145deg, rgba(255,255,255,.08), rgba(255,255,255,.02));
  border: 1px solid rgba(255,255,255,.10); border-radius: 18px; padding: 14px 18px; margin-bottom: 12px;
  box-shadow: 0 12px 32px rgba(0,0,0,.35);}
.gb-label {font-size: .68rem; letter-spacing: .16em; text-transform: uppercase; color: #8b93b8; margin-bottom: 4px;}
.gb-sign {font-size: 2.3rem; font-weight: 800; color: #fff; display: flex; align-items: center; gap: .35em; line-height: 1.1;}
.gb-emoji {font-size: 2.9rem;}
.gb-conf {font-size: 1rem; font-weight: 600; color: #67e8f9; margin-left: auto;}
.gb-bar {height: 10px; border-radius: 6px; background: rgba(255,255,255,.08); overflow: hidden; margin: 10px 0 8px;}
.gb-bar > div {height: 100%; border-radius: 6px; background: linear-gradient(90deg,#22d3ee,#a78bfa,#f472b6);
  transition: width .15s ease;}
.gb-bar.unsure > div {background: linear-gradient(90deg,#f59e0b,#fbbf24); box-shadow: 0 0 14px rgba(251,191,36,.7);}
.gb-chip {display: inline-block; padding: 7px 15px; margin: 4px 6px 4px 0; border-radius: 999px; font-weight: 700;
  color: #fff; font-size: 1.05rem; background: linear-gradient(135deg, rgba(139,92,246,.55), rgba(6,182,212,.40));
  border: 1px solid rgba(255,255,255,.18); box-shadow: 0 4px 14px rgba(139,92,246,.35); animation: gbpop .35s ease-out;}
.gb-mini {display: inline-block; padding: 2px 9px; margin: 2px 4px 0 0; border-radius: 999px; font-size: .78rem;
  color: #cbd5ff; background: rgba(255,255,255,.06); border: 1px solid rgba(255,255,255,.10);}
@keyframes gbpop {from {transform: scale(.5); opacity: 0} to {transform: scale(1); opacity: 1}}
.gb-sentence {font-size: 2.3rem; font-weight: 800; line-height: 1.2; margin: 2px 0 8px; animation: gbpop .4s ease-out;
  background: linear-gradient(90deg,#ffffff,#c4b5fd 60%,#67e8f9); -webkit-background-clip: text; background-clip: text;
  color: transparent;}
.gb-hindi {font-size: 1.7rem; font-weight: 700; color: #fbcfe8;}
.gb-muted {color: #8b93b8; font-size: .84rem; line-height: 1.55;}
.gb-msg-warning {color: #fbbf24;} .gb-msg-success {color: #34d399;} .gb-msg-info {color: #93c5fd;}
.gb-video {position: relative; padding: 3px; border-radius: 22px;
  background: linear-gradient(135deg,#8b5cf6,#22d3ee,#f472b6); box-shadow: 0 0 46px rgba(139,92,246,.38);}
.gb-video img {display: block; width: 100%; border-radius: 19px;}
.gb-badge {position: absolute; top: 14px; left: 16px; padding: 3px 11px; border-radius: 999px; font-size: .75rem;
  font-weight: 800; color: #fff; background: #ef4444; letter-spacing: .08em; animation: gbpulse 1.4s infinite;}
@keyframes gbpulse {0% {box-shadow: 0 0 0 0 rgba(239,68,68,.7)} 70% {box-shadow: 0 0 0 10px rgba(239,68,68,0)}
  100% {box-shadow: 0 0 0 0 rgba(239,68,68,0)}}
.gb-grid {display: grid; grid-template-columns: repeat(auto-fill, minmax(118px, 1fr)); gap: 10px; margin-top: 6px;}
.gb-table {width: 100%; border-collapse: collapse; font-size: .78rem;}
.gb-table th {text-align: left; color: #8b93b8; font-weight: 600; padding: 4px 6px; border-bottom: 1px solid rgba(255,255,255,.12);}
.gb-table td {padding: 4px 6px; border-bottom: 1px solid rgba(255,255,255,.06); color: #dfe4ff; vertical-align: top;}
.gb-side-title {font-weight: 800; font-size: 1.05rem; margin: 14px 0 2px; color: #fff;}
.gb-grid.side {grid-template-columns: repeat(3, 1fr); gap: 6px; margin-bottom: 14px;}
.gb-grid.side .gb-tile {padding: 7px 3px 6px; border-radius: 12px;}
.gb-grid.side .e {font-size: 1.5rem;} .gb-grid.side .n {font-size: .78rem;} .gb-grid.side .h {font-size: .6rem;}
.gb-chip.spell {background: linear-gradient(135deg, rgba(244,114,182,.55), rgba(251,191,36,.40));}
.gb-tile {text-align: center; padding: 12px 8px 10px; border-radius: 16px; background: rgba(255,255,255,.05);
  border: 1px solid rgba(255,255,255,.08); transition: transform .15s ease, border-color .15s ease;}
.gb-tile:hover {transform: translateY(-3px) scale(1.04); border-color: rgba(167,139,250,.7);
  box-shadow: 0 8px 24px rgba(139,92,246,.30);}
.gb-tile .e {font-size: 2.2rem; line-height: 1.2;} .gb-tile .n {font-weight: 800; color: #fff; font-size: .92rem;}
.gb-tile .h {font-size: .7rem; color: #8b93b8; line-height: 1.3; margin-top: 2px;}
div.stButton > button, div.stDownloadButton > button {border-radius: 12px; font-weight: 700;}
div.stButton > button[kind="primary"] {background: linear-gradient(135deg,#8b5cf6,#06b6d4); border: 0;
  box-shadow: 0 6px 22px rgba(139,92,246,.45);}
div.stButton > button[kind="primary"]:hover {filter: brightness(1.12);}
[data-testid="stTabs"] button[role="tab"] p {font-weight: 700; font-size: .95rem;}
[data-testid="stImage"] img {border-radius: 14px;}
[data-testid="stMetricValue"] {font-weight: 800;}
</style>
"""
if settings.ui.large_text:
    css += """<style>
html, body, [class*="st-"] {font-size: 1.12rem;}
.gb-sign {font-size: 3rem;} .gb-sentence {font-size: 3.2rem;} .gb-chip {font-size: 1.35rem;} .gb-hindi {font-size: 2.3rem;}
</style>"""
if settings.ui.high_contrast:
    css += """<style>
[data-testid="stAppViewContainer"], [data-testid="stHeader"] {background: #000 !important; color: #fff !important;}
[data-testid="stAppViewContainer"] p, [data-testid="stAppViewContainer"] h1, [data-testid="stAppViewContainer"] h2,
[data-testid="stAppViewContainer"] h3, [data-testid="stAppViewContainer"] label {color: #fff !important;}
.gb-card {border: 2px solid #ffd400; background: #000; box-shadow: none;}
.gb-sentence, .gb-title {background: none; color: #ffd400 !important; -webkit-text-fill-color: #ffd400; animation: none;}
.gb-sign {color: #ffd400;} .gb-chip {background: #000; border: 2px solid #fff; color: #fff; box-shadow: none;}
.gb-label, .gb-muted {color: #fff;} .gb-msg-warning {color: #ffd400;} .gb-msg-success {color: #7CFC00;}
.gb-video {background: #ffd400; box-shadow: none;}
</style>"""
st.markdown(css, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML snippets for the live panels
# ---------------------------------------------------------------------------

def esc(text) -> str:
    return html.escape(str(text))


def sign_html(state: StabiliserOutput) -> str:
    if agent.classifier is None:
        body = "<div class='gb-muted'>No sign model loaded.</div>"
    elif not state.hand_present:
        body = "<div class='gb-sign' style='opacity:.5'><span class='gb-emoji'>🖐️</span>show your hand</div>"
    elif not state.label or state.label in vocab.ignored:
        body = ("<div class='gb-sign' style='opacity:.5'><span class='gb-emoji'>🤔</span>…</div>"
                "<div class='gb-muted'>no sign yet – hold a handshape from the chart</div>")
    else:
        sure = state.confidence >= settings.recognition.threshold
        confidence, held = round(state.confidence * 20) / 20, round(state.hold_progress * 10) / 10
        others = "".join(f"<span class='gb-mini'>{EMOJI.get(l, '✋')} {esc(vocab.display(l))} {round(p * 10) / 10:.0%}</span>"
                         for l, p in state.top[1:3] if l not in vocab.ignored)
        status = "hold still …" if state.still else "hand moving"
        body = (f"<div class='gb-sign'><span class='gb-emoji'>{EMOJI.get(state.label, '✋')}</span>"
                f"{esc(vocab.display(state.label))}<span class='gb-conf'>{confidence:.0%}</span></div>"
                f"<div class='gb-bar{'' if sure else ' unsure'}'><div style='width:{held * 100:.0f}%'></div></div>"
                f"<div class='gb-muted'>{status} · {held:.0%} held {others}</div>")
    return f"<div class='gb-card'><div class='gb-label'>Current sign</div>{body}</div>"


def words_html() -> str:
    chips = "".join(f"<span class='gb-chip'>{EMOJI.get(e.gloss, '✋')} "
                    f"{'?' if e.gloss == 'QUESTION' else esc(vocab.display(e.gloss))}</span>" for e in agent.buffer)
    if agent.spelling:
        chips += f"<span class='gb-chip spell'>✍️ {esc(''.join(agent.spelling))}…</span>"
    empty = "<span class='gb-muted'>sign some words, then 👌 SPEAK (or the 🔊 Speak button)</span>"
    return f"<div class='gb-card'><div class='gb-label'>Words so far</div>{chips or empty}</div>"


def sentence_html(tr: Translation | None) -> str:
    if tr is None or tr.result is None:
        return "<div class='gb-card'><div class='gb-label'>Sentence</div><div class='gb-muted'>nothing yet</div></div>"
    r = tr.result
    if tr.status == "pending":
        main = f"<div class='gb-sentence' style='opacity:.6'>{esc(r.rule_sentence)}</div><div class='gb-muted'>polishing with the LLM…</div>"
    else:
        main = f"<div class='gb-sentence'>{esc(r.final_sentence)}</div>"
        details = [f"signs: {esc(' '.join(r.glosses))}"]
        if r.decode.changed:
            details.append(f"beam search corrected: {esc(' '.join(r.decode.greedy))}")
        details.append(f"rules: {esc(r.rule_sentence)}")
        if r.refine:
            details.append(f"LLM: {esc(r.refine.status)} – {esc(r.refine.reason)}")
        if tr.latency_ms.get("speech_start"):
            details.append(f"speech started after {tr.latency_ms['speech_start']:.0f} ms")
        if tr.error:
            details.append(f"error: {esc(tr.error)}")
        if tr.speech and not tr.speech.ok and tr.speech.message:
            details.append(f"speech: {esc(tr.speech.message)}")
        if tr.audio_error:
            details.append(f"speech: {esc(tr.audio_error)}")
        main += "<div class='gb-muted'>" + "<br>".join(details) + "</div>"
        if r.hindi:
            main += (f"<div class='gb-hindi' lang='hi'>{esc(r.hindi.text)}</div>" if r.hindi.text
                     else f"<div class='gb-muted'>Hindi not available: {esc(r.hindi.reason)}</div>")
    return f"<div class='gb-card'><div class='gb-label'>Sentence</div>{main}</div>"


def messages_html() -> str:
    items = list(agent.messages)[-4:][::-1]
    if not items:
        return ""
    lines = "".join(f"<div class='gb-msg-{level}'>{esc(text)}</div>" for _, text, level in items)
    return f"<div class='gb-card'><div class='gb-label'>Messages</div>{lines}</div>"


def trace_frame(result: TranslationResult | None) -> pd.DataFrame:
    if result is None:
        return pd.DataFrame()
    return pd.DataFrame([{"#": s.cycle, "rule": s.rule_id, "name": s.rule_name, "why": s.explanation,
                          "sentence after": s.result} for s in result.derivation.trace])


def html_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    head = "".join(f"<th>{esc(c)}</th>" for c in frame.columns)
    rows = "".join("<tr>" + "".join(f"<td>{esc(v)}</td>" for v in row) + "</tr>" for row in frame.itertuples(index=False))
    return f"<table class='gb-table'><tr>{head}</tr>{rows}</table>"


def beam_frame(result: TranslationResult | None) -> pd.DataFrame:
    if result is None or not result.decode.beam:
        return pd.DataFrame()
    return pd.DataFrame([h.as_dict() for h in result.decode.beam])


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------

st.markdown(
    "<div class='gb-hero'><div class='gb-logo'>🤟</div><div><div class='gb-title'>GestureBridge</div>"
    "<div class='gb-tag'>Sign language → sentence → speech, live in your browser.</div></div></div>"
    + "".join(f"<span class='gb-pill'>{t}</span>" for t in (
        "✋ 58 instant signs", "🧠 RF · SVM · MLP", "🔎 beam search", "📚 48 grammar rules", "🦙 local LLM + guard",
        "🔊 speech EN / हिंदी", "🔒 private: nothing leaves this laptop")),
    unsafe_allow_html=True)

tab_live, tab_rec, tab_lab, tab_dash, tab_hist, tab_rev, tab_help = st.tabs(
    ["🤟 Live translator", "🎬 Record & train", "🧪 Sentence lab", "📊 Model dashboard", "📜 History & live test",
     "🎙️ Reverse mode", "ℹ️ Help & about"])

# ----- Live translator ---------------------------------------------------------
with tab_live:
    if agent.classifier is None:
        st.info("No sign model loaded – choose one in the sidebar. The **Sentence lab** works without a model.")
    left, right = st.columns([3, 2], gap="large")
    with left:
        b1, b2, b3, b4, b5 = st.columns(5)

        def toggle_camera():
            # a callback runs before the rerun, so the button label already shows the new state
            if engine.running:
                engine.stop()
            else:
                release_camera()                     # the recording tab may still hold the camera
                try:
                    engine.start(agent, settings)
                except RuntimeError as exc:
                    ss.camera_error = str(exc)

        b1.button("⏹ Stop" if engine.running else "▶ Start", type="primary", width="stretch",
                  on_click=toggle_camera, disabled=agent.classifier is None, help="Start or stop the camera")
        if b2.button("🔊 Speak", width="stretch", help="Translate the words and speak (same as the OK sign)"):
            if agent.buffer or agent.spelling:
                agent.finalize(source="button")
            elif agent.latest:
                replay_last()
            else:
                st.toast("Nothing to say yet – sign some words (or add one below) first.", icon="✋")
        if b3.button("⌫ Delete", width="stretch", help="Remove the last word"):
            agent.delete_last()
        if b4.button("🗑 Clear", width="stretch"):
            agent.clear()
        if b5.button("❓ Ask", width="stretch", help="Make the sentence a yes/no question"):
            agent.add_gloss("QUESTION")
        if ss.get("camera_error"):
            st.error(ss.pop("camera_error"))
        elif not engine.running and engine.notice:
            st.caption("🔒 " + engine.notice + " It is only used while this page is open and visible.")
        if engine.running:
            # the browser plays the camera as a live MJPEG stream - smooth, no page updates needed
            st.markdown(f"<div class='gb-video'><span class='gb-badge'>● LIVE</span>"
                        f"<img src='{engine.url}' alt='camera'></div>", unsafe_allow_html=True)
            components.html(PAUSE_WHEN_HIDDEN_JS, height=0)
        else:
            st.info("Camera is off. Press **▶ Start**, sign words, then make the **OK** sign (SPEAK) or press "
                    "**🔊 Speak**." if agent.classifier else "Choose a sign model in the sidebar to use the camera.")
        with st.expander("Backup: add a word without signing"):
            c1, c2 = st.columns([3, 1])
            manual = c1.selectbox("Word", vocab.words, label_visibility="collapsed", format_func=vocab.display)
            if c2.button("Add", width="stretch"):
                agent.add_gloss(manual)

    def play_audio(tr) -> None:
        if tr is None or tr.status == "pending":
            return
        if tr.audio:
            key = (tr.id, ss.get("replay", 0))
            fresh = ss.get("audio_played") != key          # play each sentence once; Repeat plays it again
            ss.audio_played = key
            st.markdown(audio_html(*tr.audio, autoplay=fresh, tag=f"{key[0]}-{key[1]}"), unsafe_allow_html=True)
        elif tr.audio_error:
            st.warning(f"Speech: {tr.audio_error}")

    waiting = agent.latest is not None and not agent.latest.done.is_set()

    @st.fragment(run_every=0.3 if engine.running or waiting else None)
    def live_panels():
        """Refreshes only this column a few times per second (not the whole page)."""
        if engine.error:
            st.error(engine.error)
        latest = agent.latest
        result = latest.result if latest else None
        st.markdown(sign_html(agent.last_state), unsafe_allow_html=True)
        st.markdown(words_html(), unsafe_allow_html=True)
        st.markdown(sentence_html(latest), unsafe_allow_html=True)
        play_audio(latest)
        st.markdown(messages_html(), unsafe_allow_html=True)
        if engine.running:
            st.caption(f"{engine.fps:.1f} frames per second")
        # small HTML tables: cheap to redraw several times per second (dataframes were not)
        with st.expander("Why this sentence? (rule trace)"):
            st.markdown(html_table(trace_frame(result)) or "<span class='gb-muted'>No rules fired yet.</span>",
                        unsafe_allow_html=True)
        with st.expander("Beam search details"):
            st.markdown(html_table(beam_frame(result)) or
                        "<span class='gb-muted'>Beam search runs when a sentence is translated.</span>",
                        unsafe_allow_html=True)

    with right:
        live_panels()
        tester = st.text_input("Tester name (for the live test)", key="tester", placeholder="e.g. Rahul")
        r2, r3, r4 = st.columns(3)
        rate_good = r2.button("✅ Correct", width="stretch", help="The sentence was what the signer meant")
        rate_bad = r3.button("❌ Wrong", width="stretch", help="The sentence was not what the signer meant")
        if r4.button("🔁 Repeat", width="stretch", help="Speak the last sentence again"):
            replay_last()
            st.rerun()
        if rate_good or rate_bad:
            latest = agent.latest
            if latest and latest.done.is_set():
                latest.feedback, latest.tester = ("correct" if rate_good else "wrong"), tester
                history.set_feedback(latest.id, latest.feedback, tester)
                st.toast(f"Saved: {latest.feedback}")
            else:
                st.toast("No finished sentence to rate yet.")

# ----- Record & train -----------------------------------------------------------
with tab_rec:
    st.subheader("Record your signs and train your own model")
    st.caption("Only the 21 hand-landmark numbers per frame are saved (data/raw/landmarks.csv) - never images. "
               "Record every sign 2-3 times; every team member should record. Key-poses: docs/SIGN_GUIDE.md.")
    c1, c2, c3 = st.columns(3)
    rec_signer = c1.text_input("Your name (signer)", key="rec_signer", placeholder="e.g. nancy")
    rec_condition = c2.selectbox("Lighting / background", CONDITIONS, key="rec_condition")
    rec_samples = c3.slider("Frames per recording", 30, 200, 80, step=10, key="rec_samples")
    with st.expander("➕ Add your own word – you choose the word and record its sign"):
        a1, a2 = st.columns(2)
        new_word = a1.text_input("Word", key="new_word", placeholder="e.g. mother, school, eat")
        new_type = a2.selectbox("Type of word (for the grammar rules)", list(WORD_TYPES), key="new_type",
                                format_func=WORD_TYPES.get)
        new_how = st.text_input("How you make the sign (optional, shown in the sign guide)", key="new_how",
                                placeholder="e.g. flat hand on the chest")
        if st.button("Add word", key="add_word"):
            item, reason = add_custom(new_word, new_type, new_how, vocab.labels, list(ALL_SIGNS),
                                      agent.pipeline.kb)
            if reason:
                st.warning("⚠️ " + reason)
            else:
                vocab.add(Sign(item["label"], "word", item["how"], "custom word"))
                agent.pipeline.kb.add_custom(item)
                ss.rec_sign = item["label"]
                ss.rec_message = ("success", f"Added the word '{item['text']}'. It is selected below: record its "
                                             "sign 3 times, then train.")
                st.rerun()
    record_labels = vocab.labels + list(HANDSHAPES["letters"]) + list(HANDSHAPES["numbers"])
    rec_sign = st.selectbox("Sign to record", record_labels, key="rec_sign",
                            format_func=lambda label: f"{vocab.display(label)}  ({record_kind(label)})")
    st.info(f"**Key-pose for {vocab.display(rec_sign)}:** {keypose_text(rec_sign)}  \n"
            "Press Record, get ready during the 2-second countdown, then hold the pose and move the hand a "
            "little (closer, further, slightly tilted). Recording your own handshape for an existing sign "
            "replaces the built-in one in **Instant + your trained signs**.")
    k1, k2, _ = st.columns(3)
    if not rec_signer.strip():
        st.warning("✍️ Type your name in **Your name (signer)** at the top of this tab – the **⏺ Record** "
                   "button switches on as soon as a name is there (after reloading the page, type it again).")
    if k1.button("⏺ Record", type="primary", width="stretch", disabled=not rec_signer.strip()):
        engine.stop()
        ss.record_request = {"signer": rec_signer.strip().lower().replace(" ", "_"), "label": rec_sign,
                             "condition": rec_condition, "samples": int(rec_samples)}
    last = ss.get("last_burst")
    if k2.button("↩ Undo last recording", width="stretch", disabled=last is None):
        removed = remove_burst(last["burst"])
        st.toast(f"Removed {removed} samples of {vocab.display(last['label'])}")
        ss.last_burst = None
    rec_video = st.empty()
    rec_status = st.empty()

    data = recorded_counts()
    if data.empty:
        st.caption("Nothing recorded yet.")
    else:
        mine = data[data["signer"] == rec_signer.strip().lower().replace(" ", "_")]
        st.dataframe(pd.DataFrame({
            "sign": [vocab.display(label) for label in vocab.labels],
            "your samples": [int((mine["label"] == label).sum()) for label in vocab.labels],
            "all samples": [int((data["label"] == label).sum()) for label in vocab.labels],
            "recordings": [int(data[data["label"] == label]["burst"].nunique()) for label in vocab.labels],
        }), hide_index=True, width="stretch", height=300)
    st.divider()
    recorded_labels = sorted(data["label"].unique()) if not data.empty else []
    st.markdown(f"**Train** - {len(recorded_labels)} signs recorded so far. Training compares Random Forest, "
                "SVM and MLP, picks the best and fills the Model dashboard (about 1-3 minutes).")
    if len(recorded_labels) < 2:
        st.caption("Record at least two different signs (ideally all of them, plus NOTHING) before training.")
    if st.button("🧠 Train models", type="primary", disabled=len(recorded_labels) < 2):
        engine.stop()
        ss.train_request = True

# ----- Sentence lab -----------------------------------------------------------
with tab_lab:
    st.subheader("Sentence lab – the language pipeline without a camera")
    st.caption("Type sign glosses in signing order. Simulate an unsure recogniser with alternatives: "
               "`I WANT:0.46/HUNGRY:0.44/FOOD:0.10` means the second sign was WANT 46 %, HUNGRY 44 % or FOOD 10 %.")
    examples = ["I HUNGRY", "WATER I WANT", "I GO HOME NO", "YOU HELP I", "HELLO I HUNGRY", "HELP PLEASE",
                "YOU WANT WATER QUESTION", "I WANT:0.46/HUNGRY:0.44/FOOD:0.10", "I WANT HUNGRY:0.5/WATER:0.45"]

    def use_example():
        if ss.lab_example:
            ss.lab_text = ss.lab_example

    st.selectbox("Examples", [""] + examples, key="lab_example", on_change=use_example,
                 format_func=lambda x: x or "choose an example …")
    st.text_input("Signs", key="lab_text")
    word_cols = st.columns(9)
    for i, word in enumerate(vocab.words + ["QUESTION"]):
        def add(w=word):
            ss.lab_text = (ss.lab_text + " " + w).strip()
        word_cols[i % 9].button(vocab.display(word) if word != "QUESTION" else "?", key=f"lab_add_{word}",
                                on_click=add, width="stretch")

    o1, o2, o3, o4 = st.columns(4)
    lab_llm = o1.toggle("Use LLM", value=settings.language.use_llm, key="lab_llm")
    lab_beam = o2.toggle("Beam search", value=settings.language.use_beam, key="lab_beam")
    lab_lang = o3.radio("Language", ["en", "hi"], horizontal=True, key="lab_lang",
                        format_func=lambda x: "English" if x == "en" else "Hindi")
    def clear_lab():
        ss.lab_text = ""
        ss.lab_result = None

    o4.button("Clear signs", width="stretch", on_click=clear_lab)

    if st.button("Translate", type="primary"):
        slots = parse_gloss_text(ss.lab_text)
        if slots:
            pipeline = agent.pipeline
            use_beam_before = pipeline.settings.use_beam
            pipeline.settings.use_beam = lab_beam
            try:
                with st.spinner("Translating …"):
                    ss.lab_result = pipeline.translate(slots, use_llm=lab_llm, language=lab_lang)
            finally:
                pipeline.settings.use_beam = use_beam_before
        else:
            st.warning("Type some signs first.")

    result: TranslationResult | None = ss.lab_result
    if result:
        a, b, c = st.columns(3)
        a.metric("Recognised (top-1)", " ".join(result.decode.greedy))
        b.metric("After beam search", " ".join(result.glosses), "changed" if result.decode.changed else None)
        c.metric("Rules fired", len(result.derivation.trace))
        st.markdown(f"<div class='gb-card'><div class='gb-label'>Final sentence</div>"
                    f"<div class='gb-sentence'>{esc(result.final_sentence)}</div>"
                    + (f"<div class='gb-hindi' lang='hi'>{esc(result.hindi.text)}</div>" if result.hindi and result.hindi.text else "")
                    + "</div>", unsafe_allow_html=True)
        if result.hindi and not result.hindi.text:
            st.warning(f"Hindi not available: {result.hindi.reason}")
        if st.button("🔊 Speak this sentence", type="primary"):
            if settings.speech.output == "browser":
                with st.spinner("Speaking …"):
                    data, info = speaker.synthesize(result.spoken_text, result.spoken_language)
                if data:
                    ss.lab_audio_count = ss.get("lab_audio_count", 0) + 1
                    st.markdown(audio_html(data, info, autoplay=True, tag=str(ss.lab_audio_count)),
                                unsafe_allow_html=True)
                else:
                    st.warning(f"Speech: {info}")
            else:
                outcome = speaker.speak(result.spoken_text, result.spoken_language)
                if not outcome.ok:
                    st.warning(outcome.message)

        k1, k2 = st.columns([2, 3])
        with k1:
            st.markdown("**1 · Knowledge base facts** (lexicon)")
            st.dataframe(pd.DataFrame([{"sign": g, "facts": agent.pipeline.kb.describe(g)} for g in result.glosses]),
                         hide_index=True, width="stretch")
            st.markdown("**2 · Beam search** (score = classifier log-prob + λ · grammar log-prob)")
            st.dataframe(beam_frame(result), hide_index=True, width="stretch")
        with k2:
            st.markdown("**3 · Forward chaining** – rules fired in order")
            frame = trace_frame(result)
            if frame.empty:
                st.caption("No rule needed to fire – the signs already form a sentence.")
            else:
                st.dataframe(frame, hide_index=True, width="stretch")
            st.markdown(f"**Rule-based sentence:** {esc(result.rule_sentence)}")
            if result.refine:
                st.markdown(f"**4 · LLM** – status `{result.refine.status}`: {esc(result.refine.reason)}")
                if result.refine.raw and result.refine.raw != result.final_sentence:
                    st.caption(f"LLM said: “{result.refine.raw}” → not used")
            st.caption("Timing: " + ", ".join(f"{k} {v:.1f} ms" for k, v in result.timings_ms.items()))

    with st.expander("Run the whole test set (rule engine)"):
        if st.button("Run test sentences"):
            rows = []
            for case in load_cases():
                sentence = agent.pipeline.engine.run(case["glosses"].split()).sentence
                rows.append({"signs": case["glosses"], "rule engine": sentence, "expected": case["expected"][0],
                             "correct": any(same_sentence(sentence, e) for e in case["expected"])})
            frame = pd.DataFrame(rows)
            st.metric("Correct sentences", f"{frame['correct'].mean():.1%}", f"{frame['correct'].sum()} / {len(frame)}")
            st.dataframe(frame, hide_index=True, width="stretch")

# ----- Model dashboard ---------------------------------------------------------
with tab_dash:
    if not METRICS_PATH.exists():
        st.info("No training report yet. Record signs and train:\n\n"
                "```\npython -m gesturebridge.collect --signer <your-name>\npython -m gesturebridge.train\n```")
    else:
        doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        best = doc["models"][doc["best_model"]]
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Samples", f"{doc['dataset']['samples']:,}")
        m2.metric("Signs", len(doc["labels"]))
        m3.metric("Signers", len(doc["dataset"]["signers"]))
        m4.metric("Best model", best["name"], f"macro-F1 {best['macro_f1']:.3f}", delta_color="off")
        st.caption(f"Trained {doc['created']} · {doc['split']['strategy']} split "
                   f"({doc['split']['train_samples']} train / {doc['split']['test_samples']} test) · "
                   f"augmentation ×{doc['augment_copies']}")
        if doc["split"]["single_burst_labels"]:
            st.warning("Only one recording burst for: " + ", ".join(doc["split"]["single_burst_labels"]) +
                       " – their test accuracy is optimistic. Record a second burst for each.")

        table = pd.DataFrame([{
            "model": r["name"], "test accuracy": round(r["accuracy"], 4), "test macro-F1": round(r["macro_f1"], 4),
            "CV macro-F1": f"{r['cv']['macro_f1_mean']:.3f} ± {r['cv']['macro_f1_std']:.3f}",
            "ms / frame": round(r["latency_ms"], 2), "train s": r["train_seconds"]} for r in doc["models"].values()])
        st.dataframe(table, hide_index=True, width="stretch")

        def figure(name: str):
            path = REPORTS_DIR / doc["figures"].get(name, "")
            if name in doc["figures"] and path.exists():
                st.image(str(path), width="stretch")

        f1, f2 = st.columns(2)
        with f1:
            figure("comparison")
        with f2:
            figure("latency")
        conf_tabs = st.tabs([doc["models"][k]["name"] for k in doc["models"]])
        for tab, key in zip(conf_tabs, doc["models"]):
            with tab:
                figure(f"confusion_{key}")
                confusions = doc["models"][key]["top_confusions"]
                if confusions:
                    st.dataframe(pd.DataFrame(confusions), hide_index=True, width="stretch")
        p1, p2 = st.columns(2)
        with p1:
            figure("per_class")
        with p2:
            st.dataframe(pd.DataFrame(best["per_class"]).T.reset_index(names="sign"), hide_index=True, width="stretch")
        if doc.get("loso"):
            st.subheader("Generalisation to new people (leave-one-signer-out)")
            figure("loso")
            st.dataframe(pd.DataFrame({doc["models"][k]["name"]: {s: v["accuracy"] for s, v in per.items()}
                                       for k, per in doc["loso"].items()}), width="stretch")
        if doc.get("groups"):
            st.subheader("Fairness check: accuracy by group (best model, held-out test)")
            g_cols = st.columns(len(doc["groups"]))
            for col, (column, values) in zip(g_cols, doc["groups"].items()):
                col.markdown(f"**{column}**")
                col.dataframe(pd.DataFrame(values).T, width="stretch")
        if doc.get("feature_importance"):
            st.subheader("Which features matter")
            figure("importance")

    st.divider()
    st.subheader("Evaluation reports")
    for name, title in (("sentence_eval", "Sentence correctness"), ("decoder_eval", "Greedy vs beam search"),
                        ("latency", "Latency"), ("prompt_eval", "Prompt engineering"), ("live_eval", "Live test")):
        md_path = REPORTS_DIR / f"{name}.md"
        if md_path.exists():
            with st.expander(title):
                text = md_path.read_text(encoding="utf-8")
                st.markdown("\n".join(l for l in text.splitlines() if not l.startswith("![")))
                if name == "decoder_eval" and (FIGURES_DIR / "decoder_robustness.png").exists():
                    st.image(str(FIGURES_DIR / "decoder_robustness.png"))
    st.caption("Create or refresh them with `python -m gesturebridge.evaluate all`.")

# ----- History & live test ------------------------------------------------------
with tab_hist:
    frame = history.load()
    if frame.empty:
        st.info("No translations yet. Every sentence from the Live translator is logged here (signs, sentence, "
                "LLM outcome, latency). Testers rate sentences with ✅ / ❌ in the Live translator.")
    else:
        rated = frame[frame["feedback"].isin(["correct", "wrong"])]
        h1, h2, h3, h4 = st.columns(4)
        h1.metric("Sentences", len(frame))
        h2.metric("Rated by testers", len(rated))
        h3.metric("Judged correct", f"{(rated['feedback'] == 'correct').mean():.0%}" if len(rated) else "–")
        speech = pd.to_numeric(frame["latency_speech_ms"], errors="coerce").replace(0, pd.NA).dropna()
        h4.metric("Median time to speech", f"{speech.median():.0f} ms" if len(speech) else "–")
        if len(rated):
            per = rated.assign(tester=rated["tester"].replace("", "unnamed")).groupby("tester")["feedback"]
            st.dataframe(pd.DataFrame({"rated": per.size(), "accuracy": per.apply(lambda s: (s == "correct").mean())}),
                         width="stretch")
        st.dataframe(frame.iloc[::-1], hide_index=True, width="stretch",
                     column_order=["timestamp", "signed", "decoded", "final_sentence", "rule_sentence", "llm_status",
                                   "hindi", "latency_speech_ms", "feedback", "tester", "rules_fired", "source"])
        d1, d2 = st.columns([1, 1])
        d1.download_button("Download history (CSV)", frame.to_csv(index=False).encode("utf-8"),
                           "gesturebridge_history.csv", "text/csv")
        confirm = d2.checkbox("I want to delete the history")
        if d2.button("Delete history", disabled=not confirm):
            history.clear()
            st.rerun()

# ----- Reverse mode -------------------------------------------------------------
with tab_rev:
    st.subheader("Reverse mode – speech to text for the signer")
    st.warning("Privacy: this uses Google's free speech recognition, so the recording is sent to Google over the "
               "internet. Record only with the speaker's consent. Everything else in GestureBridge runs offline.")
    spoken_lang = st.radio("Spoken language", ["en-IN", "hi-IN"], horizontal=True,
                           format_func=lambda x: "English (India)" if x == "en-IN" else "Hindi")
    recording = st.audio_input("Record what the hearing person says")
    if recording is not None:
        with st.spinner("Transcribing …"):
            text, error = transcribe(recording.getvalue(), spoken_lang)
        if error:
            st.error(error)
        else:
            st.markdown(f"<div class='gb-card'><div class='gb-label'>They said</div>"
                        f"<div class='gb-sentence'>{esc(text)}</div></div>", unsafe_allow_html=True)
            spoken_words = {w.strip(".,!?").upper() for w in text.split()}
            known = [w for w in vocab.words if w in spoken_words or w.replace("_", " ") in text.upper()]
            if known:
                st.markdown("**Signs you can answer with:**")
                st.dataframe(pd.DataFrame([{"sign": vocab.display(w), "key-pose": vocab.get(w).keypose} for w in known]),
                             hide_index=True, width="stretch")

# ----- Help & about -------------------------------------------------------------
with tab_help:
    st.subheader("How GestureBridge works")
    st.graphviz_chart("""
digraph {
  rankdir=LR; node [shape=box, style="rounded,filled", fillcolor="#eef3fb", color="#2a78d6", fontname="Segoe UI"];
  cam [label="Camera\\n(sensor)"]; mp [label="MediaPipe\\n21 hand landmarks"]; feat [label="Feature engineering\\n106 features"];
  clf [label="Classifier\\nRF / SVM / MLP"]; stab [label="Stabiliser\\nsmoothing, hold, top-3"];
  buf [label="Word buffer\\n(agent state)"]; beam [label="Beam search\\nheuristic decoding"];
  rules [label="Knowledge base +\\nforward chaining"]; llm [label="Local LLM\\n+ meaning guard"]; tts [label="Text-to-speech\\n(actuator)"];
  cam -> mp -> feat -> clf -> stab -> buf -> beam -> rules -> llm -> tts;
}""")
    st.markdown("""
| PEAS | GestureBridge |
|---|---|
| **Performance** | sign accuracy, sentence correctness, time from sign to speech |
| **Environment** | one signer in front of a webcam; changing light, background and distance |
| **Actuators** | screen (skeleton, words, sentence, rule trace) and speaker |
| **Sensors** | webcam frames → 21 hand landmarks |

**Agent type:** model-based agent – it keeps internal state (smoothing window, hold timer, word buffer) that a single frame does not show.

| Where is the AI? | Technique | Course unit |
|---|---|---|
| Hand landmarks | pretrained CNN (MediaPipe) | computer vision (used as a sensor) |
| Sign classifier | our dataset + features; RF vs SVM vs MLP | machine learning |
| Stabiliser | probability smoothing, thresholds, entropy, top-k | reasoning under uncertainty |
| Beam search | heuristic search over top-3 candidates (constraint optimisation over word slots) | search, CSP |
| Grammar rules | knowledge base + forward-chaining inference engine with explanations (an expert-system design) | knowledge representation & reasoning, expert systems |
| LLM | local Transformer LLM (Ollama), few-shot prompts, LangChain, meaning guard | Transformers, LLMs, prompt engineering, local LLMs, privacy |
| Responsible AI | bias tests by signer/lighting/hand, privacy by design, guardrails, honest limits | responsible AI |
""")
    st.subheader("Signs and their key-poses")
    st.dataframe(pd.DataFrame([{"sign": vocab.display(s.label), "type": s.type, "key-pose": s.keypose, "note": s.note}
                               for s in vocab.signs]), hide_index=True, width="stretch")
    st.subheader("Commands")
    st.code("""python -m gesturebridge.collect --signer <name>     # record landmarks for each sign
python -m gesturebridge.train                       # train + compare RF / SVM / MLP, write reports
streamlit run app.py                                # this app
python -m gesturebridge.desktop                     # OpenCV version (backup for the demo)
python -m gesturebridge.evaluate all                # sentence, decoder, latency and prompt evaluations""", language="bash")
    st.markdown("**Responsible AI.** No video is stored – only landmark numbers. The LLM runs locally. "
                "The LLM can only polish wording; a guard rejects answers that add, drop or negate meaning. "
                "Accuracy is reported per signer and per lighting condition. GestureBridge is an assistive "
                "prototype with a small vocabulary and no facial grammar – it does not replace human interpreters. "
                "See `docs/RESPONSIBLE_AI.md`.")


# ---------------------------------------------------------------------------
# Recording and training (run last, so every tab above is already drawn)
# ---------------------------------------------------------------------------

def record_burst(request: dict) -> None:
    # the preview is the same smooth MJPEG stream as the live view: frames go to the stream server,
    # not through the page (that made the recording preview lag). It is shown at once, while the
    # camera and the hand tracker are still starting.
    engine.ensure_server()
    engine.show(cv2.putText(np.full((360, 640, 3), (36, 22, 18), np.uint8), "Starting the camera ...", (40, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 220, 255), 2, cv2.LINE_AA))
    rec_video.markdown(f"<div class='gb-video'><span class='gb-badge'>● CAMERA</span>"
                       f"<img src='{engine.url}?rec={time.time():.0f}' alt='camera'></div>", unsafe_allow_html=True)
    try:
        camera = ensure_camera()
    except RuntimeError as exc:
        ss.rec_message = ("error", str(exc))
        return
    if "rec_tracker" not in ss:
        with st.spinner("Starting the hand tracker …"):
            ss.rec_tracker = HandTracker(min_detection_confidence=settings.recognition.min_detection_confidence)
    ss.rec_session = ss.get("rec_session") or time.strftime("%Y%m%d-%H%M%S")
    ss.rec_count = ss.get("rec_count", 0) + 1
    burst = f"{request['signer']}-{ss.rec_session}-{ss.rec_count:03d}"
    label, rows = request["label"], []
    countdown_end = time.time() + 2.0
    deadline = countdown_end + max(20.0, request["samples"] / 4)
    last_progress = 0.0
    while len(rows) < request["samples"] and time.time() < deadline:
        ok, frame = camera.read()
        if not ok:
            break
        if settings.camera.mirror:
            frame = cv2.flip(frame, 1)
        hands = ss.rec_tracker.process(frame)
        now = time.time()
        if hands:
            draw_hand(frame, hands[0])
        if now < countdown_end:
            draw_label(frame, f"Get ready: {vocab.display(label)}  {countdown_end - now:.0f}", (10, 32),
                       (60, 220, 255))
        else:
            if hands:
                rows.append(make_row(label, request["signer"], ss.rec_session, burst, request["condition"],
                                     hands[0], now))
            draw_label(frame, f"REC {vocab.display(label)} {len(rows)}/{request['samples']}"
                       + ("" if hands else "  - show your hand!"), (10, 32), (60, 60, 230))
        engine.show(frame)
        if now - last_progress > 0.25:        # a few page updates per second are enough for the bar
            rec_status.progress(len(rows) / request["samples"], text=f"{len(rows)} / {request['samples']} frames")
            last_progress = now
    engine.show(cv2.putText(np.full((360, 640, 3), (36, 22, 18), np.uint8), "Saved - see below", (40, 190),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (230, 220, 255), 2, cv2.LINE_AA))
    release_camera()                  # hand the camera back to the live translator
    if rows:
        append_rows(rows)
        ss.last_burst = {"burst": burst, "label": label, "n": len(rows)}
        warning = conflict_warning(rows, label)
        ss.rec_message = (("warning", warning) if warning else
                          ("success", f"Saved {len(rows)} samples of {vocab.display(label)}."))
    else:
        ss.rec_message = ("error", "No hand was seen - nothing saved. Check the light and keep the hand in view.")


if ss.get("record_request"):
    record_burst(ss.pop("record_request"))
    st.rerun()
if ss.get("rec_message"):
    kind, text = ss.pop("rec_message")
    {"success": rec_status.success, "warning": rec_status.warning}.get(kind, rec_status.error)(text)

if ss.pop("train_request", False):
    with tab_rec, st.status("Training Random Forest, SVM and MLP …", expanded=True) as status:
        try:
            doc = train_models(augment=1, log=lambda line: status.write(line))
            best = doc["models"][doc["best_model"]]
            status.update(label=f"Done - best: {best['name']} (macro-F1 {best['macro_f1']:.3f}). "
                                "See the Model dashboard.", state="complete")
            ss.select_best_model = True
            status.write(f"'{HYBRID_CHOICE}' is now selected: your {len(doc['labels'])} trained signs plus "
                         "every other built-in sign.")
        except Exception as exc:  # show problems such as too little data in the page
            status.update(label=f"Training failed: {exc}", state="error")
    if ss.get("select_best_model"):
        time.sleep(1.5)
        st.rerun()
