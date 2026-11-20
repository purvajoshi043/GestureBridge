# GestureBridge 🤟 - AI-powered sign language to speech

GestureBridge watches a signer through an ordinary webcam, recognises their signs,
turns them into a correct English (or Hindi) sentence and speaks it aloud. It runs on a
normal laptop, offline, and explains every decision it makes.

```
camera → MediaPipe hand landmarks → 106 features → sign classifier (RF / SVM / MLP)
       → stabiliser (uncertainty) → word buffer → beam search (heuristic search)
       → knowledge base + forward chaining (grammar rules) → local LLM + meaning guard
       → text-to-speech
```

**Team:** Nancy Yadav (240710), Purva (240727), Tanisha (240667) ·
**Course:** Artificial Intelligence, BML Munjal University · **Instructor:** Saurabh Mishra

**Project report:** [report/GestureBridge_Report.pdf](report/GestureBridge_Report.pdf)
(also as [Word](report/GestureBridge_Report.docx)) · screenshots of the working website are in
[screenshots/](screenshots/).

## Features

* **Live translation** with hand skeleton, confidence bar, top-3 guesses and a hold-progress bar.
* **Instant mode - 58 signs with no training** in three modes (21 words, all 26 fingerspelling letters A-Z to
  spell names, numbers 0-10), with the sign guide in the sidebar: our knowledge-based handshape recogniser
  (`gesturebridge/handshape_rules.py`) turns the landmarks into fuzzy facts (finger straight/curled,
  thumb in/out, pointing direction, fingertips touching) and scores one IF-THEN rule per sign.
* **Your own signs (hybrid mode)**: record your own handshape for any word, letter or number - or add a
  completely new word (you choose the word, its word type and its sign) - and train. "Instant + your
  trained signs" then uses your recordings for those signs and the built-in rules for all others; a
  nearest-neighbour novelty check stops the trained model from claiming signs it never saw
  (`gesturebridge/hybrid.py`, `gesturebridge/custom_signs.py`). The app warns when a word already exists
  or when a new recording is the handshape of another sign.
* **Speech in the browser**: every sentence gets an audio player that plays by itself (or straight to the
  laptop speaker - sidebar → Speech).
* **Control signs**: OK sign = speak, V sign = clear, horns = delete last word (buttons too).
* **Own dataset + model comparison**: data-collection tool, 106 engineered features,
  Random Forest vs SVM vs MLP, honest evaluation (burst-level hold-out, grouped CV,
  leave-one-signer-out), confusion matrices and fairness tables.
* **Reasoning under uncertainty**: probability smoothing, hold time, stillness gate,
  "please repeat" when unsure.
* **Beam search** over the top-3 signs with a grammar prior learned from example sentences.
* **Explainable grammar**: 47 production rules in a JSON knowledge base, run by a
  forward-chaining engine that shows *why* each word was added ("rule trace").
  Handles ISL word order (I WATER WANT → "I want some water."), negation (I HUNGRY NO →
  "I am not hungry."), questions, requests and politeness.
* **Local LLM** (Ollama, `llama3.2:3b`) polishes the sentence - via REST or LangChain,
  three prompt styles - and a **guard** rejects any answer that changes the meaning.
* **Speech**: offline English (pyttsx3), Hindi via gTTS with a hand-checked phrasebook.
* **Sentence lab** (no camera needed), **model dashboard**, **history + live-test rating**,
  **reverse mode** (speech → text for the signer), large-text and high-contrast modes.
* **Privacy**: no images are stored, the LLM is local, the app is reachable only from this computer.

## Quick start (Windows)

1. **Install** (once): double-click `setup.bat`, or:
   ```
   python -m venv .venv
   .venv\Scripts\activate
   python -m pip install -r requirements.txt
   ```
2. **Local LLM** (optional but recommended): install [Ollama](https://ollama.com), then `ollama pull llama3.2:3b`.
   Without Ollama the app still works and uses the rule-based sentence.
3. **Check everything**: `python -m gesturebridge.check`
4. **Record signs** (every team member): in the website's **🎬 Record & train** tab, or with
   `python -m gesturebridge.collect --signer <name>` - see [docs/SIGN_GUIDE.md](docs/SIGN_GUIDE.md).
5. **Train and compare models**: the **🧠 Train models** button in that tab, or `python -m gesturebridge.train`
   (writes `models/` and `reports/`). Until then the app runs in **starter mode** (MediaPipe's pretrained
   gestures) or, by default, **Instant mode** with 58 rule-based signs.
6. **Run the app**: double-click `run_app.bat` (or `streamlit run app.py`, then open http://127.0.0.1:8502).
   Backup without a browser: `python -m gesturebridge.desktop`.
7. **Evaluate for the report**: `python -m gesturebridge.evaluate all` (see below).

All commands are run from the project folder with the environment activated
(`.venv\Scripts\activate`).

## Using the live translator

1. Press **▶ Start camera**.
2. Sign a word and **hold it still** until the bar under your hand fills (≈0.6 s).
   The word appears under *Words so far*. Relax your hand or move to the next sign.
3. Make the **OK sign** (or press **🔊 Speak**). The sentence appears and is spoken.
4. **❓ Question** turns it into a yes/no question; **⌫ Delete / 🗑 Clear** fix mistakes.
5. Testers can rate each sentence with ✅ / ❌ - this is the live-test data for the report.

## How it works

| Stage | File | Technique |
|---|---|---|
| Hand landmarks | `gesturebridge/hand_tracker.py` | MediaPipe HandLandmarker (pretrained) |
| Features | `gesturebridge/features.py` | normalisation, joint angles, distances, palm orientation |
| Classifier | `gesturebridge/train.py`, `recognizer.py` | RF / SVM (+Platt scaling) / MLP, augmentation |
| Stabiliser | `gesturebridge/stabiliser.py` | evidence averaging, thresholds, entropy, hold, motion gate |
| Agent | `gesturebridge/agent.py` | model-based agent (PEAS), sense → think → act |
| Decoder | `gesturebridge/language/decoder.py` | beam search, POS-bigram heuristic |
| Grammar | `knowledge/*.json`, `language/rule_engine.py` | knowledge base + forward chaining + explanations |
| LLM | `language/llm.py`, `language/guard.py` | local LLM, few-shot prompts, LangChain, meaning guard |
| Hindi | `language/hindi.py`, `knowledge/hindi_phrasebook.json` | phrasebook first, then the LLM |
| Speech | `speech/tts.py`, `speech/stt.py` | pyttsx3 worker process, gTTS, speech recognition |
| UI | `app.py`, `gesturebridge/desktop.py` | Streamlit, OpenCV |

**PEAS** - Performance: sign accuracy, sentence correctness, time to speech ·
Environment: a signer in front of a webcam, changing light · Actuators: screen + speaker ·
Sensors: webcam. Agent type: model-based (keeps a smoothing window, hold timer and word buffer).

## Evaluation

| Command | Measures | Output |
|---|---|---|
| `python -m gesturebridge.train` | accuracy, macro-F1, CV, confusion, per-sign F1, leave-one-signer-out, accuracy by lighting/hand/signer, latency | `reports/training_summary.md`, `reports/metrics.json`, `reports/figures/` |
| `python -m gesturebridge.evaluate sentences --llm` | sentence correctness on 59 test cases, before and after the LLM | `reports/sentence_eval.md` |
| `python -m gesturebridge.evaluate decoder` | greedy vs beam search under simulated recognition noise | `reports/decoder_eval.md` |
| `python -m gesturebridge.evaluate latency [--camera]` | time per pipeline stage | `reports/latency.md` |
| `python -m gesturebridge.evaluate prompts [--backend langchain]` | prompt-engineering comparison | `reports/prompt_eval.md` |
| `python -m gesturebridge.evaluate live` | live test: tester ratings, latency | `reports/live_eval.md` |
| `python -m pytest` | 109 automated tests | - |

### Measured so far (language side, 3 November 2026, this laptop)

The sign-recognition numbers depend on your recordings - they appear in
`reports/training_summary.md` after `python -m gesturebridge.train`.

| Measurement | Result |
|---|---|
| Rule engine, 59 test sentences (ISL word order, negation, questions, requests ...) | **59 / 59 correct** |
| After LLM polishing + guard | **59 / 59 correct**; the guard rejected 2 answers that changed the meaning |
| Prompt styles (`llama3.2:3b`): answers that kept the meaning | zero-shot 74.6 % · few-shot 96.6 % · strict few-shot **100 %** |
| Beam search vs greedy at 80 % simulated sign accuracy | 61.7 % → **70.0 %** correct sentences (at 60 %: 38.8 % → 52.0 %) |
| Time from the SPEAK sign to the start of speech | ≈ **0.55 s** (rules 0.6 ms, LLM ≈ 0.54 s, speech start ≈ 0 ms) |

Zero-shot prompting shows why the guard matters: the LLM answered "Where is the
bathroom?" with "The bathroom is located behind the door." - an invented fact that the
guard rejected (`reports/prompt_eval.md`).

## Project structure

```
app.py                      Streamlit app
gesturebridge/
  agent.py                  the agent (sense-think-act), word buffer, controls
  hand_tracker.py           MediaPipe wrapper, drawing, camera
  features.py               106 hand-shape features
  dataset.py                CSV storage, augmentation, burst groups
  train.py                  model training + comparison + report figures
  recognizer.py             loads the trained model
  stabiliser.py             decisions over time under uncertainty
  pipeline.py               decoder → rules → LLM → Hindi
  language/                 knowledge base, rule engine, decoder, LLM, guard, Hindi
  speech/                   text-to-speech worker, speech-to-text
  collect.py desktop.py check.py evaluate.py plots.py history.py signs.py config.py
knowledge/                  signs, lexicon, grammar rules, example corpus, Hindi phrasebook
data/                       test sentences; raw/landmarks.csv after recording
models/                     trained models (+ MediaPipe hand model)
reports/                    metrics, figures and evaluation reports
docs/                       sign guide, report guide, responsible AI, demo & viva
tests/                      pytest suite (synthetic hand data for tests only)
```

## Troubleshooting

| Problem | Fix |
|---|---|
| Camera picture is black / "black frames" | Open the privacy shutter or press the camera key (F-key with a camera icon); close Zoom/Teams/Camera app; Windows Settings → Privacy → Camera → allow desktop apps. |
| "Could not open camera" | Another app uses it, or try camera number 1 in the sidebar (`--camera 1` for the scripts). |
| LLM shows "off" | Start Ollama (tray icon) and `ollama pull llama3.2:3b`. The rule-based sentence is used meanwhile. |
| First sentence is slow | The LLM loads into memory the first time (~10 s); the app warms it up at start. |
| No sound | Check the sidebar *Speech* toggle and Windows volume; Hindi needs internet. |
| Signs confused | See the confusion matrix; make key-poses more different, record more bursts, retrain ([docs/SIGN_GUIDE.md](docs/SIGN_GUIDE.md)). |
| Signs added twice / too slowly | Sidebar → Recognition: hold time, pause after a sign, acceptance confidence. |

## Team split (suggested)

* **Member 1 - vision & ML**: recording sessions, `features.py`, `train.py`, model comparison, fairness tables.
* **Member 2 - language & reasoning**: stabiliser, decoder, knowledge base and rules, LLM prompts and guard.
* **Member 3 - app & evaluation**: Streamlit app, speech, live test with classmates, report, responsible AI, GitHub.

Everyone should be able to explain the whole pipeline - see [docs/DEMO_AND_VIVA.md](docs/DEMO_AND_VIVA.md).

## Documentation

* [docs/SIGN_GUIDE.md](docs/SIGN_GUIDE.md) - key-poses, recording protocol, adding signs
* [docs/REPORT_GUIDE.md](docs/REPORT_GUIDE.md) - report outline, where every number comes from, references
* [docs/RESPONSIBLE_AI.md](docs/RESPONSIBLE_AI.md) - privacy, fairness, transparency, safety, limitations, consent form
* [docs/DEMO_AND_VIVA.md](docs/DEMO_AND_VIVA.md) - demo script, backups, likely questions

GestureBridge is a student prototype for an AI course. It is an assistive tool, not a
replacement for qualified sign-language interpreters.
