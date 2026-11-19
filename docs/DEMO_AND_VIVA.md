# Demo plan and viva preparation

## Before the demo (10 minutes before)

1. Open the laptop camera's **privacy shutter** / turn the camera key on.
2. Make sure Ollama is running (its icon in the system tray), then run the check:
   ```
   .venv\Scripts\activate
   python -m gesturebridge.check
   ```
   Everything essential should say OK. The camera line must not say "black frames".
3. Start the app: double-click `run_app.bat` (opens http://127.0.0.1:8502).
4. Press **▶ Start camera**, sign `I` → `HUNGRY` → OK sign once to warm everything up.
5. Good light on your hands, plain background if possible, sit about an arm's length away.

## 5-minute demo script

| Time | Show | Say |
|---|---|---|
| 0:00 | Live tab, hand skeleton | "MediaPipe gives 21 landmarks; everything after that is ours." |
| 0:30 | Sign I → HUNGRY, hold each until the bar fills | "The stabiliser averages the last frames and waits for a still hand - that's how we deal with uncertainty." |
| 1:00 | OK sign → speech "I'm hungry." | "Then the beam search, the rule engine and the LLM build the sentence, and it is spoken." |
| 1:30 | Open "Why this sentence?" | "Forward chaining: rule R26 inserted 'am' because a pronoun is followed by a describing word." |
| 2:00 | Sign WATER → I → WANT (ISL order) → OK | "ISL is subject-object-verb; rule R10 reorders it to English." |
| 2:30 | Sentence lab: `I WANT:0.46/HUNGRY:0.44/FOOD:0.10` | "When the classifier is unsure, beam search uses grammar knowledge: 'I WANT' + end is unlikely, so it picks HUNGRY." |
| 3:00 | Sentence lab: `WATER NO` with LLM on → show "LLM said: There is no water → not used" | "The guard stops the LLM from changing the meaning." |
| 3:30 | Model dashboard | "We compared RF, SVM and MLP; burst-level hold-out and leave-one-signer-out avoid inflated accuracy." |
| 4:00 | Invite the examiner to sign | Have them sign HELLO or HELP (thumbs up). |
| 4:30 | GitHub repo, reports folder | "All numbers come from scripts in the repo and can be reproduced." |

**Backups:** camera fails → Sentence lab (whole language pipeline) or "Backup: add a word
without signing". Browser problem → `python -m gesturebridge.desktop`. Ollama off → the app
still works with the rule-based sentence (say: "this is our designed fallback").

## Likely viva questions

**What did you build, and what is a library?**
MediaPipe only returns 21 landmark coordinates. We built: the data-collection tool and
dataset, the 106 features, the training/evaluation pipeline and model comparison, the
stabiliser, the beam-search decoder and its grammar prior, the knowledge base and the
forward-chaining engine with 47 rules, the prompts and the meaning guard, the agent, the
UI and the evaluation suite.

**Where is the AI?**
Perception (pretrained CNN in MediaPipe), learning (our classifier), reasoning under
uncertainty (stabiliser), search (beam search with a heuristic grammar prior), knowledge
representation and inference (forward chaining), generative AI (local Transformer LLM).
Text-to-speech is a tool, not AI reasoning.

**Why landmarks instead of a CNN on the video?**
21×3 numbers instead of 640×480×3 pixels: far less data needed, runs in real time on a
CPU, mostly unaffected by background and clothing, and it protects privacy (no images stored).

**Why is your accuracy trustworthy?**
Whole recording bursts are held out (neighbouring frames are near-duplicates - a random split
would leak), cross-validation is grouped by burst, and leave-one-signer-out tests people the
model has never seen. Augmentation is applied to training data only.

**Why does SVM/MLP/RF win?** (answer with your numbers)
Compare macro-F1 and ms/frame in `reports/training_summary.md`. RF predicts tree by tree in
Python, so it is slower per frame; SVM needs scaling and Platt scaling for probabilities; MLP
is fast and accurate with enough data.

**Why macro-F1 and not only accuracy?** Every sign counts equally, even rare ones; accuracy
can hide a sign that is always wrong.

**What does the stabiliser do? Isn't a threshold enough?**
A single frame is noisy. We average probabilities over ~8 frames (combining evidence), require
the same sign for 0.6 s while the hand is still, use cooldown/release so a held sign is
not added twice, and ask to repeat when the confidence is in the unsure band.

**Explain the beam search.**
Each word slot has the top-3 signs with probabilities. A sequence's score is the sum of
log classifier probabilities plus λ times the log probability of each part of speech given the
previous one (bigram learned from example sign sentences). Beam search keeps the best 5 partial
sequences per step instead of all 3ⁿ. λ = 0.2 was tuned on the corpus: it never changes a
correctly recognised sentence but fixes near-ties. Results: `reports/decoder_eval.md`.

**Is it A\*? Is it optimal?** No - beam search is a bounded best-first search; it can miss the
optimum if the best path falls out of the beam. With a bigram model the exact optimum could be
found with Viterbi (dynamic programming); with 3 candidates and a beam of 5 our beam is almost
always optimal.

**How does the rule engine work?**
Working memory = tokens with facts from the lexicon + sentence facts (negated, question...).
Each cycle: match all rules → pick the activation with the highest salience (then rule order,
then leftmost) → fire it → record the explanation. Refraction stops a rule firing twice on the
same tokens. It stops at a fixed point. This is a production system / expert-system design:
knowledge base + inference engine + explanation facility.

**Forward or backward chaining - why forward?** We start from the data (the signed words) and
derive the sentence; there is no single goal to prove, so data-driven forward chaining fits.

**What if the LLM changes the meaning?**
The prompt forbids it, temperature is low, and `guard.py` checks the answer: same content words
(stems, small synonym list), same negation, same sentence type, similar length. If any check
fails, the rule-based sentence is used. We measured how often this happens (`reports/sentence_eval.md`).

**Why a local LLM?** Privacy (nothing leaves the laptop), no internet needed, no cost, and it is
part of the syllabus (local LLM deployment). Prompt-style comparison: `reports/prompt_eval.md`.

**What is the agent type? PEAS?** Model-based agent: internal state (window, hold timer, word
buffer). P = sign accuracy, sentence correctness, latency; E = signer, webcam, lighting; A =
screen + speaker; S = camera.

**How do you handle left-handed signers?** Left-hand landmarks are mirrored into a right-hand frame.

**What about bias?** We report accuracy per signer, lighting condition and hand. See `docs/RESPONSIBLE_AI.md`.

**Limitations?** Small vocabulary, static single-hand key-poses, no facial grammar, not tested
with Deaf users, not a replacement for interpreters.

**How would you add 50 more signs?** Add them to `signs.json` and `lexicon.json`, record, retrain -
no code changes. For moving signs: a sequence model over landmark windows.
