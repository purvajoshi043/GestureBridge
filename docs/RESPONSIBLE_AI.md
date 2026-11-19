# Responsible AI in GestureBridge

GestureBridge is an **assistive prototype**. It supports communication between a
signer and a non-signer; it does **not** replace a qualified sign-language interpreter,
especially in medical, legal or emergency situations.

## 1. Privacy by design

| Risk | What GestureBridge does |
|---|---|
| Video of people's faces and homes | No video or image is ever stored. Only 21 hand-landmark coordinates per frame are saved (`data/raw/landmarks.csv`). |
| Sentences sent to a cloud AI | The LLM runs **locally** with Ollama; no sentence leaves the laptop. |
| Other people on the network seeing the camera | The Streamlit app only listens on `127.0.0.1` (this computer). |
| Camera left on in the background | The camera starts only when you press Start. It is released automatically a few seconds after the page is closed or the browser tab goes into the background (nobody is watching the stream), and resumes by itself when you come back to the tab. Stopping the app always releases it. |
| Usage tracking | Streamlit usage statistics are switched off (`.streamlit/config.toml`). |
| Online features | Only two optional features use the internet, and the UI says so: Hindi speech (gTTS) and reverse mode (Google speech recognition). |
| Consent | Everyone who records data or tests the system should agree first (template below). Each signer can be deleted from the CSV by name. |

## 2. Fairness and bias - measure it, report it honestly

Hand-landmark models can work better for some people than others: hand size, skin
tone (affects the hand detector in poor light), lighting, background, left- vs
right-handedness, and signing style.

What the code does:
* **Normalisation**: landmarks are made independent of position and hand size;
  left hands are mirrored, so left-handed signers are supported.
* **Leave-one-signer-out evaluation** (`python -m gesturebridge.train`): the model is
  tested on each person after training only on the others.
* **Accuracy by group**: the training report lists accuracy per `condition` (light,
  background), per `handedness` and per `signer`.

What you should do for the report:
1. Record at least 3 people, in at least 2 lighting conditions (`--condition`).
2. Include the group tables from `reports/training_summary.md` and comment on the
   gaps. A gap is a finding, not a failure - report it.
3. In the live test, include people who did **not** record training data, and note
   their hand size / skin tone / handedness only if they agree.

## 3. Transparency and explainability

* The live screen shows the recognised sign, its confidence and the next best guesses.
* "Why this sentence?" shows every grammar rule that fired, in order, with a plain-English
  explanation (e.g. *"'I' + describing word 'hungry' needs the verb 'to be': inserted 'am'"*).
* Beam search shows when grammar knowledge changed a recognised sign, and why (scores).
* The LLM's own answer and the guard's decision are shown, including rejected answers.

## 4. Safety: preventing hallucination and misinformation

The LLM may only polish wording. `gesturebridge/language/guard.py` rejects any LLM
answer that:
* adds a meaning word (e.g. "I want some **cold** water"),
* drops a meaning word,
* adds or removes a negation (the most dangerous error: "I am **not** hungry" vs "I am hungry"),
* turns a statement into a question or the other way round,
* becomes much longer or shorter.

Then the rule-based sentence is spoken instead. Measured examples (see
`reports/sentence_eval.md`): "No water." → LLM "There is no water." (rejected: new
meaning), "Water, please." → "Can I have some water?" (rejected: changed type).
The rule engine also has a safety-net rule (R47) so a negation sign is never silently lost.

When the classifier is unsure, the app **asks the signer to repeat** instead of guessing.

## 5. Accessibility

* Large-text and high-contrast modes (sidebar → Accessibility).
* The "please repeat" prompt is **visual** by default - a Deaf signer may not hear a spoken prompt.
* Speech rate is adjustable; sentences can be repeated with one click.
* Reverse mode shows what the hearing person said as large text.

## 6. Known limitations (say them before you are asked)

* Small vocabulary (15 words + 3 controls), single-hand, static key-poses.
* No facial grammar (questions, negation by head shake) and no body-location signs.
* Real ISL is a full language with its own grammar; our rules cover common patterns only.
* Accuracy depends on lighting and on who recorded the training data.
* Hindi translations outside the hand-checked phrasebook come from a small local LLM and can be wrong; the UI labels them as machine translation.
* The system has not been evaluated with Deaf users. Future work must involve the Deaf
  community in design and testing (see Bragg et al., 2019).

## 7. Consent template (for recording and live tests)

> I agree to take part in testing the GestureBridge student project. I understand that
> only the positions of my hand joints (numbers) are recorded, not video or images,
> that the data is used only for this course project, and that I can ask for my data
> to be deleted at any time.
>
> Name: ____________  Signature: ____________  Date: ____________
