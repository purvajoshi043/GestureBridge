# Sign guide and recording protocol

The classifier learns **your** signs from **your** recordings. This guide says which
key-pose to show for each label and how to record data that gives a reliable model.

## 1. The vocabulary (18 signs + background class)

Most real ISL/ASL signs move. A single-frame classifier needs one clear, **held**
handshape per word, so every word has a *key-pose*: hold it still for about half a
second. The defaults were chosen to look clearly different to the camera; you can
replace any of them with the real ISL handshape - the code only cares about the label
(change the text in `knowledge/signs.json`).

| Label | Key-pose to hold | Based on |
|---|---|---|
| HELLO | Open hand, all five fingers spread, palm facing the camera | wave |
| I | Fist with only the little finger raised | ASL letter I |
| YOU | Index finger pointing at the camera, other fingers closed | pointing at the listener |
| WANT | Claw: all five fingers spread and bent, palm up | ASL WANT |
| WATER | Index, middle and ring fingers up and spread (W), thumb holds the little finger | ASL WATER |
| FOOD | Flat-O: all fingertips pinched together against the thumb | ASL EAT/FOOD |
| HUNGRY | C-shape: fingers and thumb curved as if holding a cup | ASL HUNGRY |
| HELP | Thumbs up | ASL HELP |
| NO | Thumbs down | prototype |
| YES | Closed fist, knuckles facing the camera | ASL YES |
| THANK_YOU | Flat hand, fingers together, palm up, fingertips towards the camera | ASL THANK YOU (end) |
| PLEASE | Flat hand on the chest, back of the hand to the camera, fingers sideways | ASL PLEASE |
| SORRY | Y-hand: thumb and little finger out | prototype |
| GO | Index finger pointing sideways (horizontal) | ASL GO |
| HOME | L-shape: thumb and index finger at a right angle | prototype |
| **SPEAK** (control) | OK sign: thumb and index finger in a circle, other fingers up | finishes the sentence and speaks it |
| **CLEAR** (control) | V sign: index and middle fingers up and spread | clears the words |
| **DELETE** (control) | Horns: index and little finger up | removes the last word |
| **NOTHING** (background) | Relaxed hand, hand moving between signs, scratching, resting | ignored |

**Record NOTHING.** Without it, every random hand position is forced into one of the
signs. With it, the model can answer "this is not a sign", which removes most false words.

There is also a **? / Question** button: press it to turn the sentence into a yes/no
question (sign languages mark questions with the eyebrows, which a hand-only system cannot see).

## 2. How to record

Easiest: open the website → **🎬 Record & train** tab → type your name, pick a sign, press
**⏺ Record** (2-second countdown, then hold the pose), repeat for every sign, then press
**🧠 Train models**. The same can be done with the OpenCV tool:

```
python -m gesturebridge.collect --signer nancy --condition bright
```

| Key | Action |
|---|---|
| N / → and P / ← | next / previous sign |
| SPACE | 2-second countdown, then records 100 frames (SPACE again stops early) |
| U | undo the last burst (deletes it from the CSV) |
| C | change the condition tag (bright, dim, backlit, outdoor, plain/busy background) |
| Q / ESC | quit |

Only the 21 landmark coordinates per frame are saved to `data/raw/landmarks.csv` - never images.

### Recording protocol (do this for a model that works on demo day)

1. **Every team member records every sign** (`--signer` with their own name). The
   leave-one-signer-out test in the training report then shows how well the model
   works for a person it has never seen.
2. **At least 3 bursts of 100 frames per sign per person**, ideally in different
   sessions. Training holds out whole bursts for testing, so one burst per sign is
   not enough for an honest test score.
3. **Vary things slightly during a burst**: move the hand closer/further, a little
   left/right, tilt it a few degrees. Keep the handshape correct.
4. **Vary the conditions across bursts**: bright and dim light, plain and busy
   background (tag them with `--condition` or the C key). The fairness section of the
   report compares accuracy per condition.
5. **Use both hands if someone is left-handed** - left hands are mirrored into a
   right-hand frame, so they help rather than confuse the model.
6. **Record NOTHING last**, with the same amount of data as a sign: relaxed hands,
   transitions between signs, hands resting near the face, etc.

Target: about **300+ samples per sign per person** → ~6,000 samples per person, about
20 minutes of recording each.

## 3. Check the data before the demo

```
python -m gesturebridge.train
```

Open the **Model dashboard** tab (or `reports/training_summary.md`):

* **Confusion matrix**: if two signs are confused more than ~10 % of the time, make
  their key-poses more different (or change one of them in `signs.json`), re-record
  them (U undoes bad bursts) and retrain.
* **Per-sign F1**: any sign below ~0.85 needs more or cleaner data.
* **Leave-one-signer-out** much lower than the normal test score → record more people
  or more variation.

## 4. Adding a new sign (no code changes)

1. Add it to `knowledge/signs.json` (label, type `word`, key-pose text).
2. Add its grammar facts to `knowledge/lexicon.json` - many words are already there
   (THIRSTY, TIRED, SICK, NEED, DOCTOR, BATHROOM, WHERE, ...), e.g.
   `"COLD": {"pos": "adjective", "text": "cold"}`.
3. Record it with the collection tool and retrain.

The grammar rules work on parts of speech, so `I COLD` immediately becomes "I am cold."
