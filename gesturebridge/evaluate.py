"""Evaluation suite - produces the numbers and charts for the report.

    python -m gesturebridge.evaluate sentences [--llm] [--default-only]
    python -m gesturebridge.evaluate decoder [--trials 30]
    python -m gesturebridge.evaluate latency [--camera] [--seconds 10]
    python -m gesturebridge.evaluate prompts [--backend ollama|langchain]
    python -m gesturebridge.evaluate live
    python -m gesturebridge.evaluate all

Every command writes reports/<name>.json and reports/<name>.md.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from . import plots
from .config import METRICS_PATH, RAW_DATA_PATH, REPORTS_DIR, TEST_SENTENCES_PATH, LanguageSettings, Settings
from .language.guard import expand_contractions
from .language.llm import PROMPT_STYLES, SentenceRefiner
from .pipeline import LanguagePipeline
from .signs import SignVocabulary


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def normalise_sentence(sentence: str) -> list[str]:
    return re.sub(r"[^a-z' ]+", " ", expand_contractions(sentence or "")).split()


def same_sentence(a: str, b: str) -> bool:
    return normalise_sentence(a) == normalise_sentence(b) and a.strip().endswith("?") == b.strip().endswith("?")


def word_error_rate(hypothesis: str, references: list[str]) -> float:
    def wer(h: list[str], r: list[str]) -> float:
        d = list(range(len(r) + 1))
        for i, hw in enumerate(h, 1):
            prev, d[0] = d[0], i
            for j, rw in enumerate(r, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (hw != rw))
        return d[len(r)] / max(1, len(r))
    hyp = normalise_sentence(hypothesis)
    return min(wer(hyp, normalise_sentence(ref)) for ref in references)


def load_cases(default_only: bool = False) -> list[dict]:
    cases = json.loads(Path(TEST_SENTENCES_PATH).read_text(encoding="utf-8"))["cases"]
    return [c for c in cases if c.get("default") or not default_only]


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(values, q)) if values else float("nan")


def timing_row(name: str, values: list[float]) -> dict:
    return {"stage": name, "n": len(values), "mean_ms": round(statistics.fmean(values), 3) if values else None,
            "median_ms": round(statistics.median(values), 3) if values else None,
            "p95_ms": round(percentile(values, 95), 3) if values else None}


def write_report(name: str, data: dict, markdown: str, reports_dir: Path = REPORTS_DIR) -> None:
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / f"{name}.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    (reports_dir / f"{name}.md").write_text(markdown, encoding="utf-8")
    print(f"\nSaved reports/{name}.json and reports/{name}.md")


# ---------------------------------------------------------------------------
# 1. Sentence correctness (rule engine, optionally + LLM)
# ---------------------------------------------------------------------------

def evaluate_sentences(use_llm: bool = False, default_only: bool = False, settings: LanguageSettings | None = None) -> dict:
    settings = settings or Settings.load().language
    pipeline = LanguagePipeline(settings)
    if use_llm:
        available, why = pipeline.refiner.availability()
        print(f"LLM: {why}")
    rows = []
    for case in load_cases(default_only):
        glosses = case["glosses"].split()
        result = pipeline.translate_glosses(glosses, use_llm=use_llm, language="en")
        row = {
            "glosses": case["glosses"], "category": case["category"], "expected": case["expected"],
            "rule_sentence": result.rule_sentence,
            "rule_correct": any(same_sentence(result.rule_sentence, e) for e in case["expected"]),
            "rule_wer": round(word_error_rate(result.rule_sentence, case["expected"]), 3),
            "rules_fired": [s.rule_id for s in result.derivation.trace],
            "rules_ms": round(result.timings_ms["rules"], 3),
        }
        if use_llm:
            row.update({
                "final_sentence": result.final_sentence,
                "final_correct": any(same_sentence(result.final_sentence, e) for e in case["expected"]),
                "llm_status": result.refine.status, "llm_raw": result.refine.raw, "llm_reason": result.refine.reason,
                "llm_ms": round(result.refine.latency_ms, 1),
            })
        rows.append(row)
        mark = "OK " if row["rule_correct"] else "BAD"
        extra = f"  | LLM [{row['llm_status']}] {row['final_sentence']}" if use_llm else ""
        print(f"{mark} {case['glosses']:<26} -> {result.rule_sentence}{extra}")

    n = len(rows)
    data = {
        "cases": n, "default_only": default_only, "use_llm": use_llm,
        "rule_accuracy": sum(r["rule_correct"] for r in rows) / n,
        "rule_mean_wer": statistics.fmean(r["rule_wer"] for r in rows),
        "by_category": {},
        "rules_used": dict(Counter(rule for r in rows for rule in r["rules_fired"]).most_common()),
        "rows": rows,
    }
    categories = defaultdict(list)
    for r in rows:
        categories[r["category"]].append(r)
    for cat, items in sorted(categories.items()):
        data["by_category"][cat] = {"cases": len(items), "rule_correct": sum(i["rule_correct"] for i in items)}
    if use_llm:
        llm_rows = [r for r in rows if r["llm_status"] not in ("unavailable", "disabled")]
        data["final_accuracy"] = sum(r["final_correct"] for r in rows) / n
        data["llm_status_counts"] = dict(Counter(r["llm_status"] for r in rows))
        data["llm_mean_ms"] = statistics.fmean(r["llm_ms"] for r in llm_rows) if llm_rows else None

    md = [f"# Sentence correctness ({n} test cases{' - default vocabulary only' if default_only else ''})", "",
          f"* Rule engine: **{data['rule_accuracy']:.1%}** exact sentences, mean word error rate {data['rule_mean_wer']:.3f}"]
    if use_llm:
        md.append(f"* After LLM refinement (+ guard): **{data['final_accuracy']:.1%}**; LLM outcomes: {data['llm_status_counts']}")
        if data["llm_mean_ms"]:
            md.append(f"* Mean LLM time: {data['llm_mean_ms']:.0f} ms")
    md += ["", "| Signs | Expected | Rule engine | OK |" + (" LLM output | LLM status | OK |" if use_llm else ""),
           "|---|---|---|---|" + ("---|---|---|" if use_llm else "")]
    for r in rows:
        line = f"| {r['glosses']} | {r['expected'][0]} | {r['rule_sentence']} | {'yes' if r['rule_correct'] else 'NO'} |"
        if use_llm:
            line += f" {r['final_sentence']} | {r['llm_status']} | {'yes' if r['final_correct'] else 'NO'} |"
        md.append(line)
    md += ["", "## By category", "", "| Category | Correct / cases |", "|---|---|"]
    md += [f"| {c} | {v['rule_correct']} / {v['cases']} |" for c, v in data["by_category"].items()]
    print(f"\nRule engine accuracy: {data['rule_accuracy']:.1%} ({sum(r['rule_correct'] for r in rows)}/{n})")
    if use_llm:
        print(f"After LLM: {data['final_accuracy']:.1%}   outcomes: {data['llm_status_counts']}")
    write_report("sentence_eval", data, "\n".join(md) + "\n")
    return data


# ---------------------------------------------------------------------------
# 2. Decoder robustness: greedy vs beam search under simulated recognition noise
# ---------------------------------------------------------------------------

def _confusion_sources(words: list[str]) -> tuple[dict[str, list[tuple[str, float]]], str]:
    """Which wrong sign does each sign tend to be confused with? From the trained model's
    confusion matrix if available, otherwise uniform over the other words."""
    if METRICS_PATH.exists():
        doc = json.loads(METRICS_PATH.read_text(encoding="utf-8"))
        conf = doc["models"][doc["best_model"]]["confusion"]
        labels, matrix = conf["labels"], np.array(conf["matrix"], dtype=float)
        sources = {}
        for i, label in enumerate(labels):
            row = [(labels[j], matrix[i, j]) for j in range(len(labels)) if j != i and labels[j] in words]
            total = sum(c for _, c in row)
            if total > 0:
                sources[label] = [(l, c / total) for l, c in row if c > 0]
        if sources:
            return sources, "confusion matrix of the trained model"
    return {}, "uniform (no trained model yet)"


def evaluate_decoder(trials: int = 30, levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5), miss_rate: float = 0.25,
                     seed: int = 7) -> dict:
    vocab = SignVocabulary.load()
    words = vocab.words
    pipeline = LanguagePipeline(LanguageSettings(use_llm=False))
    engine, decoder = pipeline.engine, pipeline.decoder
    sources, source_name = _confusion_sources(words)
    cases = load_cases(default_only=True)
    rng = np.random.default_rng(seed)

    def pick_confuser(gloss: str) -> str:
        options = sources.get(gloss)
        if options:
            labels, weights = zip(*options)
            return str(rng.choice(labels, p=np.array(weights) / sum(weights)))
        return str(rng.choice([w for w in words if w != gloss]))

    def noisy_slot(gloss: str, accuracy: float) -> list[tuple[str, float]]:
        if gloss not in words:
            return [(gloss, 1.0)]
        confuser = pick_confuser(gloss)
        other = str(rng.choice([w for w in words if w not in (gloss, confuser)]))
        if rng.random() < accuracy:
            p_true = rng.uniform(0.55, 0.95)
            p_conf = (1 - p_true) * rng.uniform(0.5, 0.9)
            return [(gloss, p_true), (confuser, p_conf), (other, 1 - p_true - p_conf)]
        p_conf = rng.uniform(0.4, 0.65)
        if rng.random() < miss_rate:  # the right sign is not even in the top 3
            third = str(rng.choice([w for w in words if w not in (gloss, confuser, other)]))
            p_other = (1 - p_conf) * rng.uniform(0.5, 0.9)
            return [(confuser, p_conf), (other, p_other), (third, 1 - p_conf - p_other)]
        p_true = (1 - p_conf) * rng.uniform(0.55, 0.9)
        return [(confuser, p_conf), (gloss, p_true), (other, 1 - p_conf - p_true)]

    results = {"Greedy (top-1)": [], "Beam search": []}
    for level in levels:
        correct = Counter()
        total = 0
        for _ in range(trials):
            for case in cases:
                slots = [noisy_slot(g, level) for g in case["glosses"].split()]
                decoded = decoder.decode(slots)
                for name, glosses in (("Greedy (top-1)", decoded.greedy), ("Beam search", decoded.best)):
                    sentence = engine.run(glosses).sentence
                    correct[name] += any(same_sentence(sentence, e) for e in case["expected"])
                total += 1
        for name in results:
            results[name].append(correct[name] / total)
        print(f"sign accuracy {level:.0%}: greedy {results['Greedy (top-1)'][-1]:.1%}  "
              f"beam {results['Beam search'][-1]:.1%}  ({total} sentences)")

    figure = plots.decoder_robustness(list(levels), results, REPORTS_DIR / "figures" / "decoder_robustness.png")
    data = {"levels": list(levels), "results": results, "trials": trials, "cases": len(cases),
            "miss_rate": miss_rate, "confusions_from": source_name, "beam_width": decoder.beam_width,
            "lm_weight": decoder.lm_weight, "figure": figure}
    md = ["# Greedy vs beam-search decoding under simulated recognition noise", "",
          f"Each of the {len(cases)} default-vocabulary test sentences was signed {trials} times per noise level. "
          f"A wrong sign is drawn from the {source_name}; in {miss_rate:.0%} of the errors the right sign is "
          "not even among the top-3 candidates. Sentences are compared after the rule engine.", "",
          "| Simulated sign accuracy | Greedy (top-1) | Beam search |", "|---|---|---|"]
    md += [f"| {lvl:.0%} | {g:.1%} | {b:.1%} |" for lvl, g, b in
           zip(levels, results["Greedy (top-1)"], results["Beam search"])]
    md += ["", "![decoder robustness](figures/decoder_robustness.png)"]
    write_report("decoder_eval", data, "\n".join(md) + "\n")
    return data


# ---------------------------------------------------------------------------
# 3. Latency of every pipeline stage
# ---------------------------------------------------------------------------

def evaluate_latency(use_camera: bool = False, seconds: float = 10.0, llm_sentences: int = 10) -> dict:
    from .dataset import load_dataset, points_array
    from .features import extract_features, palm_centre_and_size
    from .recognizer import ModelNotFoundError, SignClassifier
    from .speech.tts import Speaker
    from .stabiliser import PredictionStabiliser

    settings = Settings.load()
    stages: list[dict] = []
    notes: list[str] = []

    classifier = None
    try:
        classifier = SignClassifier.load(settings.resolved_model_path)
    except ModelNotFoundError as exc:
        notes.append(str(exc))

    if classifier and RAW_DATA_PATH.exists():
        df = load_dataset(RAW_DATA_PATH).sort_values("timestamp").head(400)
        points = points_array(df)
        hands = df["handedness"].to_numpy()
        stabiliser = PredictionStabiliser(classifier.classes, settings.recognition)
        t_feat, t_clf, t_stab = [], [], []
        for i, (p, h) in enumerate(zip(points, hands)):
            t0 = time.perf_counter()
            feats = extract_features(p, h)
            t1 = time.perf_counter()
            probs = classifier.predict_proba(feats)
            t2 = time.perf_counter()
            centre, size = palm_centre_and_size(p)
            stabiliser.update(probs, i / 30, centre, size)
            t3 = time.perf_counter()
            t_feat.append((t1 - t0) * 1000)
            t_clf.append((t2 - t1) * 1000)
            t_stab.append((t3 - t2) * 1000)
        stages += [timing_row("feature extraction", t_feat),
                   timing_row(f"classifier ({classifier.model_name})", t_clf),
                   timing_row("stabiliser", t_stab)]
    elif classifier:
        notes.append("No recorded dataset - vision stages were not timed offline.")

    if use_camera:
        from .agent import GestureBridgeAgent
        from .hand_tracker import open_camera
        agent = GestureBridgeAgent(settings, classifier)
        cap = open_camera(settings.camera.index, settings.camera.width, settings.camera.height)
        t_track, t_step, frames = [], [], 0
        end = time.time() + seconds
        print(f"Measuring the live camera loop for {seconds:.0f} s - show your hand to the camera ...")
        while time.time() < end:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            mirrored, hand = agent.perceive(frame)
            t1 = time.perf_counter()
            agent.process(hand, time.monotonic())
            t2 = time.perf_counter()
            t_track.append((t1 - t0) * 1000)
            t_step.append((t2 - t0) * 1000)
            frames += 1
        cap.release()
        agent.shutdown()
        stages += [timing_row("hand tracking (MediaPipe)", t_track), timing_row("full frame (sense + think)", t_step)]
        if t_step:
            notes.append(f"Live camera: {frames} frames in {seconds:.0f} s = {frames / seconds:.1f} FPS (camera-limited).")

    pipeline = LanguagePipeline(settings.language)
    cases = load_cases()
    t_dec, t_rules = [], []
    for case in cases:
        result = pipeline.prepare([[(g, 1.0)] for g in case["glosses"].split()])
        t_dec.append(result.timings_ms["decode"])
        t_rules.append(result.timings_ms["rules"])
    stages += [timing_row("beam-search decoder", t_dec), timing_row("rule engine (forward chaining)", t_rules)]

    available, why = pipeline.refiner.availability()
    t_llm = []
    if available:
        pipeline.refiner.client.warm_up()
        for case in cases[:llm_sentences]:
            draft = pipeline.engine.run(case["glosses"].split()).sentence
            pipeline.refiner._cache.clear()
            r = pipeline.refiner.refine(case["glosses"].split(), draft)
            if r.status not in ("unavailable", "error"):
                t_llm.append(r.latency_ms)
        stages.append(timing_row(f"LLM refinement ({settings.language.llm_model})", t_llm))
    else:
        notes.append(f"LLM not timed: {why}")

    speaker = Speaker(volume=0.0)
    speaker.warm_up()
    speaker.wait_for("warmup", "done", timeout=20)
    t_tts = []
    for text in ("I am hungry.", "Please help me.", "Thank you.", "I want some water.", "I am going home."):
        r = speaker.speak(text)
        started = speaker.wait_for(r.request_id, "started", timeout=10) if r.request_id else None
        if started:
            t_tts.append((started - r.requested_at) * 1000)
        speaker.wait_for(r.request_id, "done", timeout=15)
    speaker.close()
    stages.append(timing_row("text-to-speech start (warm worker)", t_tts))

    hold = settings.recognition.hold_seconds * 1000
    language_ms = sum(s["mean_ms"] or 0 for s in stages
                      if s["stage"].startswith(("beam", "rule", "LLM", "text-to-speech")))
    data = {"stages": stages, "notes": notes, "hold_ms": hold,
            "after_speak_sign_ms": round(language_ms, 1),
            "estimated_end_to_end_ms": round(hold + language_ms, 1)}
    md = ["# Latency per pipeline stage", "", "| Stage | n | Mean ms | Median ms | p95 ms |", "|---|---|---|---|---|"]
    md += [f"| {s['stage']} | {s['n']} | {s['mean_ms']} | {s['median_ms']} | {s['p95_ms']} |" for s in stages]
    md += ["", f"* From the moment the SPEAK sign is accepted to the start of speech: about **{language_ms:.0f} ms** "
               "(decoder + rules + LLM + speech start).",
           f"* Including the {hold:.0f} ms the SPEAK sign must be held: about **{hold + language_ms:.0f} ms**."]
    md += [f"* Note: {n}" for n in notes]
    for s in stages:
        print(f"{s['stage']:<38} mean {s['mean_ms']} ms  p95 {s['p95_ms']} ms  (n={s['n']})")
    for n in notes:
        print("note:", n)
    write_report("latency", data, "\n".join(md) + "\n")
    return data


# ---------------------------------------------------------------------------
# 4. Prompt engineering comparison (LLM evaluation)
# ---------------------------------------------------------------------------

def evaluate_prompts(backend: str = "ollama", styles=PROMPT_STYLES) -> dict:
    settings = Settings.load().language
    pipeline = LanguagePipeline(settings)
    cases = load_cases()
    summary = {}
    rows = []
    for style in styles:
        refiner = SentenceRefiner(settings.llm_model, settings.ollama_url, settings.llm_timeout, backend, style)
        available, why = refiner.availability()
        if not available:
            print(f"LLM unavailable: {why}")
            return {"error": why}
        accepted = raw_correct = final_correct = 0
        latencies = []
        for case in cases:
            glosses = case["glosses"].split()
            draft = pipeline.engine.run(glosses).sentence
            r = refiner.refine(glosses, draft)
            raw = r.raw or ""
            ok_raw = any(same_sentence(raw, e) for e in case["expected"]) if raw else False
            ok_final = any(same_sentence(r.text, e) for e in case["expected"])
            accepted += r.status in ("ok", "unchanged")
            raw_correct += ok_raw
            final_correct += ok_final
            latencies.append(r.latency_ms)
            rows.append({"style": style, "glosses": case["glosses"], "draft": draft, "llm_raw": raw,
                         "status": r.status, "reason": r.reason, "raw_correct": ok_raw, "final_correct": ok_final})
        n = len(cases)
        summary[style] = {"guard_pass_rate": accepted / n, "raw_llm_correct": raw_correct / n,
                          "final_correct": final_correct / n, "mean_latency_ms": statistics.fmean(latencies)}
        print(f"{style:<16} guard pass {accepted / n:.1%}  raw LLM correct {raw_correct / n:.1%}  "
              f"final correct {final_correct / n:.1%}  {statistics.fmean(latencies):.0f} ms")
    data = {"backend": backend, "model": settings.llm_model, "cases": len(cases), "summary": summary, "rows": rows}
    md = [f"# Prompt engineering comparison ({settings.llm_model} via {backend}, {len(cases)} sentences)", "",
          "* Guard pass = the LLM answer kept the meaning (no added/removed content or negation).",
          "* Raw LLM correct = the LLM's own answer matches a reference sentence.",
          "* Final correct = after the guard (rejected answers fall back to the rule-based sentence).", "",
          "| Prompt | Guard pass | Raw LLM correct | Final correct | Mean ms |", "|---|---|---|---|---|"]
    md += [f"| {s} | {v['guard_pass_rate']:.1%} | {v['raw_llm_correct']:.1%} | {v['final_correct']:.1%} | "
           f"{v['mean_latency_ms']:.0f} |" for s, v in summary.items()]
    rejected = [r for r in rows if r["status"] == "rejected"]
    if rejected:
        md += ["", "## Examples the guard rejected", "", "| Prompt | Draft | LLM said | Reason |", "|---|---|---|---|"]
        md += [f"| {r['style']} | {r['draft']} | {r['llm_raw']} | {r['reason']} |" for r in rejected[:15]]
    write_report("prompt_eval" if backend == "ollama" else f"prompt_eval_{backend}", data, "\n".join(md) + "\n")
    return data


# ---------------------------------------------------------------------------
# 5. Live test summary (feedback given in the app)
# ---------------------------------------------------------------------------

def evaluate_live() -> dict:
    from .history import HistoryLog
    df = HistoryLog().load()
    if df.empty:
        print("No translations recorded yet - use the app first (Live translator tab).")
        return {}
    rated = df[df["feedback"].isin(["correct", "wrong"])]
    data = {"translations": int(len(df)), "rated": int(len(rated)),
            "accuracy": float((rated["feedback"] == "correct").mean()) if len(rated) else None,
            "per_tester": {}, "llm_status": df["llm_status"].value_counts().to_dict(),
            "beam_changed": int((df["beam_changed"].str.lower() == "true").sum())}
    for tester, group in rated.groupby(rated["tester"].replace("", "unnamed")):
        data["per_tester"][tester] = {"rated": int(len(group)), "accuracy": float((group["feedback"] == "correct").mean())}
    for column in ("latency_text_ms", "latency_speech_ms"):
        values = [float(v) for v in df[column] if v not in ("", "0", "0.0")]
        data[column] = timing_row(column, values)
    md = ["# Live test results", "", f"* Translations: {data['translations']}, rated by testers: {data['rated']}"]
    if data["accuracy"] is not None:
        md.append(f"* Sentence accuracy judged by testers: **{data['accuracy']:.1%}**")
    md += [f"* Beam search changed the signed words {data['beam_changed']} times", f"* LLM outcomes: {data['llm_status']}",
           "", "| Tester | Rated | Accuracy |", "|---|---|---|"]
    md += [f"| {t} | {v['rated']} | {v['accuracy']:.1%} |" for t, v in data["per_tester"].items()]
    for column in ("latency_text_ms", "latency_speech_ms"):
        row = data[column]
        if row["n"]:
            md.append(f"\n* {column}: mean {row['mean_ms']} ms, median {row['median_ms']} ms, p95 {row['p95_ms']} ms")
    print("\n".join(md))
    write_report("live_eval", data, "\n".join(md) + "\n")
    return data


def main(argv: list[str] | None = None) -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description="GestureBridge evaluation suite")
    sub = parser.add_subparsers(dest="command", required=True)
    s = sub.add_parser("sentences", help="sentence correctness on data/test_sentences.json")
    s.add_argument("--llm", action="store_true", help="also evaluate the LLM refinement")
    s.add_argument("--default-only", action="store_true", help="only sentences of the default 15 signs")
    d = sub.add_parser("decoder", help="greedy vs beam search under simulated noise")
    d.add_argument("--trials", type=int, default=30)
    lat = sub.add_parser("latency", help="time every pipeline stage")
    lat.add_argument("--camera", action="store_true", help="also measure the live camera loop")
    lat.add_argument("--seconds", type=float, default=10.0)
    p = sub.add_parser("prompts", help="compare LLM prompt styles")
    p.add_argument("--backend", choices=["ollama", "langchain"], default="ollama")
    sub.add_parser("live", help="summarise live-test feedback from the app")
    sub.add_parser("all", help="sentences (+LLM), decoder, latency and prompts")
    args = parser.parse_args(argv)

    if args.command == "sentences":
        evaluate_sentences(args.llm, args.default_only)
    elif args.command == "decoder":
        evaluate_decoder(args.trials)
    elif args.command == "latency":
        evaluate_latency(args.camera, args.seconds)
    elif args.command == "prompts":
        evaluate_prompts(args.backend)
    elif args.command == "live":
        evaluate_live()
    else:
        evaluate_sentences(use_llm=True)
        evaluate_decoder()
        evaluate_latency()
        evaluate_prompts()


if __name__ == "__main__":
    main()
