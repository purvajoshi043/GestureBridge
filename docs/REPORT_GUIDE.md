# Report guide

A section-by-section plan for the project report, with the file or command that
provides each figure and number. Numbers that depend on your recordings come from
`reports/` after you run:

```
python -m gesturebridge.train          # classifier comparison, confusion matrices, LOSO, fairness
python -m gesturebridge.evaluate all   # sentences, beam search, latency, prompt comparison
python -m gesturebridge.evaluate live  # after the live test with classmates
```

---

## 0. Title and abstract
*GestureBridge: An Explainable, Privacy-Preserving Sign-Language-to-Speech Agent.*
Update your existing abstract with the final numbers (best model + macro-F1,
sentence correctness, median time to speech).

## 1. Introduction and problem definition
* Communication gap between signers and non-signers; interpreters are scarce.
* Existing tools need gloves/sensors or only show static information.
* Goal: camera-only, real-time, runs on a normal laptop, explainable, private.
* Scope: 15 words + 3 control signs, one hand, static key-poses (prototype).

## 2. Literature review (5-8 sources)
Suggested structure - one paragraph each, then a table "what they did / what we take / gap":

| Theme | Sources | Point for our project |
|---|---|---|
| Hand tracking | Zhang et al. 2020; Lugaresi et al. 2019 | MediaPipe gives 21 3-D landmarks in real time on a CPU - we use it as the sensor. |
| Vision-based SLR | Pigou et al. 2015; Rastgoo et al. 2021 (survey) | CNNs on raw video need large datasets and GPUs; landmarks are lighter and less affected by background/lighting. |
| Continuous recognition & translation | Koller et al. 2015; Camgöz et al. 2018, 2020 | Sign-to-text is translation, not word-by-word lookup: sign order and grammar differ from English - motivates our grammar stage. |
| Datasets | Li et al. 2020 (WLASL); Sridhar et al. 2020 (INCLUDE, Indian SL) | Large datasets exist but are video-based; we record our own landmark dataset. |
| LLMs | Vaswani et al. 2017; Brown et al. 2020; Touvron et al. 2023 | Transformer LLMs can polish text with few-shot prompts and run locally. |
| Responsible AI | Bragg et al. 2019; Bender et al. 2021; Ji et al. 2023 | Involve Deaf users; LLMs hallucinate - so we guard the LLM's output. |

**Gap you fill:** a light, camera-only pipeline that combines learned perception with
symbolic reasoning (knowledge base + inference + search), keeps the LLM under a
meaning-preservation guard and explains every step.

## 3. System design
* **PEAS table** and agent type (model-based agent) - see the Help tab or `gesturebridge/agent.py` docstring.
* **Architecture diagram** - Help tab (screenshot) or the pipeline line in `README.md`.
* **Module table**: module → file → technique → course unit (the "Where is the AI?" table in the Help tab).

## 4. Methodology
1. **Data collection** - protocol from `docs/SIGN_GUIDE.md`; dataset table (samples per sign, per signer, per condition) from `reports/training_summary.md`. Only landmarks are stored (privacy).
2. **Feature engineering** - `gesturebridge/features.py`: pixel-space correction, wrist-origin + palm-length normalisation, mirroring left hands, 106 features (coordinates, 15 joint angles, 16 distances, 5 extension ratios, palm normal + pointing direction, 4 spread angles). Explain why rotation is kept (HELP vs NO).
3. **Classifiers** - `gesturebridge/train.py`: Random Forest (100 trees), SVM with RBF kernel + Platt scaling (probabilities), MLP (128-64, early stopping); data augmentation (rotation, hand turn, landmark noise) on training data only.
4. **Evaluation protocol** - burst-level hold-out (explain data leakage: neighbouring frames are near-duplicates), grouped cross-validation, leave-one-signer-out, accuracy per group.
5. **Reasoning under uncertainty** - `gesturebridge/stabiliser.py`: averaging probabilities over a window (combining evidence), confidence threshold, hold time, stillness gate, entropy, top-3 candidates, "please repeat" when unsure, cooldown and release.
6. **Heuristic search** - `gesturebridge/language/decoder.py`: beam search over the top-3 signs per slot, score = Σ log P(classifier) + λ·Σ log P(pos | previous pos); the bigram prior is learned from `knowledge/gloss_corpus.txt`; λ = 0.2 tuned on the corpus, not on the test set. It can be seen as a constraint-optimisation problem (one variable per word slot, domains = top-3 signs, soft grammatical constraints).
7. **Knowledge representation and inference** - `knowledge/lexicon.json` (facts), `knowledge/grammar_rules.json` (47 production rules), `gesturebridge/language/rule_engine.py` (forward chaining: match → conflict resolution by salience → act; refraction; fixed point; explanation trace). Show one worked trace, e.g. `WATER I WANT` → R10 topic_to_svo → R31 partitive_some → "I want some water."
8. **LLM refinement** - local `llama3.2:3b` through Ollama (REST) or LangChain; few-shot prompt (`gesturebridge/language/llm.py`); meaning-preservation guard (`guard.py`); fallback to the rule-based sentence.
9. **Speech and UI** - pyttsx3 worker process (offline), gTTS for Hindi, Streamlit UI, OpenCV desktop backup.

## 5. Results (fill in from `reports/`)
| Result | Source |
|---|---|
| Classifier comparison table + chart | `reports/training_summary.md`, `figures/model_comparison.png`, `figures/prediction_latency.png` |
| Per-sign F1 + confusion matrix + most confused pairs | `figures/per_class_f1.png`, `figures/confusion_<model>.png`, training summary |
| Generalisation to new signers | `figures/leave_one_signer_out.png` |
| Fairness (accuracy by lighting / hand / signer) | training summary "Accuracy by ..." tables |
| Sentence correctness (rules, rules + LLM) | `reports/sentence_eval.md` |
| Beam search vs greedy under noise | `reports/decoder_eval.md`, `figures/decoder_robustness.png` (re-run after training so it uses your confusion matrix) |
| Prompt engineering comparison | `reports/prompt_eval.md` (+ `prompt_eval_langchain.md`) |
| Latency per stage, time to speech | `reports/latency.md` (`--camera` adds the live loop) |
| Live test with 5-10 classmates | `reports/live_eval.md` |

Discuss: which model wins and why (accuracy vs speed), which signs are confused and why
(similar handshapes), the gap between normal test and leave-one-signer-out, how often the
guard rejected the LLM and what it caught.

## 6. Responsible AI
Use `docs/RESPONSIBLE_AI.md`: privacy by design, fairness measurements, transparency
(rule trace, confidences), safety guard against hallucination, accessibility, limitations.

## 7. Limitations and future work
Dynamic signs (sequence model such as an LSTM [Hochreiter & Schmidhuber 1997] or a
Transformer over landmark sequences), two-handed signs, facial grammar, a larger ISL
vocabulary, testing with Deaf users, an exact Viterbi decoder.

## 8. Conclusion

## References
Check every entry in Google Scholar before submitting and use the citation style your instructor wants.

1. Bender, E. M., Gebru, T., McMillan-Major, A., & Shmitchell, S. (2021). On the dangers of stochastic parrots: Can language models be too big? *FAccT '21*.
2. Bragg, D., Koller, O., Bellard, M., Berke, L., Boudreault, P., Braffort, A., Caselli, N., Huenerfauth, M., Kacorri, H., Verhoef, T., Vogler, C., & Morris, M. R. (2019). Sign language recognition, generation, and translation: An interdisciplinary perspective. *ASSETS '19*.
3. Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5-32.
4. Brown, T. B., et al. (2020). Language models are few-shot learners. *NeurIPS 2020*.
5. Camgöz, N. C., Hadfield, S., Koller, O., Ney, H., & Bowden, R. (2018). Neural sign language translation. *CVPR 2018*.
6. Camgöz, N. C., Koller, O., Hadfield, S., & Bowden, R. (2020). Sign language transformers: Joint end-to-end sign language recognition and translation. *CVPR 2020*.
7. Cortes, C., & Vapnik, V. (1995). Support-vector networks. *Machine Learning, 20*(3), 273-297.
8. Hochreiter, S., & Schmidhuber, J. (1997). Long short-term memory. *Neural Computation, 9*(8), 1735-1780.
9. Ji, Z., et al. (2023). Survey of hallucination in natural language generation. *ACM Computing Surveys, 55*(12).
10. Koller, O., Forster, J., & Ney, H. (2015). Continuous sign language recognition: Towards large vocabulary statistical recognition systems handling multiple signers. *Computer Vision and Image Understanding, 141*, 108-125.
11. Li, D., Rodriguez, C., Yu, X., & Li, H. (2020). Word-level deep sign language recognition from video: A new large-scale dataset and methods comparison. *WACV 2020*.
12. Lugaresi, C., et al. (2019). MediaPipe: A framework for building perception pipelines. *arXiv:1906.08172*.
13. Pedregosa, F., et al. (2011). Scikit-learn: Machine learning in Python. *JMLR, 12*, 2825-2830.
14. Pigou, L., Dieleman, S., Kindermans, P.-J., & Schrauwen, B. (2015). Sign language recognition using convolutional neural networks. *ECCV 2014 Workshops*, LNCS 8925.
15. Platt, J. (1999). Probabilistic outputs for support vector machines and comparisons to regularized likelihood methods. *Advances in Large Margin Classifiers*.
16. Rastgoo, R., Kiani, K., & Escalera, S. (2021). Sign language recognition: A deep survey. *Expert Systems with Applications, 164*, 113794.
17. Russell, S., & Norvig, P. (2021). *Artificial Intelligence: A Modern Approach* (4th ed.). Pearson. (PEAS, forward chaining, beam search, uncertainty)
18. Sridhar, A., Ganesan, R. G., Kumar, P., & Khapra, M. (2020). INCLUDE: A large scale dataset for Indian Sign Language recognition. *ACM Multimedia 2020*.
19. Touvron, H., et al. (2023). LLaMA: Open and efficient foundation language models. *arXiv:2302.13971*.
20. Vaswani, A., et al. (2017). Attention is all you need. *NeurIPS 2017*.
21. Zhang, F., Bazarevsky, V., Vakunov, A., Tkachenka, A., Sung, G., Chang, C.-L., & Grundmann, M. (2020). MediaPipe Hands: On-device real-time hand tracking. *arXiv:2006.10214*.
