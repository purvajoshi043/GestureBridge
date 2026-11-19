# Latency per pipeline stage

| Stage | n | Mean ms | Median ms | p95 ms |
|---|---|---|---|---|
| beam-search decoder | 59 | 0.037 | 0.034 | 0.06 |
| rule engine (forward chaining) | 59 | 0.593 | 0.577 | 1.111 |
| LLM refinement (llama3.2:3b) | 10 | 542.707 | 540.686 | 660.212 |
| text-to-speech start (warm worker) | 5 | 0.0 | 0.0 | 0.0 |

* From the moment the SPEAK sign is accepted to the start of speech: about **543 ms** (decoder + rules + LLM + speech start).
* Including the 600 ms the SPEAK sign must be held: about **1143 ms**.
* Note: No trained model at D:\AI_PRJ\models\sign_classifier.joblib. Collect data (python -m gesturebridge.collect) and train (python -m gesturebridge.train) first.
