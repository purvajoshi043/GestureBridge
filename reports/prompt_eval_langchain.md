# Prompt engineering comparison (llama3.2:3b via langchain, 59 sentences)

* Guard pass = the LLM answer kept the meaning (no added/removed content or negation).
* Raw LLM correct = the LLM's own answer matches a reference sentence.
* Final correct = after the guard (rejected answers fall back to the rule-based sentence).

| Prompt | Guard pass | Raw LLM correct | Final correct | Mean ms |
|---|---|---|---|---|
| zero_shot | 78.0% | 76.3% | 98.3% | 762 |
| few_shot | 94.9% | 94.9% | 100.0% | 648 |
| few_shot_strict | 100.0% | 100.0% | 100.0% | 665 |

## Examples the guard rejected

| Prompt | Draft | LLM said | Reason |
|---|---|---|---|
| zero_shot | I want some food. | I'm hungry and I want some food. | the LLM added new meaning: 'hungry' |
| zero_shot | No water. | There is no water. | the LLM added new meaning: 'there' |
| zero_shot | Hello! | The correct rewritten sentence is: Hello! | the LLM made the sentence much longer (6 words) |
| zero_shot | Hello! Please help me. | A distressed person is seeking assistance by writing "HELLO HELP" in a draft. | the LLM made the sentence much longer (13 words) |
| zero_shot | Thank you. | The polite expression "Thank you" is often used to express gratitude or appreciation for something. | the LLM made the sentence much longer (15 words) |
| zero_shot | Thank you. | The polite response is simply "Thank you." | the LLM made the sentence much longer (7 words) |
| zero_shot | Yes. | The draft is approved. | the LLM dropped meaning words: 'yes' |
| zero_shot | No. | There is no indication of a draft. | the LLM made the sentence much longer (7 words) |
| zero_shot | No, thank you. | The polite way to decline an offer is to say "No, thank you." | the LLM made the sentence much longer (13 words) |
| zero_shot | Water, please. | The sign should read "Water, please." | the LLM added new meaning: 'read', 'sign' |
| zero_shot | Water, please. | The polite request for water can be expressed as "Water, please." | the LLM made the sentence much longer (11 words) |
| zero_shot | Where do you go? | You go where you want to go. | the LLM changed the sentence type (question vs statement) |
| zero_shot | I do not need help. | I don't need assistance. | the LLM dropped meaning words: 'help' |
| few_shot | No water. | There is no water. | the LLM added new meaning: 'there' |
| few_shot | Hello! Please help me. | Hello, help! | the LLM dropped meaning words: 'me' |
