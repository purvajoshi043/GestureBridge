# Prompt engineering comparison (llama3.2:3b via ollama, 59 sentences)

* Guard pass = the LLM answer kept the meaning (no added/removed content or negation).
* Raw LLM correct = the LLM's own answer matches a reference sentence.
* Final correct = after the guard (rejected answers fall back to the rule-based sentence).

| Prompt | Guard pass | Raw LLM correct | Final correct | Mean ms |
|---|---|---|---|---|
| zero_shot | 74.6% | 71.2% | 96.6% | 706 |
| few_shot | 96.6% | 96.6% | 100.0% | 648 |
| few_shot_strict | 100.0% | 100.0% | 100.0% | 666 |

## Examples the guard rejected

| Prompt | Draft | LLM said | Reason |
|---|---|---|---|
| zero_shot | I want some food. | I'm hungry and I want some food. | the LLM added new meaning: 'hungry' |
| zero_shot | No water. | There is no water. | the LLM added new meaning: 'there' |
| zero_shot | Hello! | The most basic and common greeting in English is simply "Hello". | the LLM made the sentence much longer (11 words) |
| zero_shot | Hello! Please help me. | A distressed person is seeking assistance by writing "HELLO HELP" in a draft. | the LLM made the sentence much longer (13 words) |
| zero_shot | Thank you. | The polite way to express gratitude is simply "Thank you." | the LLM made the sentence much longer (10 words) |
| zero_shot | Thank you. | The correct sentence is: Thank you. | the LLM added new meaning: 'correct', 'sentence' |
| zero_shot | Yes. | The draft is approved. | the LLM dropped meaning words: 'yes' |
| zero_shot | No. | There are no signs to indicate a draft. | the LLM made the sentence much longer (8 words) |
| zero_shot | No, thank you. | The polite response to a "No, thank you" is simply "No, thank you". | the LLM made the sentence much longer (13 words) |
| zero_shot | Yes, I am hungry. | I'm hungry. | the LLM dropped meaning words: 'yes' |
| zero_shot | Water, please. | The sign should read "Water, please." | the LLM added new meaning: 'read', 'sign' |
| zero_shot | Water, please. | Can I get you some water? | the LLM changed the sentence type (question vs statement) |
| zero_shot | Where is the bathroom? | The bathroom is located behind the door. | the LLM changed the sentence type (question vs statement) |
| zero_shot | Where do you go? | You go where you want to go. | the LLM changed the sentence type (question vs statement) |
| zero_shot | I do not need help. | I don't need any assistance. | the LLM dropped meaning words: 'help' |
