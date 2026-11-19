# Sentence correctness (59 test cases)

* Rule engine: **100.0%** exact sentences, mean word error rate 0.000
* After LLM refinement (+ guard): **100.0%**; LLM outcomes: {'ok': 34, 'unchanged': 23, 'rejected': 2}
* Mean LLM time: 754 ms

| Signs | Expected | Rule engine | OK | LLM output | LLM status | OK |
|---|---|---|---|---|---|---|
| I HUNGRY | I am hungry. | I am hungry. | yes | I'm hungry. | ok | yes |
| YOU HUNGRY | You are hungry. | You are hungry. | yes | You're hungry. | ok | yes |
| HUNGRY | I am hungry. | I am hungry. | yes | I'm hungry. | ok | yes |
| I SORRY | I am sorry. | I am sorry. | yes | I'm sorry. | ok | yes |
| SORRY | I am sorry. | I am sorry. | yes | I'm sorry. | ok | yes |
| I WANT WATER | I want some water. | I want some water. | yes | I want water. | ok | yes |
| WATER I WANT | I want some water. | I want some water. | yes | I want water. | ok | yes |
| I WATER WANT | I want some water. | I want some water. | yes | I want water. | ok | yes |
| WANT WATER | I want some water. | I want some water. | yes | I want water. | ok | yes |
| FOOD WANT | I want some food. | I want some food. | yes | I want food. | ok | yes |
| I WANT FOOD WATER | I want some food and water. | I want some food and water. | yes | I want food and water. | ok | yes |
| I HUNGRY NO | I am not hungry. | I am not hungry. | yes | I'm not hungry. | ok | yes |
| HUNGRY NO | I am not hungry. | I am not hungry. | yes | I'm not hungry. | ok | yes |
| I WANT FOOD NO | I do not want food. | I do not want food. | yes | I don't want food. | ok | yes |
| WATER NO | No water. | No water. | yes | No water. | unchanged | yes |
| HELP | Please help me. | Please help me. | yes | Help me. | ok | yes |
| HELP PLEASE | Please help me. | Please help me. | yes | Help me, please. | ok | yes |
| PLEASE HELP | Please help me. | Please help me. | yes | Help me. | ok | yes |
| HELP I | Please help me. | Please help me. | yes | Help me. | ok | yes |
| I HELP | I need help. | I need help. | yes | I need help. | unchanged | yes |
| YOU HELP I | Can you help me? | Can you help me? | yes | Can you help me? | unchanged | yes |
| YOU HELP I PLEASE | Can you help me, please? | Can you help me, please? | yes | Can you help me, please? | unchanged | yes |
| I GO HOME | I am going home. | I am going home. | yes | I'm going home. | ok | yes |
| HOME GO | I am going home. | I am going home. | yes | I'm going home. | ok | yes |
| I HOME GO | I am going home. | I am going home. | yes | I'm going home. | ok | yes |
| I GO HOME NO | I am not going home. | I am not going home. | yes | I'm not going home. | ok | yes |
| I WANT GO HOME | I want to go home. | I want to go home. | yes | I want to go home. | unchanged | yes |
| HELLO | Hello! | Hello! | yes | Hello! | unchanged | yes |
| HELLO I HUNGRY | Hello! I am hungry. | Hello! I am hungry. | yes | Hello, I'm hungry. | ok | yes |
| HELLO HELP | Hello! Please help me. | Hello! Please help me. | yes | Hello! Please help me. | rejected | yes |
| THANK_YOU | Thank you. | Thank you. | yes | Thank you. | unchanged | yes |
| THANK_YOU YOU | Thank you. | Thank you. | yes | Thank you. | unchanged | yes |
| YES | Yes. | Yes. | yes | Yes. | unchanged | yes |
| NO | No. | No. | yes | No. | unchanged | yes |
| YES PLEASE | Yes, please. | Yes, please. | yes | Yes, please. | unchanged | yes |
| NO THANK_YOU | No, thank you. | No, thank you. | yes | No, thank you. | unchanged | yes |
| YES I HUNGRY | Yes, I am hungry. | Yes, I am hungry. | yes | Yes, I'm hungry. | ok | yes |
| I WANT WATER PLEASE | I want some water, please. | I want some water, please. | yes | I want water, please. | ok | yes |
| PLEASE WATER | Water, please. | Water, please. | yes | Water, please. | rejected | yes |
| WATER PLEASE | Water, please. | Water, please. | yes | Water, please. | unchanged | yes |
| I I HUNGRY | I am hungry. | I am hungry. | yes | I'm hungry. | ok | yes |
| YOU HUNGRY QUESTION | Are you hungry? | Are you hungry? | yes | Are you hungry? | unchanged | yes |
| HUNGRY QUESTION | Are you hungry? | Are you hungry? | yes | Are you hungry? | unchanged | yes |
| YOU WANT WATER QUESTION | Do you want some water? | Do you want some water? | yes | Do you want some water? | unchanged | yes |
| WATER YOU WANT QUESTION | Do you want some water? | Do you want some water? | yes | Do you want some water? | unchanged | yes |
| I NO HUNGRY | I am not hungry. | I am not hungry. | yes | I'm not hungry. | ok | yes |
| I THIRSTY | I am thirsty. | I am thirsty. | yes | I'm thirsty. | ok | yes |
| I HUNGRY THIRSTY | I am hungry and thirsty. | I am hungry and thirsty. | yes | I'm hungry and thirsty. | ok | yes |
| HE WANT FOOD | He wants some food. | He wants some food. | yes | He wants food. | ok | yes |
| SHE SICK | She is sick. | She is sick. | yes | She is sick. | unchanged | yes |
| WE GO SCHOOL | We are going to school. | We are going to school. | yes | We're going to school. | ok | yes |
| I GO BATHROOM | I am going to the bathroom. | I am going to the bathroom. | yes | I'm going to the bathroom. | ok | yes |
| BATHROOM WHERE | Where is the bathroom? | Where is the bathroom? | yes | Where is the bathroom? | unchanged | yes |
| YOU GO WHERE | Where do you go? | Where do you go? | yes | Where do you go? | unchanged | yes |
| WHAT YOUR NAME | What is your name? | What is your name? | yes | What's your name? | ok | yes |
| I NEED DOCTOR | I need a doctor. | I need a doctor. | yes | I need a doctor. | unchanged | yes |
| DOCTOR COME | The doctor is coming. | The doctor is coming. | yes | The doctor is coming. | unchanged | yes |
| I LIKE YOU | I like you. | I like you. | yes | I like you. | unchanged | yes |
| I HELP NO | I do not need help. | I do not need help. | yes | I don't need help. | ok | yes |

## By category

| Category | Correct / cases |
|---|---|
| agreement | 1 / 1 |
| article | 2 / 2 |
| clean-up | 1 / 1 |
| conjunction | 2 / 2 |
| copula | 5 / 5 |
| formula | 2 / 2 |
| greeting | 3 / 3 |
| implicit subject | 3 / 3 |
| infinitive | 1 / 1 |
| motion | 3 / 3 |
| negation | 7 / 7 |
| object pronoun | 1 / 1 |
| partitive | 1 / 1 |
| politeness | 3 / 3 |
| reply | 5 / 5 |
| request | 7 / 7 |
| wh-question | 3 / 3 |
| word order | 2 / 2 |
| word order (SOV) | 2 / 2 |
| word order (topic) | 1 / 1 |
| yes/no question | 4 / 4 |
