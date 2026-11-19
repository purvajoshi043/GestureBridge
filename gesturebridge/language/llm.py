"""LLM sentence refinement with a local model (Ollama), guarded and with fallback.

Privacy: the model runs locally through Ollama, so no sentence leaves the laptop.
Safety:  every LLM answer goes through ``guard.check_refinement``; if it changes
         the meaning, is slow, or Ollama is not running, the rule-based draft is
         used instead, so the system always produces a sentence.
Backends: plain REST calls to Ollama, or the same prompt built as a LangChain
          chain (ChatPromptTemplate + FewShotChatMessagePromptTemplate).
"""

from __future__ import annotations

import functools
import threading
import time
from dataclasses import dataclass

import requests

from .guard import check_refinement, clean_output

SYSTEM_PROMPT = (
    "You polish English sentences produced by a sign-language translator. "
    "The draft was built from the signer's signs by grammar rules and is usually already correct.\n"
    "Rules:\n"
    "1. Keep exactly the same meaning. Never add information: no new objects, places, people, "
    "feelings, times or reasons.\n"
    "2. Keep negation exactly: if the draft says not/no, keep it; if it does not, never add it.\n"
    "3. Keep the speaker (I/you/he/she) and the sentence type (statement, question or request).\n"
    "4. Only fix grammar and make it sound natural and short. Contractions are fine.\n"
    "5. Reply with the sentence only: no quotes, no explanation."
)

ZERO_SHOT_PROMPT = "Rewrite the draft as one natural English sentence. Reply with the sentence only."

FEW_SHOT_EXAMPLES = [
    ("I HUNGRY", "I am hungry.", "I'm hungry."),
    ("WATER I WANT", "I want some water.", "I want some water."),
    ("HELP", "Please help me.", "Please help me."),
    ("I GO HOME NO", "I am not going home.", "I'm not going home."),
    ("YOU HUNGRY QUESTION", "Are you hungry?", "Are you hungry?"),
    ("PLEASE WATER", "Water, please.", "Water, please."),
]

PROMPT_STYLES = ("zero_shot", "few_shot", "few_shot_strict")


def user_message(glosses: list[str], draft: str, strict: bool = False) -> str:
    message = f"Signs: {' '.join(glosses)}\nDraft: {draft}"
    if strict:
        message += "\nIf the draft is already natural, return it unchanged."
    return message


def build_messages(glosses: list[str], draft: str, style: str = "few_shot") -> list[dict]:
    if style == "zero_shot":
        return [{"role": "system", "content": ZERO_SHOT_PROMPT},
                {"role": "user", "content": user_message(glosses, draft)}]
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for signs, example_draft, answer in FEW_SHOT_EXAMPLES:
        messages.append({"role": "user", "content": user_message(signs.split(), example_draft)})
        messages.append({"role": "assistant", "content": answer})
    messages.append({"role": "user", "content": user_message(glosses, draft, strict=style == "few_shot_strict")})
    return messages


def normalise_url(url: str) -> str:
    # On Windows "localhost" tries IPv6 first and waits ~2 s before falling
    # back to IPv4, where Ollama listens; 127.0.0.1 avoids that delay.
    return url.strip().rstrip("/").replace("://localhost", "://127.0.0.1")


class OllamaClient:
    """Minimal client for the local Ollama REST API."""

    def __init__(self, base_url: str = "http://127.0.0.1:11434", model: str = "llama3.2:3b",
                 timeout: float = 15.0):
        self.base_url = normalise_url(base_url)
        self.model = model
        self.timeout = timeout
        self._status: tuple[float, bool, list[str]] = (0.0, False, [])

    def status(self, max_age: float = 10.0) -> tuple[bool, list[str]]:
        """(server reachable, installed model names), cached for `max_age` seconds."""
        checked, up, models = self._status
        if time.monotonic() - checked < max_age:
            return up, models
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=1.5)
            response.raise_for_status()
            models = [m.get("name", "") for m in response.json().get("models", [])]
            up = True
        except (requests.RequestException, ValueError):
            up, models = False, []
        self._status = (time.monotonic(), up, models)
        return up, models

    def is_up(self) -> bool:
        return self.status()[0]

    def has_model(self) -> bool:
        up, models = self.status()
        wanted = self.model if ":" in self.model else f"{self.model}:latest"
        return up and any(m in (self.model, wanted) for m in models)

    def chat(self, messages: list[dict], temperature: float = 0.1, max_tokens: int = 60,
             timeout: float | None = None) -> str:
        payload = {
            "model": self.model,
            "messages": messages,
            "stream": False,
            "keep_alive": "30m",
            "options": {"temperature": temperature, "num_predict": max_tokens},
        }
        response = requests.post(f"{self.base_url}/api/chat", json=payload, timeout=timeout or self.timeout)
        response.raise_for_status()
        return response.json()["message"]["content"]

    def warm_up(self) -> None:
        """Load the model into memory in the background (the first call can take ~15 s)."""
        def _run():
            try:
                if self.has_model():
                    self.chat([{"role": "user", "content": "Reply with: OK"}], max_tokens=3, timeout=120)
            except requests.RequestException:
                pass
        threading.Thread(target=_run, daemon=True).start()


@functools.lru_cache(maxsize=8)
def _langchain_chain(model: str, base_url: str, timeout: float, style: str):
    """Same prompt as `build_messages`, expressed as a LangChain LCEL chain."""
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate, FewShotChatMessagePromptTemplate
    from langchain_ollama import ChatOllama

    llm = ChatOllama(model=model, base_url=base_url, num_predict=60, keep_alive="30m",
                     temperature=0.0 if style == "few_shot_strict" else 0.1,
                     client_kwargs={"timeout": timeout})
    if style == "zero_shot":
        prompt = ChatPromptTemplate.from_messages([
            ("system", ZERO_SHOT_PROMPT), ("human", "Signs: {signs}\nDraft: {draft}")])
    else:
        example_prompt = ChatPromptTemplate.from_messages([
            ("human", "Signs: {signs}\nDraft: {draft}"), ("ai", "{answer}")])
        examples = FewShotChatMessagePromptTemplate(
            example_prompt=example_prompt,
            examples=[{"signs": s, "draft": d, "answer": a} for s, d, a in FEW_SHOT_EXAMPLES],
        )
        suffix = "\nIf the draft is already natural, return it unchanged." if style == "few_shot_strict" else ""
        prompt = ChatPromptTemplate.from_messages([
            ("system", SYSTEM_PROMPT), examples, ("human", "Signs: {signs}\nDraft: {draft}" + suffix)])
    return prompt | llm | StrOutputParser()


def langchain_refine(glosses: list[str], draft: str, model: str, base_url: str,
                     timeout: float, style: str = "few_shot") -> str:
    chain = _langchain_chain(model, base_url, float(timeout), style)
    return chain.invoke({"signs": " ".join(glosses), "draft": draft})


@dataclass
class RefineResult:
    text: str              # sentence to use (LLM output if accepted, else the draft)
    raw: str | None        # what the LLM actually said
    used_llm: bool
    status: str            # ok | unchanged | rejected | unavailable | timeout | error | disabled | cached
    reason: str
    latency_ms: float = 0.0


class SentenceRefiner:
    def __init__(self, model: str = "llama3.2:3b", base_url: str = "http://127.0.0.1:11434",
                 timeout: float = 15.0, backend: str = "ollama", prompt_style: str = "few_shot"):
        self.client = OllamaClient(base_url, model, timeout)
        self.backend = backend
        self.prompt_style = prompt_style
        self._cache: dict[tuple, RefineResult] = {}

    def configure(self, model: str, base_url: str, timeout: float, backend: str, prompt_style: str) -> None:
        if (model, normalise_url(base_url)) != (self.client.model, self.client.base_url):
            self.client = OllamaClient(base_url, model, timeout)
            self._cache.clear()
        self.client.timeout = timeout
        if (backend, prompt_style) != (self.backend, self.prompt_style):
            self._cache.clear()
        self.backend = backend
        self.prompt_style = prompt_style

    def availability(self) -> tuple[bool, str]:
        up, models = self.client.status()
        if not up:
            return False, f"Ollama is not running at {self.client.base_url} (start it, or the rule-based sentence is used)"
        if not self.client.has_model():
            return False, f"model '{self.client.model}' is not installed: run `ollama pull {self.client.model}`"
        return True, f"{self.client.model} via {self.backend}"

    def _ask(self, glosses: list[str], draft: str) -> str:
        if self.backend == "langchain":
            return langchain_refine(glosses, draft, self.client.model, self.client.base_url,
                                    self.client.timeout, self.prompt_style)
        return self.client.chat(build_messages(glosses, draft, self.prompt_style),
                                temperature=0.0 if self.prompt_style == "few_shot_strict" else 0.1)

    def refine(self, glosses: list[str], draft: str) -> RefineResult:
        if not draft:
            return RefineResult(draft, None, False, "disabled", "nothing to refine")
        key = (tuple(glosses), draft)
        if key in self._cache:
            cached = self._cache[key]
            return RefineResult(cached.text, cached.raw, cached.used_llm, "cached", cached.reason, 0.0)

        available, why = self.availability()
        if not available:
            return RefineResult(draft, None, False, "unavailable", why)

        started = time.perf_counter()
        try:
            raw = self._ask(glosses, draft)
        except requests.Timeout:
            return RefineResult(draft, None, False, "timeout",
                                f"no answer within {self.client.timeout:.0f} s", _ms(started))
        except Exception as exc:  # network errors, LangChain/httpx errors, bad JSON
            if "timeout" in type(exc).__name__.lower() or "timed out" in str(exc).lower():
                return RefineResult(draft, None, False, "timeout",
                                    f"no answer within {self.client.timeout:.0f} s", _ms(started))
            return RefineResult(draft, None, False, "error", f"{type(exc).__name__}: {exc}", _ms(started))

        latency = _ms(started)
        guard = check_refinement(draft, raw)
        if not guard.ok:
            result = RefineResult(draft, clean_output(raw), False, "rejected", guard.reason, latency)
        elif _same(guard.cleaned, draft):
            result = RefineResult(draft, guard.cleaned, True, "unchanged", "draft was already natural", latency)
        else:
            result = RefineResult(guard.cleaned, guard.cleaned, True, "ok", guard.reason, latency)
        self._cache[key] = result
        return result


def _same(a: str, b: str) -> bool:
    simplify = lambda s: "".join(ch for ch in s.lower() if ch.isalnum())
    return simplify(a) == simplify(b)


def _ms(started: float) -> float:
    return (time.perf_counter() - started) * 1000
