"""Hosted-LLM fallback chain.

Chain = MUSE_PROVIDERS (comma list, priority order; default: every preset below), keeping
only providers whose <NAME>_API_KEY is set (ollama needs none). Per provider:
MUSE_<NAME>_MODEL, MUSE_<NAME>_BASE_URL and MUSE_<NAME>_TIMEOUT override the preset; unknown names are treated
as OpenAI-compatible and need MUSE_<NAME>_BASE_URL.

    python -m muse.providers            show the chain
    python -m muse.providers ask "hi"   route one prompt
    python -m muse.providers log        recent call log
"""
import os
import random
import sys
import time
import uuid
from dataclasses import dataclass
from typing import Protocol

import httpx

from muse.store import Store

PRESETS = {  # name: (kind, base_url, default model)
    "anthropic": ("anthropic", "https://api.anthropic.com/v1", "claude-haiku-4-5-20251001"),
    "openai": ("openai", "https://api.openai.com/v1", "gpt-4o-mini"),
    "gemini": ("openai", "https://generativelanguage.googleapis.com/v1beta/openai", "gemini-2.5-flash"),
    "groq": ("openai", "https://api.groq.com/openai/v1", "llama-3.3-70b-versatile"),
    "openrouter": ("openai", "https://openrouter.ai/api/v1", "meta-llama/llama-3.3-70b-instruct"),
    "ollama": ("openai", "http://127.0.0.1:11434/v1", "gemma4:e4b"),  # local, keyless; opt-in
}
DEFAULT_ORDER = ["anthropic", "openai", "gemini", "groq", "openrouter"]


@dataclass
class Completion:
    text: str
    prompt_tokens: int | None
    completion_tokens: int | None


class ProviderError(Exception):
    def __init__(self, outcome: str, retryable: bool, detail: str = "", retry_after: float | None = None):
        super().__init__(f"{outcome}: {detail}")
        self.outcome, self.retryable, self.detail, self.retry_after = outcome, retryable, detail[:300], retry_after


class Provider(Protocol):
    name: str

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> Completion: ...


def _post(client: httpx.Client, url: str, headers: dict, body: dict) -> dict:
    """POST and translate every failure mode into a ProviderError with a retry verdict."""
    try:
        r = client.post(url, headers=headers, json=body)
    except httpx.TimeoutException as e:
        raise ProviderError("timeout", True, str(e))
    except httpx.TransportError as e:
        raise ProviderError("connect_error", True, repr(e))
    if r.status_code == 429:
        ra = r.headers.get("retry-after")
        raise ProviderError("rate_limited", True, r.text, float(ra) if ra and ra.isdigit() else None)
    if r.status_code >= 500 or r.status_code == 529:
        raise ProviderError("server_error", True, f"{r.status_code} {r.text}")
    if r.status_code in (401, 403):
        raise ProviderError("auth_error", False, f"{r.status_code} {r.text}")
    if r.status_code >= 400:
        raise ProviderError("client_error", False, f"{r.status_code} {r.text}")
    try:
        return r.json()
    except ValueError:
        raise ProviderError("bad_response", False, f"200 but not JSON: {r.text}")


def _check(text, stop: str | None, filtered: set[str]) -> str:
    """A 200 is not success until it contains a usable answer."""
    if stop in filtered:
        raise ProviderError("filtered", False, f"stop reason {stop!r}")
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("empty", False, f"no text (stop reason {stop!r})")
    return text.strip()


class OpenAICompatProvider:
    def __init__(self, name: str, base_url: str, model: str, key: str | None, timeout: float):
        self.name, self.url, self.model = name, base_url.rstrip("/") + "/chat/completions", model
        self.headers = {"Authorization": f"Bearer {key}"} if key else {}
        self.client = httpx.Client(timeout=timeout)

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> Completion:
        data = _post(self.client, self.url, self.headers, {
            "model": self.model, "max_tokens": max_tokens,
            "messages": [{"role": "system", "content": system}, *messages]})
        try:
            choice, usage = data["choices"][0], data.get("usage") or {}
            text = _check(choice["message"].get("content"), choice.get("finish_reason"), {"content_filter"})
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise ProviderError("bad_response", False, f"unexpected shape: {e!r} {str(data)[:200]}")
        return Completion(text, usage.get("prompt_tokens"), usage.get("completion_tokens"))


class AnthropicProvider:
    def __init__(self, name: str, base_url: str, model: str, key: str | None, timeout: float):
        self.name, self.url, self.model = name, base_url.rstrip("/") + "/messages", model
        self.headers = {"x-api-key": key or "", "anthropic-version": "2023-06-01"}
        self.client = httpx.Client(timeout=timeout)

    def complete(self, system: str, messages: list[dict], max_tokens: int) -> Completion:
        data = _post(self.client, self.url, self.headers,
                     {"model": self.model, "max_tokens": max_tokens, "system": system, "messages": messages})
        try:
            text = "".join(b.get("text", "") for b in data["content"] if b.get("type") == "text")
            text, usage = _check(text, data.get("stop_reason"), {"refusal"}), data.get("usage") or {}
        except (KeyError, TypeError, AttributeError) as e:
            raise ProviderError("bad_response", False, f"unexpected shape: {e!r} {str(data)[:200]}")
        return Completion(text, usage.get("input_tokens"), usage.get("output_tokens"))


def build_chain() -> list[Provider]:
    names = [n.strip().lower() for n in os.getenv("MUSE_PROVIDERS", ",".join(DEFAULT_ORDER)).split(",") if n.strip()]
    timeout = float(os.getenv("MUSE_TIMEOUT", "30"))
    chain = []
    for name in names:
        kind, url, model = PRESETS.get(name, ("openai", None, None))
        env = lambda k: os.getenv(f"MUSE_{name.upper()}_{k}")  # noqa: E731
        key, url, model = os.getenv(f"{name.upper()}_API_KEY"), env("BASE_URL") or url, env("MODEL") or model
        if (key or name == "ollama") and url and model:
            cls = AnthropicProvider if kind == "anthropic" else OpenAICompatProvider
            chain.append(cls(name, url, model, key, float(env("TIMEOUT") or timeout)))
    return chain


class Router:
    def __init__(self, store: Store | None = None, chain: list[Provider] | None = None):
        self.store, self.chain = store or Store(), chain if chain is not None else build_chain()
        self.max_retries = int(os.getenv("MUSE_MAX_RETRIES", "2"))      # extra attempts per provider
        self.base, self.cap = float(os.getenv("MUSE_BACKOFF_BASE", "0.5")), float(os.getenv("MUSE_BACKOFF_CAP", "8"))

    def _backoff(self, attempt: int, retry_after: float | None) -> float:
        delay = min(self.cap, self.base * 2 ** attempt) * random.uniform(0.5, 1.5)  # exp + jitter
        return min(self.cap, max(delay, retry_after or 0))

    def complete(self, system: str, messages: list[dict], max_tokens: int = 1024) -> tuple[Completion, str, list[dict]]:
        """Walk the chain. Returns (completion, provider name, per-attempt log). Raises if all fail."""
        if not self.chain:
            raise RuntimeError("No providers configured: set e.g. ANTHROPIC_API_KEY, or MUSE_PROVIDERS=ollama")
        rid, log, served = uuid.uuid4().hex[:12], [], None
        for p in self.chain:
            for attempt in range(self.max_retries + 1):
                t0 = time.perf_counter()
                row = {"request_id": rid, "provider": p.name, "attempt": attempt + 1,
                       "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
                try:
                    c = p.complete(system, messages, max_tokens)
                    log.append({**row, "outcome": "ok", "latency_ms": int((time.perf_counter() - t0) * 1000),
                                "prompt_tokens": c.prompt_tokens, "completion_tokens": c.completion_tokens})
                    served = p.name
                    break
                except ProviderError as e:
                    log.append({**row, "outcome": e.outcome, "detail": e.detail,
                                "latency_ms": int((time.perf_counter() - t0) * 1000)})
                    if not e.retryable or attempt == self.max_retries:
                        break
                    time.sleep(self._backoff(attempt, e.retry_after))
            if served:
                break
        for r in log:
            r["served_by"] = served
        self.store.log_calls(log)
        if not served:
            raise RuntimeError(f"All providers failed (request {rid}): " +
                               "; ".join(f"{r['provider']}#{r['attempt']} {r['outcome']}" for r in log))
        return c, served, log


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "chain"
    if cmd == "log":
        for r in reversed(Store().recent_calls(int(sys.argv[2]) if len(sys.argv) > 2 else 20)):
            print(f"{r['created_at']} {r['request_id']} {r['provider']:>10}#{r['attempt']} {r['outcome']:<13} "
                  f"{r['latency_ms']:>6}ms tok={r['prompt_tokens']}/{r['completion_tokens']} "
                  f"served_by={r['served_by']} {(r['detail'] or '')[:60]}")
    elif cmd == "ask":
        c, who, _ = Router().complete("Be brief.", [{"role": "user", "content": " ".join(sys.argv[2:])}])
        print(f"[{who}] {c.text}")
    else:
        print("chain:", [f"{p.name}:{p.model}" for p in build_chain()] or "EMPTY (no keys set)")
