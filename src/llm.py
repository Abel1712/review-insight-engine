"""One door to the LLMs: complete_json() picks a pinned model per role, validates output with pydantic,
and caches every answer on disk. Groq and Gemini are both reached through their OpenAI-compatible endpoints."""
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Literal

import openai
from pydantic import BaseModel, ValidationError
from tenacity import retry, retry_if_exception_type, stop_after_attempt

from config import CACHE, CFG, SEED

LLM = CFG["llm"]
LLM_CACHE = CACHE / "llm"
LLM_CACHE.mkdir(parents=True, exist_ok=True)
USAGE_LOG = LLM_CACHE / "usage.jsonl"
SWITCH_LOG = LLM_CACHE / "switches.jsonl"
SYSTEM = "You are a precise data-extraction assistant. Always reply with one valid JSON object and nothing else."
TRANSIENT = (openai.RateLimitError, openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError)
GONE_MARKERS = ("decommissioned", "model_not_found", "does not exist", "not found")

_clients: dict[str, openai.OpenAI] = {}
_catalog: dict[str, set[str]] = {}   # provider -> model ids available right now
_dead: set[tuple[str, str]] = set()  # (provider, model) found unavailable during this run
_limits: dict[str, dict] = {}        # model -> last-seen per-minute token bucket, for throttling


class DailyLimitReached(RuntimeError):
    """The provider wants us to wait hours, not seconds. Stop; the cache keeps all finished work."""


class NoModelAvailable(RuntimeError):
    """Every ranked model for this role is unavailable."""


class RequestTooLarge(RuntimeError):
    """This single request exceeds a per-minute limit (e.g. Groq OTPM); waiting can't help. Caller must shrink it."""


class OutputTruncated(RequestTooLarge):
    """The answer hit max_completion_tokens and was cut off. Never repaired or cached; caller must shrink the request."""


# ---------- providers and model selection ----------

def _client(provider: str) -> openai.OpenAI:
    if provider not in _clients:
        p = LLM["providers"][provider]
        key = os.getenv(p["key_env"])
        if not key:
            raise RuntimeError(f"{p['key_env']} is empty. Add it to review-insight-engine/.env")
        _clients[provider] = openai.OpenAI(api_key=key, base_url=p["base_url"], max_retries=0, timeout=120)
    return _clients[provider]


def _available(provider: str) -> set[str]:
    """List the provider's models at runtime (free catalogs change; never trust memory)."""
    if provider not in _catalog:
        try:
            _catalog[provider] = {m.id.removeprefix("models/") for m in _client(provider).models.list().data}
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            print(f"[llm] could not list {provider} models ({type(exc).__name__}); treating provider as down")
            _catalog[provider] = set()
    return _catalog[provider]


def _chain(role: str) -> list[dict]:
    """Ranked candidates. Main falls back to the secondary list last; secondary never borrows main-family models."""
    chain = list(LLM["roles"][role])
    return chain + LLM["roles"]["secondary"] if role == "main" else chain


def _mark_dead(role: str, cand: dict, reason: str) -> None:
    _dead.add((cand["provider"], cand["model"]))
    print(f"[llm] SWITCH role={role}: {cand['provider']}/{cand['model']} unavailable ({reason[:150]}) -> next model")
    with open(SWITCH_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), "role": role, "provider": cand["provider"],
                            "model": cand["model"], "reason": reason[:500]}) + "\n")


# ---------- helpers ----------

def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _seconds(duration: str | None) -> float:
    """Parse durations like '7.66s', '2m59.5s', '1h2m', '250ms'."""
    units = {"ms": 0.001, "h": 3600, "m": 60, "s": 1}
    return sum(float(n) * units[u] for n, u in re.findall(r"([\d.]+)(ms|h|m|s)", duration or ""))


def _retry_after(exc) -> float | None:
    """How long the provider asked us to wait: header first, then the error text (Groq / Gemini styles)."""
    response = getattr(exc, "response", None)
    if response is not None and response.headers.get("retry-after"):
        return float(response.headers["retry-after"])
    msg = str(exc)
    if m := re.search(r"try again in ([\d.hms]+)", msg):
        return _seconds(m.group(1))
    if m := re.search(r"retryDelay\W+([\d.]+)s", msg):
        return float(m.group(1))
    return None


def _wait(retry_state) -> float:
    """Honor the provider's requested wait, else exponential backoff (2, 4, 8, ... max 60s)."""
    exc = retry_state.outcome.exception()
    wait = _retry_after(exc)
    wait = wait + 0.5 if wait else min(60.0, 2.0 ** retry_state.attempt_number)
    print(f"[llm] {type(exc).__name__}, retry {retry_state.attempt_number} in {wait:.1f}s (same model)")
    return wait


def _cache_path(cand: dict, prompt: str):
    key = json.dumps({"provider": cand["provider"], "model": cand["model"], "prompt": prompt,
                      "temperature": LLM["temperature"], "params": cand.get("params", {})}, sort_keys=True)
    return LLM_CACHE / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def _throttle(model: str, prompt: str) -> None:
    """Sleep until the per-minute token bucket resets if the next call probably won't fit."""
    state = _limits.get(model)
    estimate = len(prompt) // 3 + 1000  # rough: ~3-4 chars per token, plus room for the answer
    if state and estimate > state["remaining_tokens"] and time.time() < state["reset_at"]:
        wait = state["reset_at"] - time.time() + 0.5
        print(f"[llm] throttle: ~{estimate} tokens needed, {state['remaining_tokens']} left this minute -> {wait:.1f}s")
        time.sleep(wait)


def tokens_used_last_24h(model: str) -> int:
    """Headers only report the per-minute bucket, so the daily budget is tracked from our own log."""
    if not USAGE_LOG.exists():
        return 0
    cutoff = time.time() - 24 * 3600
    return sum(r["usage"]["total_tokens"] for r in map(json.loads, USAGE_LOG.read_text(encoding="utf-8").splitlines())
               if r["model"] == model and datetime.fromisoformat(r["ts"]).timestamp() >= cutoff)


def _parse_json(text: str):
    """Strip stray reasoning tags / code fences and parse JSON."""
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text).strip()
    return json.loads(text)


# ---------- the API call ----------

@retry(retry=retry_if_exception_type(TRANSIENT), wait=_wait, stop=stop_after_attempt(LLM["max_retries"]), reraise=True)
def _call(cand: dict, prompt: str) -> tuple[str, dict]:
    """One request in JSON mode. Logs usage. Returns (text, usage)."""
    provider, model = cand["provider"], cand["model"]
    kwargs = {"temperature": LLM["temperature"], "max_completion_tokens": LLM["max_completion_tokens"]}
    if LLM["providers"][provider].get("supports_seed"):
        kwargs["seed"] = SEED
    kwargs.update(cand.get("params", {}))  # per-model overrides, e.g. a lower max_completion_tokens for an OTPM cap
    try:
        raw = _client(provider).chat.completions.with_raw_response.create(
            model=model,
            messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            **kwargs,
        )
    except openai.RateLimitError as exc:
        if "request too large" in str(exc).lower():
            raise RequestTooLarge(f"{provider}/{model}: {str(exc)[:200]}") from exc
        wait = _retry_after(exc) or 0
        if wait > LLM["max_short_wait_seconds"] or "perday" in str(exc).lower().replace(" ", ""):
            raise DailyLimitReached(f"{provider}/{model} daily limit (retry in ~{wait / 60:.0f} min). "
                                    "Finished calls are cached; rerun later to resume.") from exc
        raise

    completion = raw.parse()
    h = raw.headers
    if h.get("x-ratelimit-remaining-tokens") is not None:  # Groq sends these; Gemini doesn't
        _limits[model] = {"remaining_tokens": int(float(h["x-ratelimit-remaining-tokens"])),
                          "reset_at": time.time() + _seconds(h.get("x-ratelimit-reset-tokens"))}
    u = completion.usage
    usage = {"prompt_tokens": u.prompt_tokens, "completion_tokens": u.completion_tokens, "total_tokens": u.total_tokens}
    with open(USAGE_LOG, "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": _now(), "provider": provider, "model": model, "usage": usage,
                            "tpm_limit": h.get("x-ratelimit-limit-tokens"),
                            "tpm_remaining": h.get("x-ratelimit-remaining-tokens")}) + "\n")
    if completion.choices[0].finish_reason == "length":
        raise OutputTruncated(f"{provider}/{model}: answer cut off at {usage['completion_tokens']} tokens")
    return completion.choices[0].message.content or "", usage


def is_cached(prompt: str, role: str = "main", model: str | None = None) -> bool:
    """True if some candidate for this role (or the forced model) already has a cached answer to this prompt."""
    chain = [c for c in LLM["roles"]["main"] + LLM["roles"]["secondary"] if c["model"] == model] if model else _chain(role)
    return any(_cache_path(c, prompt).exists() for c in chain)


def _ask(cand: dict, prompt: str, schema: type[BaseModel], verbose: bool) -> BaseModel:
    """Call one model, validate against the schema, repair once if invalid, then cache the valid answer."""
    marker = _cache_path(cand, prompt).with_suffix(".too_large")
    if marker.exists():  # this exact request was too large before: don't pay for it again
        raise RequestTooLarge(f"{cand['model']}: known too large ({marker.name[:12]})")
    try:
        return _ask_live(cand, prompt, schema, verbose)
    except RequestTooLarge:
        marker.write_text(_now(), encoding="utf-8")
        raise


def _ask_live(cand: dict, prompt: str, schema: type[BaseModel], verbose: bool) -> BaseModel:
    _throttle(cand["model"], prompt)
    start, attempt_prompt, total_tokens = time.time(), prompt, 0
    for attempt in (1, 2):
        try:
            text, usage = _call(cand, attempt_prompt)
            total_tokens += usage["total_tokens"]
            result = schema.model_validate(_parse_json(text))
            break
        except openai.BadRequestError as exc:  # Groq rejects broken JSON in JSON mode with this code
            if "json_validate_failed" not in str(exc) or attempt == 2:
                raise
            problem = "it was not valid JSON"
        except (json.JSONDecodeError, ValidationError) as exc:
            if attempt == 2:
                raise
            problem = f"it failed validation: {str(exc)[:600]}"
        print(f"[llm] invalid answer ({problem[:60]}...) -> one repair retry")
        attempt_prompt = f"{prompt}\n\nYour previous answer was rejected because {problem}. Return valid JSON only."

    # Reached only after schema validation passed: invalid answers are never cached.
    seconds = round(time.time() - start, 2)
    _cache_path(cand, prompt).write_text(json.dumps({
        "ts": _now(), "provider": cand["provider"], "model": cand["model"],
        "temperature": LLM["temperature"], "params": cand.get("params", {}),
        "usage": usage,                       # token counts of the accepted answer
        "total_tokens_incl_retries": total_tokens,
        "seconds": seconds, "prompt": prompt, "response": text, "parsed": result.model_dump(mode="json"),
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    if verbose:
        tpd = LLM["providers"][cand["provider"]].get("tokens_per_day")
        day = f", last 24h: {tokens_used_last_24h(cand['model'])}/{tpd}" if tpd else ""
        left = _limits.get(cand["model"], {}).get("remaining_tokens")
        minute = f"left this minute: {left}" if left is not None else "no per-minute headers"
        print(f"[llm] {cand['provider']}/{cand['model']}  {total_tokens} tokens  {seconds}s  ({minute}{day})")
    return result


def complete_json(prompt: str, schema: type[BaseModel], role: str = "main",
                  verbose: bool = True, model: str | None = None) -> tuple[BaseModel, str]:
    """Ask the pinned model for `role` and return (validated result, model name).

    The prompt must describe the JSON shape and mention JSON (JSON mode requires it).
    Same prompt + same model = served from cache, never a second API call.
    `model` forces one specific configured model (used by the bake-off), with no fallback.
    """
    chain = _chain(role)
    if model:
        chain = [c for c in LLM["roles"]["main"] + LLM["roles"]["secondary"] if c["model"] == model][:1]
        if not chain:
            raise ValueError(f"Model {model!r} is not configured in llm.roles")

    # 1. Cache first (no network): the highest-ranked model that already answered this prompt.
    for cand in chain:
        path = _cache_path(cand, prompt)
        if path.exists():
            if verbose:
                print(f"[llm] cache hit  {cand['provider']}/{cand['model']}  {path.name[:12]}")
            return schema.model_validate(json.loads(path.read_text(encoding="utf-8"))["parsed"]), cand["model"]

    # 2. Live call on the highest-ranked available model. Switch only when a model is really gone.
    for cand in chain:
        if (cand["provider"], cand["model"]) in _dead:
            continue
        if cand["model"] not in _available(cand["provider"]):
            _mark_dead(role, cand, "not in the provider's current model list")
            continue
        try:
            return _ask(cand, prompt, schema, verbose), cand["model"]
        except openai.NotFoundError as exc:
            _mark_dead(role, cand, f"NotFoundError: {exc}")
        except openai.BadRequestError as exc:
            if not any(m in str(exc).lower() for m in GONE_MARKERS):
                raise  # a bug in our request, not a missing model: surface it
            _mark_dead(role, cand, f"BadRequestError: {exc}")
        except (openai.APIConnectionError, openai.APITimeoutError, openai.InternalServerError) as exc:
            _mark_dead(role, cand, f"repeated failures: {type(exc).__name__}")
    raise NoModelAvailable(f"No model available for role '{role}'. Tried: {[c['model'] for c in chain]}")


# ---------- self-tests ----------

class _Aspect(BaseModel):
    phrase: str
    sentiment: Literal["neg", "pos"]


class _Review(BaseModel):
    id: str
    aspects: list[_Aspect]


class _TestResult(BaseModel):
    results: list[_Review]


# Synthetic reviews written only to test the plumbing. Not real data.
_TEST_PROMPT = """For EACH review, list every distinct issue or praise as a short English phrase (3-8 words),
even if the review is in Hinglish. Give sentiment "neg" or "pos".
Return JSON: {"results": [{"id": "t1", "aspects": [{"phrase": "...", "sentiment": "neg"}]}]}
<<<t1>>> Captain ne ride cancel kar di aur cancellation fee mere se kaat li.
<<<t2>>> Very good app, cheap bike rides and the captain was polite."""


def _show(result: _TestResult) -> str:
    return " | ".join(f"{r.id}: " + "; ".join(f"{a.phrase} ({a.sentiment})" for a in r.aspects) for r in result.results)


def _probe() -> None:
    """One tiny live call per configured candidate: which models actually answer on our free tiers?"""
    seen = set()
    for role in ("main", "secondary"):
        for cand in LLM["roles"][role]:
            if cand["model"] in seen:
                continue
            seen.add(cand["model"])
            label = f"{role:<9} {cand['provider']}/{cand['model']}"
            if cand["model"] not in _available(cand["provider"]):
                print(f"{label}: NOT IN CATALOG")
                continue
            try:
                result = _ask(cand, _TEST_PROMPT, _TestResult, verbose=False)
                print(f"{label}: OK   {_show(result)}")
            except Exception as exc:  # report and keep probing the others
                print(f"{label}: FAILED {type(exc).__name__}: {str(exc)[:250]}")


def _acceptance() -> None:
    """Step 0 acceptance: schema-valid JSON from both roles; the identical second call hits the cache."""
    for role in ("main", "secondary"):
        for n in (1, 2):
            result, model = complete_json(_TEST_PROMPT, _TestResult, role=role)
            print(f"  {role} call {n} -> model={model}: {_show(result)}")


if __name__ == "__main__":
    _probe() if "--probe" in sys.argv else _acceptance()
