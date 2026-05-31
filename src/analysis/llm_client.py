# -*- coding: utf-8 -*-
"""
Unified LLM client with:
  1. TokenSaver middleware (rule-based compression + LRU cache)
  2. Automatic fallback chain:
       Groq llama-3.3-70b-versatile  (free tier)
       Groq qwen/qwen3-32b            (free tier alternative)
       OpenAI gpt-4o-mini             (paid fallback)
"""
import os, json, sys
from pathlib import Path

# ── Add TokenSaver to path so we can import its zero-cost rule optimizer ──────
_TS_PATH = Path(__file__).parent.parent.parent / "TokenSaver - Copy"
if str(_TS_PATH) not in sys.path:
    sys.path.insert(0, str(_TS_PATH))

try:
    from rule_optimizer import rule_optimize as _rule_optimize
    from prompt_cache import cache_get, cache_set, cache_stats as _cache_stats
    from token_counter import count_tokens as _count_tokens
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False

GROQ_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_FALLBACK = "qwen/qwen3-32b"
OPENAI_MODEL  = "gpt-4o-mini"

# Financial prompts must preserve all numbers — use light compression only
_FINANCE_AGGRESSIVENESS = "light"


def _optimize_messages(messages: list) -> tuple[list, dict]:
    """
    Apply TokenSaver rule optimizer to user message content.
    Returns (optimized_messages, stats_dict).
    Only touches the last user message; system/structure is preserved.
    Skips JSON-response prompts to avoid corrupting format instructions.
    """
    if not _TS_AVAILABLE:
        return messages, {"skipped": "TokenSaver not available"}

    stats = {"original_tokens": 0, "optimized_tokens": 0, "savings_pct": 0, "cache_hit": False}
    optimized = list(messages)

    for idx, msg in enumerate(optimized):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 50:
            continue  # too short to compress meaningfully

        # Check cache first
        cached = cache_get(content, _FINANCE_AGGRESSIVENESS)
        if cached:
            optimized[idx] = {**msg, "content": cached}
            stats["cache_hit"] = True
            stats["savings_pct"] = round((1 - len(cached) / len(content)) * 100, 1)
            continue

        orig_tokens = _count_tokens(content)
        stats["original_tokens"] += orig_tokens

        try:
            opt_content = _rule_optimize(content, aggressiveness=_FINANCE_AGGRESSIVENESS)
            if not isinstance(opt_content, str):
                opt_content = content
        except Exception:
            opt_content = content  # fall back to original on any error

        opt_tokens = _count_tokens(opt_content)
        stats["optimized_tokens"] += opt_tokens
        saved = orig_tokens - opt_tokens
        if saved > 0:
            cache_set(content, _FINANCE_AGGRESSIVENESS, opt_content)
            optimized[idx] = {**msg, "content": opt_content}

    if stats["original_tokens"] > 0:
        stats["savings_pct"] = round(
            (stats["original_tokens"] - stats["optimized_tokens"])
            / stats["original_tokens"] * 100, 1
        )

    return optimized, stats


def token_saver_stats() -> dict:
    """Return LRU cache hit/miss stats."""
    if _TS_AVAILABLE:
        return _cache_stats()
    return {"available": False}


# ── LLM provider helpers ──────────────────────────────────────────────────────

def _groq(model, messages, response_format, temperature, max_tokens):
    from groq import Groq
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        raise RuntimeError("GROQ_API_KEY not set")
    kwargs = dict(model=model, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if response_format:
        kwargs["response_format"] = response_format
    resp = Groq(api_key=key).chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _openai(messages, response_format, temperature, max_tokens):
    from openai import OpenAI
    key = os.environ.get("OPENAI_API_KEY", "")
    if not key:
        raise RuntimeError("OPENAI_API_KEY not set")
    kwargs = dict(model=OPENAI_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if response_format:
        kwargs["response_format"] = {"type": "json_object"}
    resp = OpenAI(api_key=key).chat.completions.create(**kwargs)
    return resp.choices[0].message.content


# ── Public API ────────────────────────────────────────────────────────────────

def complete(messages: list, response_format=None,
             temperature: float = 0.1, max_tokens: int = 1000) -> str:
    """
    Optimize prompt via TokenSaver rules, then call LLMs in fallback order.
    Returns raw string content.
    """
    # Apply TokenSaver middleware (zero-cost rule compression + LRU cache)
    opt_messages, ts_stats = _optimize_messages(messages)
    if ts_stats.get("savings_pct", 0) > 0:
        print(f"[TokenSaver] {ts_stats['savings_pct']}% tokens saved "
              f"({'cache hit' if ts_stats.get('cache_hit') else 'rule opt'})")

    errors = []

    # 1. Groq primary
    try:
        return _groq(GROQ_PRIMARY, opt_messages, response_format, temperature, max_tokens)
    except Exception as e:
        errors.append(f"Groq/{GROQ_PRIMARY}: {e}")
        if "429" not in str(e) and "rate" not in str(e).lower() and "quota" not in str(e).lower():
            raise

    # 2. Groq secondary
    try:
        return _groq(GROQ_FALLBACK, opt_messages, response_format, temperature, max_tokens)
    except Exception as e:
        errors.append(f"Groq/{GROQ_FALLBACK}: {e}")

    # 3. OpenAI paid fallback
    try:
        print(f"[llm_client] Groq quota exhausted, using OpenAI. Errors: {errors}")
        return _openai(opt_messages, response_format, temperature, max_tokens)
    except Exception as e:
        errors.append(f"OpenAI/{OPENAI_MODEL}: {e}")

    raise RuntimeError(f"All LLMs failed: {errors}")


def complete_json(messages: list, temperature: float = 0.1,
                  max_tokens: int = 1000) -> dict:
    """Convenience wrapper — returns parsed dict."""
    raw = complete(messages,
                   response_format={"type": "json_object"},
                   temperature=temperature,
                   max_tokens=max_tokens)
    return json.loads(raw)
