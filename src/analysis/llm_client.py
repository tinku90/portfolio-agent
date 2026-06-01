# -*- coding: utf-8 -*-
"""
Unified LLM client with TokenSaver middleware and automatic fallback chain:

  1. Groq  llama-3.3-70b-versatile   (free - 100K TPD)
  2. Groq  qwen/qwen3-32b             (free - separate TPD limit)
  3. Gemini 2.0 Flash                 (free - 1M tokens/day via Google AI Studio)
  4. OpenAI gpt-4o-mini               (paid - your OpenAI key)

Rate-limit detection uses exception TYPE (not fragile string parsing).
Every attempt is logged so you can see exactly which model ran.
"""
import os, json, sys
from pathlib import Path

# Windows consoles default to cp1252 which crashes on Unicode in print().
# Reconfigure stdout/stderr to UTF-8 so logging never raises UnicodeEncodeError.
for _stream in (sys.stdout, sys.stderr):
    try:
        _stream.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

try:
    from src.analysis.token_saver import finance_optimize as _optimize, \
        cache_get, cache_set, cache_stats as _cache_stats, count_tokens as _count_tokens
    from src.analysis.token_saver import response_cache as _rcache
    _TS_AVAILABLE = True
except ImportError:
    _TS_AVAILABLE = False
    _rcache = None

# -- Model names ---------------------------------------------------------------
GROQ_PRIMARY    = "llama-3.3-70b-versatile"
GROQ_SECONDARY  = "qwen/qwen3-32b"
GEMINI_MODEL    = "gemini-2.0-flash"
OPENAI_MODEL    = "gpt-4o-mini"


# -- TokenSaver middleware -----------------------------------------------------

def _optimize_messages(messages: list) -> tuple[list, dict]:
    if not _TS_AVAILABLE:
        return messages, {}
    stats     = {"original_tokens": 0, "optimized_tokens": 0, "savings_pct": 0}
    optimized = list(messages)
    for idx, msg in enumerate(optimized):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str) or len(content) < 50:
            continue
        cached = cache_get(content, "light")
        if cached:
            optimized[idx] = {**msg, "content": cached}
            continue
        orig = _count_tokens(content)
        try:
            opt = _optimize(content)
        except Exception:
            opt = content
        opt_tok = _count_tokens(opt)
        if orig > opt_tok:
            cache_set(content, "light", opt)
            optimized[idx] = {**msg, "content": opt}
            stats["original_tokens"]  += orig
            stats["optimized_tokens"] += opt_tok
    if stats["original_tokens"] > 0:
        stats["savings_pct"] = round(
            (stats["original_tokens"] - stats["optimized_tokens"])
            / stats["original_tokens"] * 100, 1)
    return optimized, stats


def token_saver_stats() -> dict:
    return _cache_stats() if _TS_AVAILABLE else {"available": False}


# -- Rate-limit detection ------------------------------------------------------

def _is_rate_limit(e: Exception) -> bool:
    """
    True if this exception is a quota / rate-limit error.
    Checks exception TYPE first (reliable), then falls back to string matching.
    """
    try:
        from groq import RateLimitError as GroqRLE
        if isinstance(e, GroqRLE):
            return True
    except ImportError:
        pass
    try:
        from openai import RateLimitError as OpenAIRLE
        if isinstance(e, OpenAIRLE):
            return True
    except ImportError:
        pass
    s = str(e).lower()
    return any(kw in s for kw in ("429", "rate limit", "quota", "rate_limit",
                                   "tokens per day", "requests per day",
                                   "resource_exhausted", "too many requests"))


# -- Provider helpers ----------------------------------------------------------

def _call_groq(model: str, messages, response_format, temperature, max_tokens) -> str:
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


def _call_gemini(messages, response_format, temperature, max_tokens) -> str:
    """
    Uses Google's OpenAI-compatible endpoint - same interface, no extra SDK.
    Get a free key at: https://aistudio.google.com/app/apikey
    """
    from openai import OpenAI
    key = os.environ.get("GEMINI_API_KEY", "")
    if not key:
        raise RuntimeError("GEMINI_API_KEY not set")
    client = OpenAI(
        api_key=key,
        base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    )
    kwargs = dict(model=GEMINI_MODEL, messages=messages,
                  temperature=temperature, max_tokens=max_tokens)
    if response_format:
        kwargs["response_format"] = {"type": "json_object"}
    resp = client.chat.completions.create(**kwargs)
    return resp.choices[0].message.content


def _call_openai(messages, response_format, temperature, max_tokens) -> str:
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


# -- Fallback chain ------------------------------------------------------------

_PROVIDERS = [
    ("Groq/llama-3.3-70b",  lambda m, rf, t, mx: _call_groq(GROQ_PRIMARY,   m, rf, t, mx)),
    ("Groq/qwen3-32b",      lambda m, rf, t, mx: _call_groq(GROQ_SECONDARY,  m, rf, t, mx)),
    ("Gemini/flash-2.0",    lambda m, rf, t, mx: _call_gemini(m, rf, t, mx)),
    ("OpenAI/gpt-4o-mini",  lambda m, rf, t, mx: _call_openai(m, rf, t, mx)),
]


def complete(messages: list, response_format=None,
             temperature: float = 0.1, max_tokens: int = 1000,
             use_cache: bool = True) -> str:
    """
    Try each provider in order. Falls back on rate-limit or quota errors.
    Raises immediately on auth / format / non-quota errors from the first provider.
    Identical requests are served from the disk response cache (use_cache=False to force fresh).
    """
    # TokenSaver: compress prompts before sending
    opt_msgs, ts = _optimize_messages(messages)
    if ts.get("savings_pct", 0) > 0:
        print(f"[TokenSaver] {ts['savings_pct']}% saved")

    # LLM output cache: identical request -> served from disk, zero API cost
    cache_key = None
    if _rcache and use_cache:
        cache_key = _rcache.make_key(opt_msgs, response_format, temperature, max_tokens)
        hit = _rcache.get(cache_key)
        if hit is not None:
            print("[LLM cache] HIT - served from cache, no API call")
            return hit

    errors      = []
    hard_failed = False   # a non-recoverable error occurred on the first provider

    for name, fn in _PROVIDERS:
        try:
            print(f"[LLM] trying {name}...")
            result = fn(opt_msgs, response_format, temperature, max_tokens)
            print(f"[LLM] {name} OK")
            if cache_key:
                _rcache.set(cache_key, result)
            return result
        except Exception as e:
            err_str = f"{name}: {type(e).__name__}: {e}"
            errors.append(err_str)
            print(f"[LLM] {name} FAILED - {type(e).__name__}: {str(e)[:120]}")

            if _is_rate_limit(e):
                print(f"[LLM] rate limit / quota hit on {name} -> trying next provider")
                continue   # always fall through on rate limits

            # Non-quota error on first provider = likely a bad prompt / auth issue.
            # Still try next providers but flag it.
            if name == _PROVIDERS[0][0]:
                hard_failed = True
                print(f"[LLM] non-quota error on primary - still trying fallbacks")
            continue

    raise RuntimeError(
        f"All {len(_PROVIDERS)} LLM providers failed.\n" +
        "\n".join(f"  {e}" for e in errors)
    )


def complete_json(messages: list, temperature: float = 0.1,
                  max_tokens: int = 1000, use_cache: bool = True) -> dict:
    raw = complete(messages,
                   response_format={"type": "json_object"},
                   temperature=temperature,
                   max_tokens=max_tokens,
                   use_cache=use_cache)
    return json.loads(raw)


def response_cache_stats() -> dict:
    return _rcache.stats() if _rcache else {"available": False}


def clear_response_cache():
    if _rcache:
        _rcache.clear()


# -- Vision (image -> text) with fallback ---------------------------------------

_VISION_PROVIDERS = [
    ("Groq/llama-4-scout", "groq",   "meta-llama/llama-4-scout-17b-16e-instruct"),
    ("Gemini/flash-2.0",   "gemini", "gemini-2.0-flash"),
    ("OpenAI/gpt-4o-mini", "openai", "gpt-4o-mini"),
]


def complete_vision(prompt_text: str, image_b64: str, mime: str = "image/jpeg",
                    max_tokens: int = 2000) -> str:
    """
    Extract text/data from an image. Falls back across vision-capable providers.
    Uses the OpenAI-compatible content format which all three accept.
    """
    content = [
        {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{image_b64}"}},
        {"type": "text", "text": prompt_text},
    ]
    messages = [{"role": "user", "content": content}]
    errors = []

    # Cache by image bytes + prompt — re-uploading the same image is free
    vkey = None
    if _rcache:
        vkey = _rcache.make_key([{"img": image_b64, "p": prompt_text}], None, 0, max_tokens)
        hit = _rcache.get(vkey)
        if hit is not None:
            print("[Vision cache] HIT - image already extracted, no API call")
            return hit

    for name, kind, model in _VISION_PROVIDERS:
        try:
            print(f"[Vision] trying {name}...")
            if kind == "groq":
                from groq import Groq
                key = os.environ.get("GROQ_API_KEY", "")
                if not key:
                    raise RuntimeError("GROQ_API_KEY not set")
                resp = Groq(api_key=key).chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens)
            elif kind == "gemini":
                from openai import OpenAI
                key = os.environ.get("GEMINI_API_KEY", "")
                if not key:
                    raise RuntimeError("GEMINI_API_KEY not set")
                resp = OpenAI(api_key=key,
                    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
                ).chat.completions.create(model=model, messages=messages, max_tokens=max_tokens)
            else:  # openai
                from openai import OpenAI
                key = os.environ.get("OPENAI_API_KEY", "")
                if not key:
                    raise RuntimeError("OPENAI_API_KEY not set")
                resp = OpenAI(api_key=key).chat.completions.create(
                    model=model, messages=messages, max_tokens=max_tokens)
            print(f"[Vision] {name} OK")
            out = resp.choices[0].message.content
            if vkey:
                _rcache.set(vkey, out)
            return out
        except Exception as e:
            errors.append(f"{name}: {type(e).__name__}: {e}")
            print(f"[Vision] {name} FAILED - {type(e).__name__}: {str(e)[:100]}")
            continue

    return f"[Image extraction failed across all providers: {errors}]"
