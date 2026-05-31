# -*- coding: utf-8 -*-
"""
Unified LLM client with automatic fallback:
  1. Groq  llama-3.3-70b-versatile  (free tier)
  2. Groq  qwen/qwen3-32b            (free tier alternative)
  3. OpenAI gpt-4o-mini              (paid fallback)

Fallback triggers on: RateLimitError, quota exhausted (429),
or any Groq connectivity failure.
"""
import os, json

GROQ_PRIMARY  = "llama-3.3-70b-versatile"
GROQ_FALLBACK = "qwen/qwen3-32b"
OPENAI_MODEL  = "gpt-4o-mini"


def _groq(model, messages, response_format, temperature, max_tokens):
    from groq import Groq, RateLimitError
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


def complete(messages: list, response_format=None,
             temperature: float = 0.1, max_tokens: int = 1000) -> str:
    """
    Call LLMs in order until one succeeds.
    Returns the raw string content from the model.
    """
    errors = []

    # 1. Groq primary
    try:
        return _groq(GROQ_PRIMARY, messages, response_format, temperature, max_tokens)
    except Exception as e:
        errors.append(f"Groq/{GROQ_PRIMARY}: {e}")
        if "429" not in str(e) and "rate" not in str(e).lower() and "quota" not in str(e).lower():
            raise  # non-rate-limit error — don't fall back silently

    # 2. Groq secondary model
    try:
        return _groq(GROQ_FALLBACK, messages, response_format, temperature, max_tokens)
    except Exception as e:
        errors.append(f"Groq/{GROQ_FALLBACK}: {e}")

    # 3. OpenAI paid fallback
    try:
        print(f"[llm_client] Groq quota exhausted, using OpenAI. Previous errors: {errors}")
        return _openai(messages, response_format, temperature, max_tokens)
    except Exception as e:
        errors.append(f"OpenAI/{OPENAI_MODEL}: {e}")

    raise RuntimeError(f"All LLMs failed: {errors}")


def complete_json(messages: list, temperature: float = 0.1,
                  max_tokens: int = 1000) -> dict:
    """Convenience wrapper that returns a parsed dict."""
    raw = complete(messages,
                   response_format={"type": "json_object"},
                   temperature=temperature,
                   max_tokens=max_tokens)
    return json.loads(raw)
