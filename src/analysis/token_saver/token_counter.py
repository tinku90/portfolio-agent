import re
import tiktoken
from typing import Optional


# Cost per 1K INPUT tokens (approximate, update regularly)
COST_PER_1K_INPUT = {
    # OpenAI
    "gpt-4o":          0.0025,
    "gpt-4o-mini":     0.00015,
    "gpt-4-turbo":     0.01,
    "gpt-3.5-turbo":   0.0005,
    # Anthropic
    "claude-opus-4-6":           0.015,
    "claude-sonnet-4-6":         0.003,
    "claude-haiku-4-5-20251001": 0.00025,
    # Google
    "gemini-1.5-pro":   0.00125,
    "gemini-1.5-flash": 0.000075,
    "gemini-2.0-flash": 0.0001,
    # Qwen (DashScope)
    "qwen-coder-plus":             0.0011,
    "qwen2.5-coder-32b-instruct":  0.0011,
    "qwen-plus":                   0.0008,
    "qwen-turbo":                  0.0005,
    # Default fallback
    "default": 0.002,
}

# Cost per 1K OUTPUT tokens — typically 3-5x the input rate
COST_PER_1K_OUTPUT = {
    # OpenAI
    "gpt-4o":          0.010,
    "gpt-4o-mini":     0.00060,
    "gpt-4-turbo":     0.030,
    "gpt-3.5-turbo":   0.0015,
    # Anthropic
    "claude-opus-4-6":           0.075,
    "claude-sonnet-4-6":         0.015,
    "claude-haiku-4-5-20251001": 0.00125,
    # Google
    "gemini-1.5-pro":   0.005,
    "gemini-1.5-flash": 0.00030,
    "gemini-2.0-flash": 0.0004,
    # Qwen
    "qwen-coder-plus":             0.0034,
    "qwen2.5-coder-32b-instruct":  0.0034,
    "qwen-plus":                   0.0024,
    "qwen-turbo":                  0.0015,
    # Default fallback (~4x input)
    "default": 0.008,
}


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """
    Count tokens using cl100k_base (GPT-4 encoding) as a universal approximation.
    Always uses the locally-bundled encoding — never makes a network request.
    """
    try:
        encoding = tiktoken.encoding_for_model(model)
    except Exception:
        # KeyError = unknown model; network/OSError = o200k_base not cached locally.
        # cl100k_base is always bundled with tiktoken — safe offline fallback.
        encoding = tiktoken.get_encoding("cl100k_base")

    try:
        return len(encoding.encode(text))
    except Exception:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))


def _lookup(table: dict, model: str) -> float:
    if model in table:
        return table[model]
    for key in table:
        if model.startswith(key):
            return table[key]
    return table["default"]


def get_cost_per_1k(model: str) -> float:
    """Get INPUT cost per 1K tokens for a given model."""
    return _lookup(COST_PER_1K_INPUT, model)


def get_output_cost_per_1k(model: str) -> float:
    """Get OUTPUT cost per 1K tokens for a given model."""
    return _lookup(COST_PER_1K_OUTPUT, model)


def calculate_savings(
    original_tokens: int,
    optimized_tokens: int,
    model: str = "default"
) -> dict:
    """Calculate token savings stats."""
    tokens_saved = original_tokens - optimized_tokens
    savings_percent = (tokens_saved / original_tokens * 100) if original_tokens > 0 else 0
    cost_per_1k = get_cost_per_1k(model)
    cost_saved = (tokens_saved / 1000) * cost_per_1k

    return {
        "original_tokens": original_tokens,
        "optimized_tokens": optimized_tokens,
        "tokens_saved": tokens_saved,
        "savings_percent": round(savings_percent, 1),
        "estimated_cost_saved_usd": round(cost_saved, 6),
    }


def analyze_verbosity(text: str) -> float:
    """
    Score text verbosity from 0 (terse) to 1 (very verbose/compressible).
    Simple heuristic based on filler word density.
    """
    filler_phrases = [
        "please", "could you", "can you", "I would like", "I want you to",
        "I need you to", "kindly", "as you know", "it is important to note",
        "in order to", "due to the fact that", "at this point in time",
        "in the event that", "for the purpose of", "with regard to",
        "it should be noted", "as mentioned", "basically", "essentially",
        "very", "really", "quite", "actually", "certainly", "definitely",
        "absolutely", "of course", "please note that", "please be aware",
        "I am writing to", "I wanted to", "just wanted to",
    ]

    text_lower = text.lower()
    word_count = len(text.split())
    if word_count == 0:
        return 0.0

    filler_count = sum(text_lower.count(phrase) for phrase in filler_phrases)
    # Normalize: more than 10 fillers per 100 words = highly verbose
    score = min(1.0, filler_count / max(word_count / 10, 1))
    return round(score, 2)


# ── Output token estimation ──────────────────────────────────────────────────

# (min, typical, max) heuristic token ranges per prompt type
_OUTPUT_TOKEN_RANGES: dict[str, tuple[int, int, int]] = {
    # Legacy decision-engine types
    "question":   (80,   150,  250),
    "analysis":   (250,  400,  650),
    "creative":   (400,  700, 1100),
    "task":       (100,  250,  500),
    "code_task":  (350,  600, 1000),

    # Classifier types (prompt_classifier.py) — must match real ChatGPT behaviour:
    # explanation_complex: multi-section asks (history + industries + examples)
    #   real outputs measured at 350-700 tokens; use 500 typical
    "explanation_complex":  (250, 500, 900),
    # explanation_simple: "explain X" — a few paragraphs
    "explanation_simple":   (150, 300, 600),
    # factual: bounded by nature — short answer
    "factual":              (40,  90,  200),
    # list / comparison: bullets but can be long; 250 typical
    "list":                 (120, 250, 500),
    # critical analysis/review: dense evaluative reasoning with evidence and gaps
    "analysis_critic":      (180, 380, 800),
    # code: varies widely; 400 typical (explanation + code block)
    "code":                 (250, 450, 900),
    # creative: user-defined length; keep existing creative range
    "general":              (80,  200, 450),
}


# Live-calibrated overrides populated by analytics.record_type_sample()
# Format: {prompt_type: (min, typical, max)} — replaces static prior once N≥10 samples
_calibrated_ranges: dict[str, tuple[int, int, int]] = {}


def set_calibrated_range(prompt_type: str, min_: int, typical: int, max_: int) -> None:
    """Called by analytics when enough real samples accumulate for a prompt type."""
    _calibrated_ranges[prompt_type] = (min_, typical, max_)


_DETAIL_INTENT_WORDS = (
    "in very great detail", "in great detail", "in detail",
    "step by step", "step-by-step", "with examples", "with real examples",
    "comprehensive", "comprehensively", "thorough", "thoroughly",
    "elaborate", "in depth", "in-depth", "end-to-end", "detailed",
    "deep dive", "exhaustive", "extensively",
)

# Multi-topic signals: each matched topic adds depth to the expected response
_MULTI_TOPIC_RE = re.compile(
    r"\b(authentication|authorization|rate.?limit|retry|pagination|caching"
    r"|logging|monitoring|security|deployment|testing|validation|error.?handling"
    r"|performance|scalability|microservice|database|api|architecture)\b",
    re.IGNORECASE,
)


# Reduction factors applied to the output range when a constraint suffix is present.
# Keys match what main.py detects from the optimized prompt.
_CONSTRAINT_REDUCTION: dict[str, float] = {
    "concise":       0.48,   # "Be concise. Max 3 sentences. No preamble."
    "direct":        0.68,   # "Answer directly. No intro or outro."
    "bullet":        0.58,   # "Bullet points only."
    "code_only":     0.88,   # "Complete working code only." — code stays dense
    "structured":    0.78,   # "Use clear sections. No filler."
    "none":          1.00,
}


def estimate_output_range(
    prompt: str,
    prompt_type: str,
    constraint_type: str = "none",
) -> dict:
    """
    Return output token ranges for both unoptimized and optimized versions.

    constraint_type: one of "concise" | "direct" | "bullet" | "code_only" |
                     "structured" | "none"

    Returns:
        {
            "unoptimized": {"min": int, "typical": int, "max": int},
            "optimized":   {"min": int, "typical": int, "max": int},
            "constraint_type": str,
            "prompt_type": str,
        }
    """
    base = estimate_output_tokens(prompt, prompt_type)
    factor = _CONSTRAINT_REDUCTION.get(constraint_type, 1.0)

    def _apply(d: dict, f: float) -> dict:
        return {
            "min":     max(10, round(d["min"]     * f)),
            "typical": max(10, round(d["typical"] * f)),
            "max":     max(10, round(d["max"]     * f)),
        }

    return {
        "unoptimized":      _apply(base, 1.0),
        "optimized":        _apply(base, factor),
        "constraint_type":  constraint_type,
        "prompt_type":      prompt_type,
    }


def estimate_output_tokens(prompt: str, prompt_type: str) -> dict:
    """
    Estimate expected output tokens using prompt-type heuristics.
    Scales with:
      - input length  (longer input → more output)
      - detail intent (explicit depth request → 3x multiplier)
      - topic count   (multi-topic prompts → up to 2x multiplier)

    Returns:
        {
            "min": int,
            "typical": int,
            "max": int,
            "prompt_type": str,
        }
    """
    # Use live-calibrated range if available, else static prior
    _cal = _calibrated_ranges.get(prompt_type)
    low, mid, high = _cal if _cal else _OUTPUT_TOKEN_RANGES.get(prompt_type, (100, 250, 500))

    # Scale slightly with input length: every 100 input tokens adds ~5% to estimate
    input_tokens = count_tokens(prompt)
    scale = 1.0 + min(input_tokens / 2000, 0.5)  # cap at +50%

    # Intent multiplier: user explicitly asked for depth → expect 2.5-3x more output
    prompt_lower = prompt.lower()
    if any(w in prompt_lower for w in _DETAIL_INTENT_WORDS):
        scale *= 3.0

    # Multi-topic boost: each distinct topic adds ~25% to expected output
    topic_count = len(set(_MULTI_TOPIC_RE.findall(prompt_lower)))
    if topic_count >= 4:
        scale *= 2.0
    elif topic_count >= 2:
        scale *= 1.5

    return {
        "min":         round(low  * scale),
        "typical":     round(mid  * scale),
        "max":         round(high * scale),
        "prompt_type": prompt_type,
    }


# ── Calibrated output estimate (blends prior with rolling actuals) ────────────

_MIN_SAMPLES_TO_CALIBRATE = 5      # don't trust fewer than 5 actuals
_BLEND_WEIGHT_ACTUAL      = 0.7    # 70% actual, 30% prior once calibrated


def get_calibrated_estimate(
    prompt: str,
    prompt_type: str,
    history: list,           # pass list(analytics._history) directly
) -> dict:
    """
    Blends the hardcoded prior with rolling actual averages from history.
    Falls back to static estimate when sample count < _MIN_SAMPLES_TO_CALIBRATE.
    Used for display only — constraint injection keeps using the static prior.
    """
    static = estimate_output_tokens(prompt, prompt_type)

    actuals = [
        h["actual_output_tokens"]
        for h in history
        if h.get("prompt_type") == prompt_type
        and h.get("actual_output_tokens") is not None
        and h["actual_output_tokens"] > 0
    ]

    if len(actuals) < _MIN_SAMPLES_TO_CALIBRATE:
        return {**static, "calibrated": False, "sample_count": len(actuals)}

    rolling_avg = sum(actuals) / len(actuals)

    blended_typical = int(
        _BLEND_WEIGHT_ACTUAL * rolling_avg
        + (1 - _BLEND_WEIGHT_ACTUAL) * static["typical"]
    )

    ratio_min = static["min"] / static["typical"] if static["typical"] > 0 else 0.5
    ratio_max = static["max"] / static["typical"] if static["typical"] > 0 else 2.0

    return {
        "min":          int(blended_typical * ratio_min),
        "typical":      blended_typical,
        "max":          int(blended_typical * ratio_max),
        "prompt_type":  prompt_type,
        "calibrated":   True,
        "sample_count": len(actuals),
        "rolling_avg":  round(rolling_avg, 1),
    }


# ── Detailed savings calculation ─────────────────────────────────────────────

def calculate_full_savings(
    original_input_tokens: int,
    optimized_input_tokens: int,
    estimated_output_original: int,
    actual_output_tokens: int,
    model: str = "default",
) -> dict:
    """
    Full cost-aware savings breakdown using separate input and output pricing.

    Output tokens are priced at 3-5x the input rate depending on model, so
    even small output reductions have outsized cost impact.

    Returns:
        {
            "input_saved": int,
            "output_saved_estimated": int,
            "total_saved": int,               # token count (input + output)
            "savings_percent": float,          # % of total original tokens saved
            "estimated_cost_saved_usd": float, # using real input+output pricing
            "usage_multiplier": float,         # e.g. 1.43 = 43% more prompts per dollar
        }
    """
    input_saved  = max(0, original_input_tokens - optimized_input_tokens)
    output_saved = max(0, estimated_output_original - actual_output_tokens)
    total_saved  = input_saved + output_saved

    total_original = original_input_tokens + estimated_output_original
    savings_pct = (
        round(total_saved / total_original * 100, 1) if total_original > 0 else 0.0
    )

    # Use separate input / output rates — output typically costs 3-5x more
    input_rate  = get_cost_per_1k(model)
    output_rate = get_output_cost_per_1k(model)
    cost_saved  = round(
        (input_saved  / 1000) * input_rate +
        (output_saved / 1000) * output_rate,
        6,
    )

    # Usage multiplier: if you save X%, your budget goes (1/(1-X%)) further
    usage_multiplier = (
        round(1 / (1 - savings_pct / 100), 2)
        if 0 < savings_pct < 100 else 1.0
    )

    return {
        "input_saved":              input_saved,
        "output_saved_estimated":   output_saved,
        "total_saved":              total_saved,
        "savings_percent":          savings_pct,
        "estimated_cost_saved_usd": cost_saved,
        "usage_multiplier":         usage_multiplier,
    }


def get_optimization_suggestions(text: str, decision: dict | None = None) -> list[str]:
    """
    Return human-readable suggestions for improving prompt efficiency.

    Pass the result of decide_strategy() as `decision` so suggestions
    speak with the same voice as the decision engine — no contradictions.
    """
    suggestions = []
    text_lower = text.lower()

    if any(p in text_lower for p in ["please", "could you", "can you", "kindly"]):
        suggestions.append("Remove polite filler words (please, could you, kindly) — LLMs don't need them")

    if any(p in text_lower for p in ["i would like", "i want you to", "i need you to"]):
        suggestions.append("Replace 'I would like you to...' with a direct imperative (e.g., 'Summarize:')")

    if "in order to" in text_lower:
        suggestions.append("Replace 'in order to' with 'to'")

    if "due to the fact that" in text_lower:
        suggestions.append("Replace 'due to the fact that' with 'because'")

    if len(text) > 500 and len(suggestions) == 0:
        suggestions.append("Consider breaking this into a shorter system prompt + concise user message")

    if text.count("\n\n") > 4:
        suggestions.append("Reduce blank lines — they consume tokens with no value")

    # ── Decision-engine-aware suggestions ────────────────────────────────────
    if decision:
        typical_out = decision.get("estimated_output_typical", 0)
        add_constraints = decision.get("add_constraints", False)
        optimize = decision.get("optimize", False)

        if add_constraints and typical_out >= 300:
            suggestions.append(
                f"High output expected (~{typical_out} tokens) — "
                "output constraints will be injected to cap response length"
            )
        elif not optimize and typical_out < 120:
            suggestions.append(
                f"Low output expected (~{typical_out} tokens) — "
                "optimization skipped, this call is already cost-effective"
            )

    if not suggestions:
        suggestions.append("Prompt looks efficient — minimal optimization needed")

    return suggestions
