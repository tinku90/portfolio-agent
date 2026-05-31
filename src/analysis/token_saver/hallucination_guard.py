# -*- coding: utf-8 -*-
"""
HallucinationGuard — three-layer defence against LLM fabrication.

Layer 1: ground_prompt()      — inject verified facts before sending
Layer 2: anti_hallucination_suffix() — append explicit "don't guess" rules
Layer 3: validate_response()  — sanity-check numbers post-response

Designed for financial analysis but generic enough for any domain.
"""
import re
from typing import Any


# ── Layer 1: Factual grounding ────────────────────────────────────────────────

def ground_prompt(prompt: str, facts: dict) -> str:
    """
    Prepend a VERIFIED FACTS block to the prompt.
    The LLM is instructed to use ONLY these numbers for calculations.

    facts: dict of {label: value} — e.g. {"Current price": 214.21, "P/E": 45.2}
    Only non-None, non-zero values are injected.
    """
    if not facts:
        return prompt

    lines = []
    for label, value in facts.items():
        if value is None or value == 0 or value == "":
            continue
        lines.append(f"  {label}: {value}")

    if not lines:
        return prompt

    grounding = (
        "VERIFIED MARKET DATA — use these exact figures in all calculations. "
        "Do NOT substitute your own estimates for any value listed here:\n"
        + "\n".join(lines)
        + "\n\n"
    )
    return grounding + prompt


# ── Layer 2: Anti-hallucination instruction suffix ───────────────────────────

_FINANCIAL_GUARD = """
ANTI-HALLUCINATION RULES (mandatory):
- Base ALL numerical calculations on the verified market data provided above.
- If a required input (EPS, FCF, growth rate) is missing from the data, return null for that field — do NOT invent a number.
- Do NOT use round numbers (e.g. exactly 20%, exactly 100) unless the data supports them.
- Show your calculation basis in the valuation_basis field.
- Flag any assumption you had to make with the prefix "ASSUMED:".
"""

_GENERAL_GUARD = """
ACCURACY RULES:
- Only state facts you are certain of. If uncertain, say so explicitly.
- Do NOT fabricate statistics, dates, names, or figures.
- If you cannot answer part of the question from available information, return null for that field.
"""


def add_guard(prompt: str, domain: str = "financial") -> str:
    """
    Append anti-hallucination instructions to a prompt.
    domain: "financial" | "general"
    """
    suffix = _FINANCIAL_GUARD if domain == "financial" else _GENERAL_GUARD
    # Avoid duplicate injection
    if "ANTI-HALLUCINATION" in prompt or "ACCURACY RULES" in prompt:
        return prompt
    return prompt.rstrip() + "\n" + suffix


# ── Layer 3: Output validation ────────────────────────────────────────────────

class ValidationResult:
    def __init__(self):
        self.passed   = True
        self.warnings = []
        self.errors   = []

    def warn(self, msg):
        self.warnings.append(msg)

    def fail(self, msg):
        self.errors.append(msg)
        self.passed = False

    def as_dict(self):
        return {
            "passed":   self.passed,
            "warnings": self.warnings,
            "errors":   self.errors,
        }


def validate_financial_response(result: dict, current_price: float,
                                 avg_cost: float = 0) -> ValidationResult:
    """
    Sanity-check a financial analysis JSON response.

    Checks:
    - Fair values within 0.05× – 10× current price
    - Target > fair value (12m target should be above intrinsic value)
    - Stop loss < current price
    - Growth rate in plausible range (-50% to +200%)
    - Required fields present and not null
    - DCF and PEG should broadly agree (warn if >3× apart)
    """
    v = ValidationResult()

    if not result or not isinstance(result, dict):
        v.fail("Response is empty or not a dict")
        return v

    # ── Required field presence ───────────────────────────────────────────────
    for field in ["thesis", "risks"]:
        if not result.get(field):
            v.warn(f"Field '{field}' is missing or empty — possible hallucination or truncation")

    # ── Numerical bounds ──────────────────────────────────────────────────────
    ref_price = current_price or avg_cost or 1  # fallback to avoid div-by-zero

    for key in ("peg_fair_value", "dcf_fair_value", "fair_value"):
        val = result.get(key)
        if val is None:
            continue
        try:
            val = float(val)
        except (TypeError, ValueError):
            v.fail(f"'{key}' is not a number: {val}")
            continue
        ratio = val / ref_price
        if ratio < 0.05:
            v.fail(f"'{key}' = {val} is <5% of current price {ref_price} — likely hallucinated")
        elif ratio > 10:
            v.fail(f"'{key}' = {val} is >10× current price {ref_price} — likely hallucinated")
        elif ratio < 0.2:
            v.warn(f"'{key}' = {val} implies >80% downside — verify assumptions")
        elif ratio > 5:
            v.warn(f"'{key}' = {val} implies >400% upside — verify assumptions")

    # ── Target should exceed fair value ───────────────────────────────────────
    for fv_key, tgt_key in [("peg_fair_value", "peg_target_12m"),
                             ("dcf_fair_value", "dcf_target_12m"),
                             ("fair_value",     "target")]:
        fv  = result.get(fv_key)
        tgt = result.get(tgt_key)
        if fv and tgt:
            try:
                if float(tgt) < float(fv) * 0.95:
                    v.warn(f"Target {tgt} is below fair value {fv} — check DCF/PEG assumptions")
            except (TypeError, ValueError):
                pass

    # ── Stop loss below current price ─────────────────────────────────────────
    sl = result.get("stop_loss")
    if sl and current_price:
        try:
            if float(sl) >= current_price:
                v.fail(f"Stop loss {sl} >= current price {current_price} — nonsensical")
            elif float(sl) < current_price * 0.3:
                v.warn(f"Stop loss {sl} is >70% below current price — unusually wide")
        except (TypeError, ValueError):
            pass

    # ── Growth rate plausibility ───────────────────────────────────────────────
    for g_key in ("expected_growth_pct", "new_growth_pct", "prev_growth_pct"):
        g = result.get(g_key)
        if g is None:
            continue
        try:
            g = float(g)
            if g < -50:
                v.fail(f"Growth rate '{g_key}' = {g}% is below -50% — implausible")
            elif g > 200:
                v.warn(f"Growth rate '{g_key}' = {g}% exceeds 200% — verify source")
        except (TypeError, ValueError):
            pass

    # ── DCF vs PEG cross-check ────────────────────────────────────────────────
    peg_fv = result.get("peg_fair_value")
    dcf_fv = result.get("dcf_fair_value")
    if peg_fv and dcf_fv:
        try:
            ratio = max(float(peg_fv), float(dcf_fv)) / max(min(float(peg_fv), float(dcf_fv)), 0.01)
            if ratio > 3:
                v.warn(
                    f"PEG FV {peg_fv} and DCF FV {dcf_fv} diverge by {ratio:.1f}× — "
                    "one may be hallucinated; review assumptions"
                )
        except (TypeError, ValueError, ZeroDivisionError):
            pass

    # ── Detect ASSUMED: flags left by LLM ────────────────────────────────────
    basis = result.get("valuation_basis", "")
    thesis = result.get("thesis", "")
    for text in [basis, thesis]:
        if isinstance(text, str) and "ASSUMED:" in text:
            v.warn("LLM flagged assumptions in valuation_basis/thesis — review carefully")
            break

    return v


# ── Convenience: full pipeline ────────────────────────────────────────────────

def prepare_prompt(prompt: str, facts: dict = None,
                   domain: str = "financial") -> str:
    """Ground + guard a prompt in one call."""
    if facts:
        prompt = ground_prompt(prompt, facts)
    prompt = add_guard(prompt, domain)
    return prompt
