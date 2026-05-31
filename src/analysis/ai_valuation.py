# -*- coding: utf-8 -*-
"""
AI Valuation - hybrid architecture:
  LLM estimates forward-looking INPUTS  (growth, WACC, EPS estimate, thesis, risks)
  Python computes all MATH              (DCF fair value, PEG fair value, targets, MOS)

This eliminates hallucinated calculations entirely.
If LLM cannot estimate an input, it returns null -> Python returns null for that method.
"""
from src.analysis.llm_client import complete_json
from src.analysis.valuation import build_valuation
from src.analysis.token_saver.hallucination_guard import (
    prepare_prompt, validate_financial_response,
)

# -- LLM prompt: ask ONLY for inputs, not calculations ------------------------
INPUTS_PROMPT = """You are a buy-side equity analyst. Estimate the INPUTS needed to value {ticker} ({market}-listed).
DO NOT calculate fair value yourself - provide only the input figures listed below.

Recent news:
{news_text}

{docs_section}

Return ONLY valid JSON. Use null for any field you cannot estimate from available data - do NOT guess:
{{
  "estimated_eps":           <number | null>,   // trailing or forward EPS per share
  "estimated_fcf_per_share": <number | null>,   // free cash flow per share (preferred over EPS for DCF)
  "estimated_growth_pct":    <number | null>,   // expected annual earnings growth % next 3-5 years
  "wacc_pct":                <number | null>,   // your WACC estimate % (or null to use default)
  "terminal_growth_pct":     <number | null>,   // perpetual growth % (usually 2-4, null = use 3)
  "business_risk":           "low|medium|high", // risk level for stop-loss calibration
  "thesis":                  "2-3 sentence investment thesis",
  "risks":                   ["risk1", "risk2", "risk3"],
  "opportunities":           ["opp1", "opp2", "opp3"],
  "data_confidence":         "high|medium|low"  // how confident you are in the estimates
}}"""


def _get_market_facts(ticker, market, avg_cost, current_price, currency) -> dict:
    """Fetch real fundamentals from yfinance to ground the prompt."""
    try:
        from src.data.prices import get_fundamentals
        f = get_fundamentals(ticker, market)
        return {
            f"Current price ({currency})": current_price,
            "Investor avg buy cost":        avg_cost,
            "Trailing P/E":                 f.get("pe"),
            "Trailing EPS":                 f.get("eps"),
            "Forward P/E":                  f.get("forward_pe"),
            "Earnings growth (YoY)":        f.get("earnings_growth"),
            "Revenue growth (YoY)":         f.get("revenue_growth"),
            "Market cap":                   f.get("market_cap"),
        }
    except Exception:
        return {
            f"Current price ({currency})": current_price,
            "Investor avg buy cost":        avg_cost,
        }


def analyze_position(ticker, market, avg_cost, current_price, currency,
                     news_items=None, extra_texts=None) -> dict:
    """
    Step 1: LLM returns estimated inputs (null if unknown, never guesses).
    Step 2: Python calculates DCF and PEG from those inputs.
    Step 3: HallucinationGuard validates the final numbers.
    """
    news_text = "\n".join(f"- {n['title']}" for n in (news_items or [])[:10]) or "No recent news."
    docs_section = (
        "Uploaded documents:\n" + "\n\n---\n\n".join(
            f"[{name}]:\n{text[:15000]}" for name, text in extra_texts
        ) if extra_texts else
        "No documents - estimate from news and your knowledge of this company."
    )

    base_prompt = INPUTS_PROMPT.format(
        ticker=ticker, market=market,
        news_text=news_text, docs_section=docs_section,
    )

    # Ground with real market data + add anti-hallucination guard
    facts  = _get_market_facts(ticker, market, avg_cost, current_price, currency)
    prompt = prepare_prompt(base_prompt, facts=facts, domain="financial")

    # -- Step 1: LLM estimates inputs ------------------------------------------
    try:
        inputs = complete_json([{"role": "user", "content": prompt}], max_tokens=600)
    except Exception as e:
        return {"error": str(e)}

    if "error" in inputs:
        return inputs

    # -- Step 2: Python calculates valuations (pure math, no hallucination) ----
    result = build_valuation(market, current_price, avg_cost, inputs)
    # Attach LLM narrative
    result["thesis"]          = inputs.get("thesis", "")
    result["risks"]           = inputs.get("risks", [])
    result["opportunities"]   = inputs.get("opportunities", [])
    result["data_confidence"] = inputs.get("data_confidence", "medium")

    # -- Step 3: Validate outputs ----------------------------------------------
    validation = validate_financial_response(result, current_price, avg_cost)
    if not validation.passed:
        result["_validation_errors"]   = validation.errors
        result["_validation_warnings"] = validation.warnings
        print(f"[HallucinationGuard] {ticker}: {validation.errors}")
    elif validation.warnings:
        result["_validation_warnings"] = validation.warnings

    return result


def recompute(market, current_price, avg_cost, inputs: dict, narrative: dict = None) -> dict:
    """
    Recompute valuation from user-completed inputs — NO LLM call.
    Used when the user manually fills inputs the LLM could not estimate.

    inputs: {estimated_eps, estimated_fcf_per_share, estimated_growth_pct,
             wacc_pct, terminal_growth_pct, business_risk}
    narrative: optional {thesis, risks, opportunities} to carry over.
    """
    result = build_valuation(market, current_price, avg_cost, inputs)
    narrative = narrative or {}
    result["thesis"]        = narrative.get("thesis", "")
    result["risks"]         = narrative.get("risks", [])
    result["opportunities"] = narrative.get("opportunities", [])
    result["data_confidence"] = "user-supplied"

    validation = validate_financial_response(result, current_price, avg_cost)
    if not validation.passed:
        result["_validation_errors"]   = validation.errors
        result["_validation_warnings"] = validation.warnings
    elif validation.warnings:
        result["_validation_warnings"] = validation.warnings
    return result
