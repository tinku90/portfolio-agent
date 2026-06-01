# -*- coding: utf-8 -*-
"""
Watchlist AI Valuation - hybrid architecture (same as ai_valuation):
  LLM estimates INPUTS (growth, WACC, EPS, risk, thesis)
  Python computes all MATH (PEG FV, DCF FV, targets, entry price)

Uses the llm_client fallback chain (Groq -> Gemini -> OpenAI), so it
never dies on a single-provider 429.
"""
from src.analysis.llm_client import complete_json
from src.analysis.valuation import build_valuation
from src.analysis.token_saver.hallucination_guard import (
    prepare_prompt, validate_financial_response,
)

INPUTS_PROMPT = """You are a buy-side equity analyst evaluating {ticker} ({market}-listed) as a potential BUY.
Estimate ONLY the input figures below. DO NOT calculate fair value yourself.

Recent news:
{news_text}

{docs_section}

Return ONLY valid JSON. Use null for any field you cannot estimate - do NOT guess:
{{
  "estimated_eps":           <number | null>,
  "estimated_fcf_per_share": <number | null>,
  "estimated_growth_pct":    <number | null>,
  "wacc_pct":                <number | null>,
  "terminal_growth_pct":     <number | null>,
  "business_risk":           "low|medium|high",
  "suggested_peg_threshold": <number>,      // max PEG you'd pay for this quality of business
  "suggested_mos_pct":       <number>,      // min margin of safety (0.15-0.40) given the risk
  "thesis":                  "2-3 sentence why-watch thesis",
  "risks":                   ["risk1", "risk2", "risk3"],
  "opportunities":           ["opp1", "opp2", "opp3"],
  "data_confidence":         "high|medium|low"
}}"""


def analyze_watchlist(ticker, market, current_price, currency,
                      news_items=None, extra_texts=None, use_cache=True) -> dict:
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

    # Ground with real market data + anti-hallucination guard
    facts = {f"Current price ({currency})": current_price}
    try:
        from src.data.prices import get_fundamentals
        f = get_fundamentals(ticker, market)
        facts.update({
            "Trailing P/E":  f.get("pe"),
            "Trailing EPS":  f.get("eps"),
            "Forward P/E":   f.get("forward_pe"),
            "Revenue growth (YoY)": f.get("revenue_growth"),
        })
    except Exception:
        pass
    prompt = prepare_prompt(base_prompt, facts=facts, domain="financial")

    # -- Step 1: LLM estimates inputs (fallback chain handles 429s) ------------
    try:
        inp = complete_json([{"role": "user", "content": prompt}], max_tokens=600, use_cache=use_cache)
    except Exception as e:
        return {"error": str(e)}
    if "error" in inp:
        return inp

    # -- Step 2: Python computes valuations (pure math) ------------------------
    result = _assemble_watchlist(market, current_price, inp)

    # -- Step 3: validate ------------------------------------------------------
    validation = validate_financial_response(result, current_price)
    if not validation.passed:
        result["_validation_errors"] = validation.errors
    if validation.warnings:
        result["_validation_warnings"] = validation.warnings

    return result


def _assemble_watchlist(market, current_price, inp: dict) -> dict:
    """Build the watchlist result dict (entry price, thresholds) from inputs."""
    val     = build_valuation(market, current_price, None, inp)
    mos_pct = inp.get("suggested_mos_pct") or 0.20
    fvs     = [v for v in (val["peg_fair_value"], val["dcf_fair_value"]) if v]
    entry   = round(min(fvs) * (1 - mos_pct), 2) if fvs else None
    return {
        "expected_growth_pct":     inp.get("estimated_growth_pct"),
        "peg_fair_value":          val["peg_fair_value"],
        "peg_target_12m":          val["peg_target_12m"],
        "dcf_fair_value":          val["dcf_fair_value"],
        "dcf_target_12m":          val["dcf_target_12m"],
        "suggested_peg_threshold": inp.get("suggested_peg_threshold", 1.0),
        "suggested_mos_pct":       mos_pct,
        "entry_price":             entry,
        "thesis":                  inp.get("thesis", ""),
        "risks":                   inp.get("risks", []),
        "opportunities":           inp.get("opportunities", []),
        "data_confidence":         inp.get("data_confidence", "medium"),
        "missing_inputs":          val["missing_inputs"],
        "_inputs":                 val["_inputs"],
        "valuation_basis":         val["valuation_basis"],
    }


def recompute_watchlist(market, current_price, inputs: dict, narrative: dict = None) -> dict:
    """Recompute watchlist valuation from user-completed inputs — NO LLM call."""
    narrative = narrative or {}
    merged = dict(inputs)
    merged.setdefault("suggested_peg_threshold", narrative.get("suggested_peg_threshold", 1.0))
    merged.setdefault("suggested_mos_pct",       narrative.get("suggested_mos_pct", 0.20))
    merged["thesis"]        = narrative.get("thesis", "")
    merged["risks"]         = narrative.get("risks", [])
    merged["opportunities"] = narrative.get("opportunities", [])
    result = _assemble_watchlist(market, current_price, merged)
    result["data_confidence"] = "user-supplied"
    return result
