# -*- coding: utf-8 -*-
"""
AI Valuation — hybrid architecture:
  LLM estimates forward-looking INPUTS  (growth, WACC, EPS estimate, thesis, risks)
  Python computes all MATH              (DCF fair value, PEG fair value, targets, MOS)

This eliminates hallucinated calculations entirely.
If LLM cannot estimate an input, it returns null → Python returns null for that method.
"""
from src.analysis.llm_client import complete_json
from src.analysis.valuation import (
    peg_fair_value, peg_target, dcf_fair_value, dcf_target,
    margin_of_safety, suggested_stop_loss, default_wacc,
)
from src.analysis.token_saver.hallucination_guard import (
    prepare_prompt, validate_financial_response,
)

# ── LLM prompt: ask ONLY for inputs, not calculations ────────────────────────
INPUTS_PROMPT = """You are a buy-side equity analyst. Estimate the INPUTS needed to value {ticker} ({market}-listed).
DO NOT calculate fair value yourself — provide only the input figures listed below.

Recent news:
{news_text}

{docs_section}

Return ONLY valid JSON. Use null for any field you cannot estimate from available data — do NOT guess:
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
        "No documents — estimate from news and your knowledge of this company."
    )

    base_prompt = INPUTS_PROMPT.format(
        ticker=ticker, market=market,
        news_text=news_text, docs_section=docs_section,
    )

    # Ground with real market data + add anti-hallucination guard
    facts  = _get_market_facts(ticker, market, avg_cost, current_price, currency)
    prompt = prepare_prompt(base_prompt, facts=facts, domain="financial")

    # ── Step 1: LLM estimates inputs ──────────────────────────────────────────
    try:
        inputs = complete_json([{"role": "user", "content": prompt}], max_tokens=600)
    except Exception as e:
        return {"error": str(e)}

    if "error" in inputs:
        return inputs

    # ── Step 2: Python calculates valuations ──────────────────────────────────
    eps       = inputs.get("estimated_eps")
    fcf       = inputs.get("estimated_fcf_per_share") or eps   # FCF preferred; fallback to EPS
    growth    = inputs.get("estimated_growth_pct")
    wacc      = inputs.get("wacc_pct") or default_wacc(market, inputs.get("business_risk", "medium"))
    tg        = inputs.get("terminal_growth_pct") or 3.0
    risk      = inputs.get("business_risk", "medium")

    # PEG calculation (needs EPS + growth)
    peg_fv  = peg_fair_value(eps, growth)
    peg_tgt = peg_target(eps, growth) if peg_fv else None

    # DCF calculation (needs FCF/EPS + growth + WACC)
    dcf_fv  = dcf_fair_value(fcf, growth, wacc, tg) if (fcf and growth) else None
    dcf_tgt = dcf_target(fcf, growth, wacc, tg)      if dcf_fv        else None

    # Stop loss from risk level (never hallucinated — pure formula)
    sl = suggested_stop_loss(current_price, risk)

    # MOS
    best_fv = peg_fv or dcf_fv
    mos     = margin_of_safety(current_price, best_fv) if best_fv else None

    result = {
        # Calculated values (Python, no hallucination)
        "peg_fair_value":  peg_fv,
        "peg_target_12m":  peg_tgt,
        "dcf_fair_value":  dcf_fv,
        "dcf_target_12m":  dcf_tgt,
        "stop_loss":       sl,
        "mos":             mos,
        # LLM narrative (what it's good at)
        "thesis":          inputs.get("thesis", ""),
        "risks":           inputs.get("risks", []),
        "opportunities":   inputs.get("opportunities", []),
        "data_confidence": inputs.get("data_confidence", "medium"),
        # Inputs used (full transparency)
        "_inputs": {
            "eps":    eps,    "fcf":    fcf,
            "growth": growth, "wacc":   wacc,
            "tg":     tg,     "risk":   risk,
        },
        "valuation_basis": (
            f"PEG: EPS={eps}, growth={growth}% → FV={peg_fv}. "
            f"DCF: FCF={fcf}, growth={growth}%, WACC={wacc}%, TG={tg}% → FV={dcf_fv}. "
            f"Confidence: {inputs.get('data_confidence','medium')}."
        ),
    }

    # ── Step 3: Validate outputs ──────────────────────────────────────────────
    validation = validate_financial_response(result, current_price, avg_cost)
    if not validation.passed:
        result["_validation_errors"]   = validation.errors
        result["_validation_warnings"] = validation.warnings
        print(f"[HallucinationGuard] {ticker}: {validation.errors}")
    elif validation.warnings:
        result["_validation_warnings"] = validation.warnings

    return result
