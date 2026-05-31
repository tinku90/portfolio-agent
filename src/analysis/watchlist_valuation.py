# -*- coding: utf-8 -*-
"""
Watchlist AI Valuation — hybrid architecture (same as ai_valuation):
  LLM estimates INPUTS (growth, WACC, EPS, risk, thesis)
  Python computes all MATH (PEG FV, DCF FV, targets, entry price)

Uses the llm_client fallback chain (Groq -> Gemini -> OpenAI), so it
never dies on a single-provider 429.
"""
from src.analysis.llm_client import complete_json
from src.analysis.valuation import (
    peg_fair_value, peg_target, dcf_fair_value, dcf_target, default_wacc,
)
from src.analysis.token_saver.hallucination_guard import (
    prepare_prompt, validate_financial_response,
)

INPUTS_PROMPT = """You are a buy-side equity analyst evaluating {ticker} ({market}-listed) as a potential BUY.
Estimate ONLY the input figures below. DO NOT calculate fair value yourself.

Recent news:
{news_text}

{docs_section}

Return ONLY valid JSON. Use null for any field you cannot estimate — do NOT guess:
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
                      news_items=None, extra_texts=None) -> dict:
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

    # ── Step 1: LLM estimates inputs (fallback chain handles 429s) ────────────
    try:
        inp = complete_json([{"role": "user", "content": prompt}], max_tokens=600)
    except Exception as e:
        return {"error": str(e)}
    if "error" in inp:
        return inp

    # ── Step 2: Python computes valuations ────────────────────────────────────
    eps    = inp.get("estimated_eps")
    fcf    = inp.get("estimated_fcf_per_share") or eps
    growth = inp.get("estimated_growth_pct")
    wacc   = inp.get("wacc_pct") or default_wacc(market, inp.get("business_risk", "medium"))
    tg     = inp.get("terminal_growth_pct") or 3.0

    peg_fv  = peg_fair_value(eps, growth)
    peg_tgt = peg_target(eps, growth) if peg_fv else None
    dcf_fv  = dcf_fair_value(fcf, growth, wacc, tg) if (fcf and growth) else None
    dcf_tgt = dcf_target(fcf, growth, wacc, tg)      if dcf_fv        else None

    mos_pct = inp.get("suggested_mos_pct") or 0.20
    fvs     = [v for v in (peg_fv, dcf_fv) if v]
    entry   = round(min(fvs) * (1 - mos_pct), 2) if fvs else None

    result = {
        "expected_growth_pct":     growth,
        "peg_fair_value":          peg_fv,
        "peg_target_12m":          peg_tgt,
        "dcf_fair_value":          dcf_fv,
        "dcf_target_12m":          dcf_tgt,
        "suggested_peg_threshold": inp.get("suggested_peg_threshold", 1.0),
        "suggested_mos_pct":       mos_pct,
        "entry_price":             entry,
        "thesis":                  inp.get("thesis", ""),
        "risks":                   inp.get("risks", []),
        "opportunities":           inp.get("opportunities", []),
        "data_confidence":         inp.get("data_confidence", "medium"),
        "_inputs": {"eps": eps, "fcf": fcf, "growth": growth, "wacc": wacc, "tg": tg},
        "valuation_basis": (
            f"PEG: EPS={eps}, g={growth}% -> FV={peg_fv}. "
            f"DCF: FCF={fcf}, g={growth}%, WACC={wacc}%, TG={tg}% -> FV={dcf_fv}."
        ),
    }

    # ── Step 3: validate ──────────────────────────────────────────────────────
    validation = validate_financial_response(result, current_price)
    if not validation.passed:
        result["_validation_errors"] = validation.errors
    if validation.warnings:
        result["_validation_warnings"] = validation.warnings

    return result
