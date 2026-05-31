# -*- coding: utf-8 -*-
import os
from src.analysis.llm_client import complete_json

PROMPT = """You are a buy-side equity analyst. Analyze {ticker} ({market}-listed) using the data below.

Recent news:
{news_text}

{docs_section}

Current price : {current_price} {currency}
Investor's avg buy cost: {avg_cost}

Use BOTH valuation methods. Do not skip either.

A) PEG-based (Peter Lynch):
   - Fair P/E = expected earnings growth rate (e.g. 20% growth -> P/E 20)
   - PEG fair value = EPS x (1 + g)^2 x fair_P/E
   - PEG 12m target = PEG fair value x 1.10

B) DCF-based (5-year):
   - Use FCF per share (or EPS as proxy if FCF unavailable)
   - Project 5 years at estimated growth rate
   - Terminal value using 3% perpetual growth
   - Discount rate: 12% for IN stocks, 10% for US stocks
   - DCF fair value = PV of projected FCFs + PV of terminal value
   - DCF 12m target = DCF fair value x 1.08

Also:
- Set an absolute stop-loss price (not a %).
- Write a 2-3 sentence investment thesis.
- List 3-5 key risk factors.
- List 3-5 growth opportunities or upcoming catalysts.
- One sentence explaining key assumptions.

Return ONLY valid JSON:
{{
  "peg_fair_value": <number>,
  "peg_target_12m": <number>,
  "dcf_fair_value": <number>,
  "dcf_target_12m": <number>,
  "stop_loss": <number>,
  "thesis": "...",
  "risks": ["...", "..."],
  "opportunities": ["...", "..."],
  "valuation_basis": "..."
}}"""


def analyze_position(ticker, market, avg_cost, current_price, currency,
                     news_items=None, extra_texts=None) -> dict:
    news_text = "\n".join(f"- {n['title']}" for n in (news_items or [])[:10]) or "No recent news."
    docs_section = (
        "Uploaded documents:\n" + "\n\n---\n\n".join(
            f"[{name}]:\n{text[:15000]}" for name, text in extra_texts
        ) if extra_texts else
        "No documents - base analysis on news and general knowledge of this company."
    )
    prompt = PROMPT.format(ticker=ticker, market=market, news_text=news_text,
                           docs_section=docs_section, current_price=current_price,
                           currency=currency, avg_cost=avg_cost)
    try:
        return complete_json([{"role": "user", "content": prompt}], max_tokens=1000)
    except Exception as e:
        return {"error": str(e)}
