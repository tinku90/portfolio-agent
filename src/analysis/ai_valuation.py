import os, json
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

PROMPT = """You are a buy-side equity analyst. Analyze {ticker} ({market}-listed) using the data below.

Recent news:
{news_text}

{docs_section}

Current price : {current_price} {currency}
Investor's avg buy cost: {avg_cost}

Use BOTH valuation methods. Do not skip either.

A) PEG-based (Peter Lynch):
   - Fair P/E = expected earnings growth rate (e.g. 20% growth → P/E 20)
   - PEG fair value = EPS × (1 + g)² × fair_P/E
   - PEG 12m target = PEG fair value × 1.10

B) DCF-based (5-year):
   - Use FCF per share (or EPS as proxy if FCF unavailable)
   - Project 5 years at estimated growth rate
   - Terminal value using 3% perpetual growth
   - Discount rate: 12% for IN stocks, 10% for US stocks
   - DCF fair value = PV of projected FCFs + PV of terminal value
   - DCF 12m target = DCF fair value × 1.08

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
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return {"error": "GROQ_API_KEY not set"}

    news_text = "\n".join(f"- {n['title']}" for n in (news_items or [])[:10]) or "No recent news."
    if extra_texts:
        docs_section = "Uploaded documents:\n" + "\n\n---\n\n".join(
            f"[{name}]:\n{text[:15000]}" for name, text in extra_texts
        )
    else:
        docs_section = "No documents — base analysis on news and general knowledge of this company."

    prompt = PROMPT.format(
        ticker=ticker, market=market, news_text=news_text, docs_section=docs_section,
        current_price=current_price, currency=currency, avg_cost=avg_cost,
    )
    try:
        resp = Groq(api_key=key).chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}
