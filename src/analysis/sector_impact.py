# -*- coding: utf-8 -*-
from src.analysis.llm_client import complete_json

PROMPT = """A macro/sector news event has occurred. Analyze its impact on the listed stocks.

NEWS HEADLINE: {news_title}

AFFECTED STOCKS:
{stocks_info}

For each stock, assess:
1. Impact level: HIGH / MEDIUM / LOW / NONE
2. One sentence on why
3. Revised near-term sentiment: BULLISH / NEUTRAL / BEARISH
4. Suggested action (one line)

Also write a 2-sentence sector summary.

Return ONLY valid JSON:
{{
  "sector_summary": "...",
  "macro_theme": "...",
  "stocks": [
    {{
      "ticker": "...",
      "impact": "HIGH|MEDIUM|LOW|NONE",
      "reason": "...",
      "sentiment": "BULLISH|NEUTRAL|BEARISH",
      "action": "..."
    }}
  ]
}}"""


def analyze_sector_impact(news_title: str, stocks: list) -> dict:
    """
    stocks: list of dicts with keys ticker, market, sector, industry, thesis
    """
    stocks_info = "\n".join(
        f"- {s['ticker']} ({s['market']}) | Sector: {s.get('sector','?')} | "
        f"Industry: {s.get('industry','?')} | Thesis: {s.get('thesis','')[:80]}"
        for s in stocks
    )
    prompt = PROMPT.format(news_title=news_title, stocks_info=stocks_info)
    try:
        return complete_json([{"role": "user", "content": prompt}], max_tokens=1200)
    except Exception as e:
        return {"error": str(e)}
