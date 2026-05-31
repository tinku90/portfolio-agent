import os, json
from groq import Groq

MODEL = "llama-3.3-70b-versatile"

PROMPT = """You are a buy-side analyst reviewing a material news event for {ticker} ({market}).

CURRENT POSITION:
- Avg buy cost      : {avg_cost}
- PEG fair value    : {peg_fv}
- DCF fair value    : {dcf_fv}
- 12m target        : {target}
- Prior growth est. : {prev_growth}%
- Thesis            : {thesis}

MATERIAL NEWS:
{news_title}

{extra_context}

TASKS:
1. Summarise the news in 3-5 bullet points (facts only, no opinion).
2. Based on this news, what is your revised expected annual earnings growth rate?
3. Recalculate PEG fair value:  EPS_est × (1 + new_g)² × new_g  (Peter Lynch).
4. Recalculate DCF fair value:  5-year DCF, 3% terminal, 12% discount for IN / 10% for US.
5. Rerate: HOLD / ACCUMULATE / TRIM / EXIT with a one-sentence reason.
6. One specific action instruction.

Return ONLY valid JSON:
{{
  "summary_bullets": ["...", "..."],
  "prev_growth_pct": <number>,
  "new_growth_pct": <number>,
  "growth_change_reason": "...",
  "new_peg_fair_value": <number>,
  "new_dcf_fair_value": <number>,
  "new_peg_target": <number>,
  "new_dcf_target": <number>,
  "verdict": "HOLD|ACCUMULATE|TRIM|EXIT",
  "verdict_reason": "...",
  "action": "..."
}}"""


def rerate_on_news(position: dict, news_title: str, extra_context: str = "") -> dict:
    key = os.environ.get("GROQ_API_KEY", "")
    if not key:
        return {"error": "GROQ_API_KEY not set"}
    prompt = PROMPT.format(
        ticker       = position["ticker"],
        market       = position["market"],
        avg_cost     = position["avg_cost"],
        peg_fv       = position.get("fair_value", 0),
        dcf_fv       = position.get("dcf_fair_value", 0),
        target       = position.get("target", 0),
        thesis       = position.get("thesis", ""),
        prev_growth  = position.get("expected_growth", 15),
        news_title   = news_title,
        extra_context= extra_context or "No additional context.",
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
