import os
import json
import re
import io
import requests
import pdfplumber
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "openai/gpt-oss-120b"

RERATING_PROMPT = """You are a sell-side analyst rerating a stock after new results.

Prior thesis for {ticker}:
{prior_thesis}

Held at:
- Entry: {avg_cost}
- Current fair value estimate: {fair_value}
- Stop loss: {stop_loss}
- Target: {target}

New filing/results content:
{results_text}

Tasks:
1. Did management meet, beat, or miss the prior guidance?
2. Did any material change to the thesis occur (margins, growth, debt, governance, capex)?
3. Estimate new fair value range using PEG cross-check and simple forward P/E (assume reasonable industry multiples).
4. Decide: HOLD, ACCUMULATE, TRIM, EXIT.
5. Risks that changed.

Output JSON only:
{{
  "verdict": "HOLD|ACCUMULATE|TRIM|EXIT",
  "guidance_status": "BEAT|MEET|MISS|MIXED",
  "new_fair_value_low": <number>,
  "new_fair_value_high": <number>,
  "thesis_delta": "1-2 sentences on what changed",
  "key_numbers": {{"revenue_growth_pct": <number>, "ebitda_margin_pct": <number>, "pat_growth_pct": <number>}},
  "new_risks": ["risk1", "risk2"],
  "action": "1-line specific instruction"
}}"""

def extract_pdf_text(url: str, max_chars: int = 30000) -> str:
    try:
        r = requests.get(url, timeout=30, headers={"User-Agent": "Mozilla/5.0"})
        r.raise_for_status()
        text = ""
        with pdfplumber.open(io.BytesIO(r.content)) as pdf:
            for page in pdf.pages[:20]:
                text += (page.extract_text() or "") + "\n"
                if len(text) > max_chars:
                    break
        return text[:max_chars]
    except Exception as e:
        return f"[PDF extract failed: {e}]"

def rerate(position: dict, results_text: str):
    prompt = RERATING_PROMPT.format(
        ticker=position["ticker"],
        prior_thesis=position["thesis"],
        avg_cost=position["avg_cost"],
        fair_value=position["fair_value"],
        stop_loss=position["stop_loss"],
        target=position["target"],
        results_text=results_text[:25000],
    )
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=1500,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        text = resp.choices[0].message.content if 'resp' in dir() else ""
        m = re.search(r'\{.*\}', text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group())
            except:
                pass
        return {"error": str(e), "raw": text}
