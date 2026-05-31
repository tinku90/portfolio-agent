import os
import json
import re
from groq import Groq

client = Groq(api_key=os.environ["GROQ_API_KEY"])
MODEL = "llama-3.3-70b-versatile"

PROMPT = """You are a financial filings classifier. Decide if this headline about {ticker} is MATERIAL to the investment thesis.

MATERIAL: quarterly/annual results, guidance changes, M&A, rating changes, regulatory action, promoter pledge/sale, large block deals, major contracts, plant commissioning, capex announcements, leadership exits.
NOT MATERIAL: AGM notices, address changes, secretarial filings, routine intimations, share splits without context, generic market commentary.

Headline: {title}
Source: {source}

Return ONLY valid JSON: {{"material": true|false, "reason": "one short sentence", "category": "results|guidance|ma|rating|regulatory|promoter|contract|other"}}"""

def classify(ticker: str, title: str, source: str = "news"):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": PROMPT.format(ticker=ticker, title=title, source=source)}],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=200,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"material": False, "reason": f"error: {e}", "category": "other"}

SUMMARY_PROMPT = """Summarize this news item about {ticker} in exactly 2 lines for an investor who already holds the stock. Prior thesis: {thesis}

Headline: {title}

Line 1: What happened (facts only).
Line 2: Impact on thesis (positive/negative/neutral and why).

Return only the 2 lines, no preamble."""

def summarize(ticker: str, title: str, thesis: str):
    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[{"role": "user", "content": SUMMARY_PROMPT.format(ticker=ticker, title=title, thesis=thesis)}],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        return f"Summary failed: {e}"
