# -*- coding: utf-8 -*-
import os
from src.analysis.llm_client import complete_json, complete

PROMPT = """You are a financial filings classifier. Decide if this headline about {ticker} is MATERIAL to the investment thesis.

MATERIAL: quarterly/annual results, guidance changes, M&A, rating changes, regulatory action, promoter pledge/sale, large block deals, major contracts, plant commissioning, capex announcements, leadership exits.
NOT MATERIAL: AGM notices, address changes, secretarial filings, routine intimations, share splits without context, generic market commentary.

Headline: {title}
Source: {source}

Return ONLY valid JSON: {{"material": true|false, "reason": "one short sentence", "category": "results|guidance|ma|rating|regulatory|promoter|contract|other"}}"""

SUMMARY_PROMPT = """Summarize this news item about {ticker} in exactly 2 lines for an investor who already holds the stock. Prior thesis: {thesis}

Headline: {title}

Line 1: What happened (facts only).
Line 2: Impact on thesis (positive/negative/neutral and why).

Return only the 2 lines, no preamble."""


def classify(ticker: str, title: str, source: str = "news"):
    try:
        return complete_json([{"role": "user", "content": PROMPT.format(
            ticker=ticker, title=title, source=source)}], max_tokens=200)
    except Exception as e:
        return {"material": False, "reason": f"error: {e}", "category": "other"}


def summarize(ticker: str, title: str, thesis: str):
    try:
        return complete([{"role": "user", "content": SUMMARY_PROMPT.format(
            ticker=ticker, title=title, thesis=thesis)}],
            temperature=0.2, max_tokens=200)
    except Exception as e:
        return f"Summary failed: {e}"
