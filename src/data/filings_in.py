import feedparser
import re
from urllib.parse import quote_plus

BSE_ANNOUNCEMENTS = "https://www.bseindia.com/data/xml/notices.xml"
# Per-company is the reliable approach: search Google News with "site:bseindia.com" or use IR pages.
# BSE has no per-stock RSS. We use Google News as a proxy for filings on bseindia.com.

def fetch_bse_filings(ticker: str):
    q = quote_plus(f'"{ticker}" site:bseindia.com OR site:nseindia.com announcement')
    url = f"https://news.google.com/rss/search?q={q}&hl=en-IN&gl=IN&ceid=IN:en"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:20]:
        out.append({
            "ticker": ticker,
            "source": "BSE/NSE",
            "title": e.title,
            "url": e.link,
            "published": e.get("published", ""),
        })
    return out

RESULTS_KEYWORDS = re.compile(r"\b(financial result|audited result|quarter|q[1-4]|annual report|earnings|investor present)\b", re.I)

def is_results_filing(title: str) -> bool:
    return bool(RESULTS_KEYWORDS.search(title))
