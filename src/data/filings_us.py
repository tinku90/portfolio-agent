import feedparser
import re

def fetch_sec_filings(ticker: str):
    # SEC EDGAR per-company atom feed
    url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={ticker}&type=&dateb=&owner=include&count=20&output=atom"
    feed = feedparser.parse(url, request_headers={"User-Agent": "PortfolioAgent contact@example.com"})
    out = []
    for e in feed.entries[:20]:
        out.append({
            "ticker": ticker,
            "source": "SEC",
            "title": e.title,
            "url": e.link,
            "published": e.get("updated", ""),
        })
    return out

US_RESULTS = re.compile(r"\b(10-Q|10-K|8-K|earnings|results)\b", re.I)

def is_results_filing(title: str) -> bool:
    return bool(US_RESULTS.search(title))
