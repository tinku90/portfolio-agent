import feedparser
from urllib.parse import quote_plus

def fetch_news(ticker: str, market: str, company_hint: str = ""):
    q_term = f'"{ticker}" {company_hint}' if company_hint else f'"{ticker}"'
    geo = "hl=en-IN&gl=IN&ceid=IN:en" if market == "IN" else "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={quote_plus(q_term)}&{geo}"
    feed = feedparser.parse(url)
    out = []
    for e in feed.entries[:15]:
        out.append({
            "ticker": ticker,
            "title": e.title,
            "url": e.link,
            "published": e.get("published", ""),
        })
    return out
