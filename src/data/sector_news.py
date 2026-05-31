# -*- coding: utf-8 -*-
import feedparser
from urllib.parse import quote_plus

MACRO_QUERIES = [
    "RBI repo rate decision",
    "Federal Reserve interest rate",
    "India CPI inflation",
    "US CPI inflation",
    "India GDP growth",
    "US GDP growth",
    "crude oil price",
    "rupee dollar exchange rate",
    "FII FPI flows India",
    "US recession",
]

SECTOR_QUERIES = {
    "Technology":    ["India IT sector outlook", "US technology sector earnings"],
    "Financials":    ["India banking sector NPA", "US banking earnings"],
    "Energy":        ["oil gas sector India", "crude oil demand supply"],
    "Healthcare":    ["India pharma USFDA", "US pharma sector"],
    "Utilities":     ["India water infrastructure EPC", "India power sector"],
    "Industrials":   ["India capex infrastructure", "US industrial output"],
    "ConsumerGoods": ["India FMCG rural demand", "US consumer spending"],
    "RealEstate":    ["India real estate demand", "US housing market"],
    "Semiconductors":["semiconductor chip demand", "TSMC NVIDIA earnings"],
}


def _fetch_google_news(query: str, geo: str = "IN", limit: int = 5) -> list:
    geo_params = "hl=en-IN&gl=IN&ceid=IN:en" if geo == "IN" else "hl=en-US&gl=US&ceid=US:en"
    url = f"https://news.google.com/rss/search?q={quote_plus(query)}&{geo_params}"
    feed = feedparser.parse(url)
    return [
        {"title": e.title, "url": e.link, "published": e.get("published", ""), "query": query}
        for e in feed.entries[:limit]
    ]


def fetch_macro_news() -> list:
    items = []
    for q in MACRO_QUERIES:
        geo = "IN" if any(k in q.lower() for k in ["rbi", "india", "rupee", "fii"]) else "US"
        items.extend(_fetch_google_news(q, geo=geo, limit=3))
    return items


def fetch_sector_news(sectors: list) -> list:
    """Fetch news for a list of sector names."""
    items = []
    for sector in sectors:
        queries = SECTOR_QUERIES.get(sector, [f"{sector} sector outlook"])
        for q in queries[:1]:
            items.extend(_fetch_google_news(q, limit=3))
    return items
