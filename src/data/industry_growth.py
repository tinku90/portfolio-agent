# -*- coding: utf-8 -*-
"""
Expert consensus sector growth forecasts.
Static baseline (sourced from IMF, RBI, NASSCOM, CRISIL, Gartner, IDC, WSTS etc.)
+ live headline extraction via Groq to keep numbers fresh.
"""
import feedparser, json
from urllib.parse import quote_plus
from src.analysis.llm_client import complete_json

FORECAST_YEAR = "FY2025-26"

# ── Static baseline ───────────────────────────────────────────────────────────
# Format: sector -> (low%, high%, source, notes)
INDIA_BASELINE = {
    "Banking & Financial Services": (12, 15, "RBI / CRISIL",
        "RBI FY26 credit growth projection; NBFC growth ~14%"),
    "Information Technology": (5, 7, "NASSCOM",
        "NASSCOM FY26 revenue growth estimate for Indian IT industry"),
    "Pharmaceuticals & Healthcare": (10, 12, "IQVIA / CRISIL",
        "Domestic formulations + exports recovery; US generics stable"),
    "FMCG & Consumer Goods": (8, 10, "Nielsen / Motilal Oswal",
        "Rural recovery + premiumisation; volume growth ~6-7%"),
    "Automobiles": (6, 9, "SIAM / ICRA",
        "SUV+EV mix upgrade; CV moderation; 2W rural recovery"),
    "Infrastructure & EPC": (15, 20, "NITI Aayog / ICICI Sec",
        "Govt capex Rs 11.1L Cr FY26; Jal Jeevan, roads, railways"),
    "Renewables & Power": (20, 25, "MNRE / JMK Research",
        "500 GW target by 2030; solar + wind + green hydrogen"),
    "Real Estate": (12, 15, "JLL / Anarock",
        "Housing demand strong in top 7 cities; office absorption record"),
    "Telecom": (8, 10, "TRAI / ICRA",
        "ARPU expansion post tariff hike; 5G data monetisation"),
    "Metals & Mining": (5, 8, "World Steel / ICRA",
        "Steel demand ~7% on infra; aluminium/copper tighter supply"),
    "Oil & Gas": (4, 6, "PPAC / Kotak Sec",
        "Refining margins normalising; city gas distribution growing"),
    "Chemicals & Speciality": (10, 14, "Chemexcil / Edelweiss",
        "China+1 tailwind; agrochemicals recovery in H2 FY26"),
    "Capital Goods & Defence": (14, 18, "ICICI Sec / CLSA",
        "Order pipeline strong; defence indigenisation + exports"),
    "Cement": (7, 9, "CMA / Motilal Oswal",
        "Infra + housing demand; overcapacity risk in South India"),
    "Insurance": (12, 15, "IRDAI / Kotak Sec",
        "Health + life underpenetrated; digital distribution boost"),
    "Aviation & Logistics": (10, 14, "IATA / ICRA",
        "Air traffic 15% CAGR; logistics formalisation accelerating"),
    "Agri & Food Processing": (8, 10, "FICCI / NABARD",
        "PLI scheme benefit; exports improving; cold chain investment"),
    "Textiles & Apparel": (6, 9, "AEPC / ICRA",
        "India market share gain post China; PLI traction"),
    "Water & Utilities": (18, 22, "Jal Shakti / NITI Aayog",
        "JJM 2.0; industrial wastewater norms; STP/WTP capex surge"),
    "Education & EdTech": (12, 15, "RedSeer / Redseer",
        "K-12 + upskilling demand; NEP implementation tailwind"),
}

US_BASELINE = {
    "Technology (Software & Cloud)": (12, 15, "Gartner / IDC",
        "IT spending $5.1T globally FY25; cloud still mid-cycle"),
    "Artificial Intelligence": (25, 35, "IDC / McKinsey",
        "GenAI infra + applications; enterprise adoption accelerating"),
    "Semiconductors": (12, 18, "WSTS / Gartner",
        "Memory recovery; AI chips supply-constrained; WSTS CY25 +12.5%"),
    "Healthcare & Biotech": (5, 7, "Deloitte / EY",
        "GLP-1 drugs super-cycle; med-tech devices steady 6%"),
    "Financial Services & Banking": (5, 8, "S&P Global / KBW",
        "NIM compression from rate cuts; fee income resilient"),
    "Energy (Oil & Gas)": (2, 5, "EIA / BP Energy Outlook",
        "Demand plateau; transition risk; US shale production stable"),
    "Renewables & Clean Energy": (18, 25, "BloombergNEF / Wood Mac",
        "IRA tailwind; solar +35% installs; offshore wind ramp"),
    "Consumer Discretionary": (4, 6, "NRF / Deloitte",
        "Services > goods; travel + experiences leading; auto moderate"),
    "Consumer Staples": (3, 5, "Nielsen / Kantar",
        "Volume flat; price-mix +3%; private label pressure"),
    "Industrials & Manufacturing": (4, 7, "ISM / Deloitte",
        "Reshoring + nearshoring; defence strong; logistics normalising"),
    "Defense & Aerospace": (6, 9, "Pentagon / Jefferies",
        "NATO 2% target; US defence budget $886B; space programmes"),
    "Cybersecurity": (12, 16, "Gartner / Forrester",
        "AI-native threats driving spend; identity security fastest growing"),
    "Real Estate (REITs)": (4, 7, "CBRE / JLL",
        "Office absorption recovering; data centre REITs +20%; multifamily stable"),
    "Pharmaceuticals": (4, 6, "IQVIA / Evaluate Pharma",
        "GLP-1 + oncology super-cycles; biosimilar headwinds for innovators"),
    "Electric Vehicles": (15, 22, "BloombergNEF / Wood Mac",
        "China EV price war; US IRA credits; infrastructure buildout"),
    "Data Centers & Infrastructure": (20, 28, "Synergy Research / IDC",
        "AI workload demand; hyperscaler capex +30% YoY"),
    "E-commerce & Retail Tech": (8, 12, "eMarketer / Forrester",
        "AI-personalisation; social commerce; BNPL regulation risk"),
    "Insurance": (5, 7, "Swiss Re / Deloitte",
        "Hard market in P&C; life insurance recovery; InsurTech normalising"),
    "Telecom & 5G": (3, 5, "Ericsson / GSMA",
        "5G mid-band rollout complete; enterprise private networks growing"),
    "Media & Entertainment": (4, 7, "PwC / Deloitte",
        "Streaming profitability phase; live events + sports rights surge"),
}


def _fetch_update_from_news(sector: str, market: str) -> dict:
    """Try to get a fresher estimate from Google News. Returns {} on failure."""
    geo = "IN" if market == "IN" else "US"
    geo_params = "hl=en-IN&gl=IN&ceid=IN:en" if geo == "IN" else "hl=en-US&gl=US&ceid=US:en"
    q = quote_plus(f"{sector} sector growth forecast 2025 2026")
    url = f"https://news.google.com/rss/search?q={q}&{geo_params}"
    feed = feedparser.parse(url)
    headlines = [e.title for e in feed.entries[:5]]
    if not headlines:
        return {}
    prompt = f"""Extract the expert consensus revenue/earnings growth forecast for the {sector} sector in {market} market from these recent headlines. Return a single JSON with keys: growth_low (number), growth_high (number), source (organisation name), headline (most relevant title). If no clear number found return null for growth fields.

Headlines:
{chr(10).join(f'- {h}' for h in headlines)}

Return ONLY valid JSON."""
    try:
        result = complete_json([{"role": "user", "content": prompt}], max_tokens=200)
        if result.get("growth_low"):
            return result
    except Exception:
        pass
    return {}


def build_sector_forecasts(market: str) -> list:
    """
    Returns list of forecast dicts for all 20 sectors in the given market.
    Tries to refresh from news; falls back to static baseline.
    """
    baseline = INDIA_BASELINE if market == "IN" else US_BASELINE
    results = []
    for sector, (low, high, source, notes) in baseline.items():
        entry = {
            "sector": sector, "market": market,
            "growth_low": low, "growth_high": high,
            "source": source, "notes": notes,
            "forecast_year": FORECAST_YEAR,
            "headline": notes,
            "refreshed": False,
        }
        # Try to refresh from live news
        update = _fetch_update_from_news(sector, market)
        if update.get("growth_low") and update.get("growth_high"):
            entry["growth_low"]  = update["growth_low"]
            entry["growth_high"] = update["growth_high"]
            entry["source"]      = update.get("source", source)
            entry["headline"]    = update.get("headline", notes)
            entry["refreshed"]   = True
        results.append(entry)
    return results
