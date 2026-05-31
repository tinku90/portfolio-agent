# -*- coding: utf-8 -*-
"""
Compare management / AI-estimated growth for each held stock
against the expert consensus for its sector.
"""

# Map yfinance sector/industry names -> our forecast sector keys
SECTOR_MAP = {
    # US / generic
    "Technology":                        "Technology (Software & Cloud)",
    "Semiconductors":                    "Semiconductors",
    "Software - Application":            "Technology (Software & Cloud)",
    "Software - Infrastructure":         "Technology (Software & Cloud)",
    "Internet Content & Information":    "Technology (Software & Cloud)",
    "Communication Services":            "Telecom & 5G",
    "Healthcare":                        "Healthcare & Biotech",
    "Biotech":                           "Healthcare & Biotech",
    "Financials":                        "Financial Services & Banking",
    "Consumer Discretionary":            "Consumer Discretionary",
    "Consumer Staples":                  "Consumer Staples",
    "Energy":                            "Energy (Oil & Gas)",
    "Oil & Gas E&P":                     "Energy (Oil & Gas)",
    "Industrials":                       "Industrials & Manufacturing",
    "Engineering & Construction":        "Infrastructure & EPC",
    "Real Estate":                       "Real Estate (REITs)",
    "Materials":                         "Metals & Mining",
    "Utilities":                         "Renewables & Clean Energy",
    # India specific
    "Infrastructure & EPC":              "Infrastructure & EPC",
    "Pharmaceuticals":                   "Pharmaceuticals & Healthcare",
    "Banking":                           "Banking & Financial Services",
    "NBFC":                              "Banking & Financial Services",
    "Automobiles":                       "Automobiles",
    "FMCG":                              "FMCG & Consumer Goods",
    "Chemicals":                         "Chemicals & Speciality",
    "Cement":                            "Cement",
    "Metals":                            "Metals & Mining",
    "Telecom":                           "Telecom",
    "Power":                             "Renewables & Power",
    "Water":                             "Water & Utilities",
    "Defence":                           "Capital Goods & Defence",
}


def map_sector(stock: dict) -> str:
    """Return the best matching forecast sector key for a stock."""
    industry = stock.get("industry", "")
    sector   = stock.get("sector", "")
    # Try industry first (more specific), then sector
    return (SECTOR_MAP.get(industry)
            or SECTOR_MAP.get(sector)
            or sector
            or "Unknown")


def compare(stock: dict, forecast: dict) -> dict:
    """
    stock    : position or watchlist dict (has expected_growth, sector, ticker)
    forecast : sector_forecast dict (has growth_low, growth_high)
    """
    mgmt_g  = stock.get("expected_growth") or stock.get("fair_value") and None
    exp_low  = forecast.get("growth_low", 0)
    exp_high = forecast.get("growth_high", 0)
    exp_mid  = (exp_low + exp_high) / 2

    if not mgmt_g:
        return {"status": "no_mgmt_estimate"}

    diff = mgmt_g - exp_mid
    if diff > 5:
        verdict = "PREMIUM"
        emoji   = "🚀"
        note    = f"Mgmt {mgmt_g}% beats expert mid {exp_mid:.0f}% by {diff:.0f}pp — above-consensus grower"
    elif diff < -5:
        verdict = "DISCOUNT"
        emoji   = "⚠️"
        note    = f"Mgmt {mgmt_g}% trails expert mid {exp_mid:.0f}% by {abs(diff):.0f}pp — watch guidance"
    else:
        verdict = "IN-LINE"
        emoji   = "✅"
        note    = f"Mgmt {mgmt_g}% in line with expert range {exp_low}-{exp_high}%"

    return {
        "ticker":    stock["ticker"],
        "sector":    stock.get("sector", "Unknown"),
        "mgmt_growth":   mgmt_g,
        "expert_low":    exp_low,
        "expert_high":   exp_high,
        "expert_mid":    exp_mid,
        "diff_pp":       round(diff, 1),
        "verdict":       verdict,
        "emoji":         emoji,
        "note":          note,
        "source":        forecast.get("source", ""),
    }
