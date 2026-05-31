import yfinance as yf

def yf_symbol(ticker: str, market: str) -> str:
    if market == "IN":
        return f"{ticker}.NS"
    return ticker

def get_price(ticker: str, market: str):
    try:
        t = yf.Ticker(yf_symbol(ticker, market))
        info = t.fast_info
        return {
            "ticker": ticker,
            "market": market,
            "price": float(info.last_price),
            "prev_close": float(info.previous_close),
            "currency": info.currency,
        }
    except Exception as e:
        return {"ticker": ticker, "market": market, "error": str(e)}

def get_fundamentals(ticker: str, market: str):
    """Returns PE, EPS, growth proxy from yfinance. Best for US; for IN use screener.py."""
    try:
        t = yf.Ticker(yf_symbol(ticker, market))
        info = t.info
        return {
            "pe": info.get("trailingPE"),
            "forward_pe": info.get("forwardPE"),
            "eps": info.get("trailingEps"),
            "earnings_growth": info.get("earningsGrowth"),
            "revenue_growth": info.get("revenueGrowth"),
            "market_cap": info.get("marketCap"),
        }
    except Exception:
        return {}
