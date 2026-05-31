# -*- coding: utf-8 -*-
"""
Portfolio Agent REST API
Run: uvicorn api:app --reload --port 8000

Authentication: pass  X-API-Key: <your key>  in every request header.
Set API_KEY in .env to protect your endpoints.
"""
import os, sys, json, yaml
from pathlib import Path
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, HTTPException, Depends, Header, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

from src.storage import db
from src.data import prices, news as news_mod
from src.analysis import ai_valuation
from src.analysis.watchlist_valuation import analyze_watchlist
from src.notify.telegram import send as tg_send

API_KEY = os.environ.get("API_KEY", "change-me-in-env")

app = FastAPI(
    title="Portfolio Agent API",
    description="REST API for the personal portfolio agent — positions, watchlist, AI analysis, alerts.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth ──────────────────────────────────────────────────────────────────────

def require_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


# ── Helpers ───────────────────────────────────────────────────────────────────

PORTFOLIO_FILE = ROOT / "config" / "portfolio.yaml"
WATCHLIST_FILE = ROOT / "config" / "watchlist.yaml"

def _load_portfolio():
    with open(PORTFOLIO_FILE, encoding='utf-8') as f:
        return yaml.safe_load(f).get("positions", [])

def _load_watchlist():
    with open(WATCHLIST_FILE, encoding='utf-8') as f:
        return yaml.safe_load(f).get("candidates", [])

def _save_portfolio(positions):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        yaml.dump({"positions": positions}, f, default_flow_style=False, allow_unicode=True)

def _save_watchlist(watchlist):
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        yaml.dump({"candidates": watchlist}, f, default_flow_style=False, allow_unicode=True)


# ── Pydantic models ───────────────────────────────────────────────────────────

class PositionIn(BaseModel):
    ticker: str
    market: str          # IN or US
    qty: float
    avg_cost: float
    thesis: Optional[str] = "Pending analysis"

class WatchlistIn(BaseModel):
    ticker: str
    market: str
    expected_growth: float = 20.0
    peg_threshold: float  = 1.0
    mos_pct: float        = 0.20
    thesis: Optional[str] = ""

class AnalysisRequest(BaseModel):
    extra_text: Optional[str] = None   # paste raw text for context


# ── Health ────────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ── Portfolio ─────────────────────────────────────────────────────────────────

@app.get("/portfolio", tags=["portfolio"], dependencies=[Depends(require_key)])
def get_portfolio():
    """List all holdings with latest price."""
    positions = _load_portfolio()
    result = []
    for p in positions:
        price = prices.get_price(p["ticker"], p["market"])
        result.append({**p, "current_price": price.get("price"), "currency": price.get("currency")})
    return result


@app.post("/portfolio", tags=["portfolio"], dependencies=[Depends(require_key)])
def add_position(body: PositionIn):
    """Add a new holding."""
    positions = _load_portfolio()
    ticker = body.ticker.upper()
    if any(p["ticker"] == ticker for p in positions):
        raise HTTPException(400, f"{ticker} already in portfolio")
    positions.append({
        "ticker": ticker, "market": body.market.upper(),
        "qty": body.qty, "avg_cost": body.avg_cost,
        "entry_date": datetime.today().strftime("%Y-%m-%d"),
        "thesis": body.thesis or "Pending analysis",
        "fair_value": 0, "dcf_fair_value": 0, "stop_loss": 0, "target": 0, "dcf_target": 0,
    })
    _save_portfolio(positions)
    db.init()
    db.sync_yaml(positions, _load_watchlist())
    return {"added": ticker}


@app.delete("/portfolio/{ticker}", tags=["portfolio"], dependencies=[Depends(require_key)])
def remove_position(ticker: str):
    """Remove a holding."""
    ticker = ticker.upper()
    positions = _load_portfolio()
    updated = [p for p in positions if p["ticker"] != ticker]
    if len(updated) == len(positions):
        raise HTTPException(404, f"{ticker} not found")
    _save_portfolio(updated)
    db.sync_yaml(updated, _load_watchlist())
    return {"removed": ticker}


# ── Watchlist ─────────────────────────────────────────────────────────────────

@app.get("/watchlist", tags=["watchlist"], dependencies=[Depends(require_key)])
def get_watchlist():
    return _load_watchlist()


@app.post("/watchlist", tags=["watchlist"], dependencies=[Depends(require_key)])
def add_watchlist(body: WatchlistIn):
    """Add to watchlist."""
    watchlist = _load_watchlist()
    ticker = body.ticker.upper()
    if any(w["ticker"] == ticker for w in watchlist):
        raise HTTPException(400, f"{ticker} already in watchlist")
    watchlist.append({
        "ticker": ticker, "market": body.market.upper(),
        "expected_growth": body.expected_growth, "peg_threshold": body.peg_threshold,
        "mos_pct": body.mos_pct, "thesis": body.thesis,
    })
    _save_watchlist(watchlist)
    db.sync_yaml(_load_portfolio(), watchlist)
    return {"added": ticker}


@app.delete("/watchlist/{ticker}", tags=["watchlist"], dependencies=[Depends(require_key)])
def remove_watchlist(ticker: str):
    ticker = ticker.upper()
    watchlist = _load_watchlist()
    updated = [w for w in watchlist if w["ticker"] != ticker]
    if len(updated) == len(watchlist):
        raise HTTPException(404, f"{ticker} not found")
    _save_watchlist(updated)
    return {"removed": ticker}


# ── AI Analysis ───────────────────────────────────────────────────────────────

@app.post("/analysis/position/{ticker}", tags=["analysis"], dependencies=[Depends(require_key)])
def analyze_position(ticker: str, body: AnalysisRequest, background_tasks: BackgroundTasks):
    """
    Run DCF + PEG AI analysis for a held position.
    Returns immediately; sends result to Telegram in background.
    """
    ticker = ticker.upper()
    positions = _load_portfolio()
    pos = next((p for p in positions if p["ticker"] == ticker), None)
    if not pos:
        raise HTTPException(404, f"{ticker} not in portfolio")

    price_data    = prices.get_price(ticker, pos["market"])
    current_price = price_data.get("price", pos["avg_cost"]) if "error" not in price_data else pos["avg_cost"]
    currency      = price_data.get("currency", "INR" if pos["market"] == "IN" else "USD")

    try:
        news_items = news_mod.fetch_news(ticker, pos["market"], "")
    except Exception:
        news_items = []

    extra = [("user_context", body.extra_text)] if body.extra_text else None
    result = ai_valuation.analyze_position(ticker, pos["market"], pos["avg_cost"],
                                           current_price, currency, news_items, extra)
    if "error" in result:
        raise HTTPException(500, result["error"])

    # Persist to YAML
    for p in positions:
        if p["ticker"] == ticker:
            p.update(
                fair_value=result.get("peg_fair_value", 0),
                target=result.get("peg_target_12m", 0),
                dcf_fair_value=result.get("dcf_fair_value", 0),
                dcf_target=result.get("dcf_target_12m", 0),
                stop_loss=result.get("stop_loss", 0),
                thesis=result.get("thesis", p.get("thesis", "")),
                risks=result.get("risks", []),
                opportunities=result.get("opportunities", []),
            )
    _save_portfolio(positions)

    # Send Telegram in background
    def _notify():
        msg  = f"*API Analysis: {ticker}*\n"
        msg += f"PEG FV: `{result.get('peg_fair_value')}` | Target: `{result.get('peg_target_12m')}`\n"
        msg += f"DCF FV: `{result.get('dcf_fair_value')}` | Target: `{result.get('dcf_target_12m')}`\n"
        msg += f"_{result.get('thesis', '')}_"
        tg_send(msg)
    background_tasks.add_task(_notify)

    return result


@app.post("/analysis/watchlist/{ticker}", tags=["analysis"], dependencies=[Depends(require_key)])
def analyze_watchlist_item(ticker: str, body: AnalysisRequest):
    """Run AI analysis for a watchlist candidate."""
    ticker = ticker.upper()
    watchlist = _load_watchlist()
    item = next((w for w in watchlist if w["ticker"] == ticker), None)
    if not item:
        raise HTTPException(404, f"{ticker} not in watchlist")

    price_data    = prices.get_price(ticker, item["market"])
    current_price = price_data.get("price", 0) if "error" not in price_data else 0
    currency      = price_data.get("currency", "INR" if item["market"] == "IN" else "USD")

    extra = [("user_context", body.extra_text)] if body.extra_text else None
    result = analyze_watchlist(ticker, item["market"], current_price, currency, extra_texts=extra)
    if "error" in result:
        raise HTTPException(500, result["error"])
    return result


# ── News ──────────────────────────────────────────────────────────────────────

@app.get("/news/{ticker}", tags=["news"], dependencies=[Depends(require_key)])
def get_news(ticker: str, market: str = "US", limit: int = 10):
    """Fetch latest news for any ticker."""
    items = news_mod.fetch_news(ticker.upper(), market.upper(), "")
    return items[:limit]


# ── Alerts ────────────────────────────────────────────────────────────────────

@app.get("/alerts", tags=["alerts"], dependencies=[Depends(require_key)])
def get_alerts(days: int = 7, limit: int = 50):
    """Get recent alerts from the DB."""
    db.init()
    with db.conn() as c:
        rows = [dict(r) for r in c.execute(
            f"SELECT * FROM alerts WHERE sent_at > datetime('now','-{days} days') "
            f"ORDER BY sent_at DESC LIMIT {limit}"
        )]
    return rows


# ── Sector forecasts ──────────────────────────────────────────────────────────

@app.get("/sector-forecasts", tags=["sectors"], dependencies=[Depends(require_key)])
def get_sector_forecasts(market: Optional[str] = None):
    """Get saved expert sector growth forecasts."""
    db.init()
    return db.get_sector_forecasts(market.upper() if market else None)


# ── Jobs ──────────────────────────────────────────────────────────────────────

@app.post("/jobs/{job}", tags=["jobs"], dependencies=[Depends(require_key)])
def trigger_job(job: str, background_tasks: BackgroundTasks):
    """
    Trigger a job in the background.
    job: bootstrap | hourly | daily | weekly | sector-forecasts
    """
    from src.main import bootstrap, job_hourly, job_daily, job_weekly, run_sector_forecast_digest
    jobs = {
        "bootstrap":        bootstrap,
        "hourly":           job_hourly,
        "daily":            job_daily,
        "weekly":           job_weekly,
        "sector-forecasts": run_sector_forecast_digest,
    }
    if job not in jobs:
        raise HTTPException(400, f"Unknown job '{job}'. Valid: {list(jobs)}")
    background_tasks.add_task(jobs[job])
    return {"triggered": job, "status": "running in background"}
