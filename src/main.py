import sys
import yaml
import json
from pathlib import Path
from datetime import datetime, timedelta

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.storage import db
from src.data import prices, filings_in, filings_us, news
from src.analysis import valuation, materiality, rerating, ai_valuation
from src.notify import telegram

ROOT = Path(__file__).parent.parent

def load_config():
    with open(ROOT / "config" / "portfolio.yaml") as f:
        pf = yaml.safe_load(f)
    with open(ROOT / "config" / "watchlist.yaml") as f:
        wl = yaml.safe_load(f)
    return pf["positions"], wl["candidates"]

def bootstrap():
    db.init()
    positions, watchlist = load_config()
    db.sync_yaml(positions, watchlist)
    print(f"Loaded {len(positions)} positions, {len(watchlist)} watchlist")

def job_hourly():
    """Prices + filings + news scan every 15 min during market hours."""
    bootstrap()
    all_tickers = db.all_positions() + db.all_watchlist()

    # 1. Price triggers for positions
    for pos in db.all_positions():
        p = prices.get_price(pos["ticker"], pos["market"])
        if "error" in p:
            continue
        trigger = None
        if pos.get("stop_loss", 0) > 0 and p["price"] <= pos["stop_loss"]:
            trigger = f"Stop loss hit ({pos['stop_loss']})"
        elif pos.get("target", 0) > 0 and p["price"] >= pos["target"]:
            trigger = f"Target reached ({pos['target']})"
        elif abs(p["price"]/p["prev_close"] - 1) > 0.05:
            trigger = f"Moved {(p['price']/p['prev_close']-1)*100:+.1f}% intraday"
        if trigger:
            p["trigger"] = trigger
            telegram.alert_price(p)
            db.log_alert(pos["ticker"], "price", json.dumps(p))

    # 2. Filing scan
    for stock in all_tickers:
        try:
            if stock["market"] == "IN":
                fs = filings_in.fetch_bse_filings(stock["ticker"])
            else:
                fs = filings_us.fetch_sec_filings(stock["ticker"])
        except Exception as e:
            print(f"Filing fetch failed {stock['ticker']}: {e}")
            continue

        for f in fs:
            fid = db.save_filing(f["ticker"], f["source"], f["title"], f["url"], f["published"])
            if not fid:
                continue
            # Is it a results filing? Trigger rerating
            is_result = (filings_in.is_results_filing if stock["market"] == "IN" else filings_us.is_results_filing)(f["title"])
            position = next((p for p in db.all_positions() if p["ticker"] == stock["ticker"]), None)
            if is_result and position:
                text = rerating.extract_pdf_text(f["url"]) if f["url"].endswith(".pdf") else f["title"]
                rr = rerating.rerate(position, text)
                if "error" not in rr:
                    telegram.alert_rerating(stock["ticker"], position, rr)
                    db.log_alert(stock["ticker"], "rerating", json.dumps(rr))
                else:
                    telegram.alert_filing(f, summary="Results filing detected — manual review needed")
            else:
                telegram.alert_filing(f)
            db.mark_processed(fid)

    # 3. News scan (deduplicated — Groq only called on headlines not seen before)
    for stock in all_tickers:
        try:
            items = news.fetch_news(stock["ticker"], stock["market"], stock.get("thesis", "")[:50])
        except Exception as e:
            print(f"News fetch failed {stock['ticker']}: {e}")
            continue
        for n in items:
            nid = db.save_news(stock["ticker"], n["title"], n["url"], n["published"])
            if not nid:
                continue  # already seen — skip Groq call
            cls = materiality.classify(stock["ticker"], n["title"])
            summary = None
            if cls["material"]:
                summary = materiality.summarize(stock["ticker"], n["title"], stock.get("thesis", ""))
                telegram.alert_filing({"ticker": stock["ticker"], "source": "News", "title": n["title"], "url": n["url"]}, summary=summary)
                db.log_alert(stock["ticker"], "news_material", json.dumps({"title": n["title"], "summary": summary}))
            db.update_news_materiality(nid, cls["material"], summary)

def _update_yaml_valuation(ticker, result):
    """Write AI analysis results back to portfolio.yaml."""
    pf_file = ROOT / "config" / "portfolio.yaml"
    with open(pf_file) as f:
        pf = yaml.safe_load(f)
    for p in pf["positions"]:
        if p["ticker"] == ticker:
            p.update(
                fair_value     = result.get("peg_fair_value", 0),
                target         = result.get("peg_target_12m", 0),
                dcf_fair_value = result.get("dcf_fair_value", 0),
                dcf_target     = result.get("dcf_target_12m", 0),
                stop_loss      = result.get("stop_loss", p.get("stop_loss", 0)),
                thesis         = result.get("thesis", p.get("thesis", "")),
                risks          = result.get("risks", []),
                opportunities  = result.get("opportunities", []),
            )
            break
    with open(pf_file, "w") as f:
        yaml.dump(pf, f, default_flow_style=False, allow_unicode=True)

def auto_analyze_new_positions():
    """Run AI valuation for any position where fair_value is still 0."""
    positions, _ = load_config()
    unanalyzed = [p for p in positions if not p.get("fair_value")]
    if not unanalyzed:
        return
    print(f"Auto-analyzing {len(unanalyzed)} unanalyzed position(s)…")
    for pos in unanalyzed:
        ticker, market = pos["ticker"], pos["market"]
        try:
            news_items = news.fetch_news(ticker, market, pos.get("thesis", "")[:50])
        except Exception:
            news_items = []
        price_data    = prices.get_price(ticker, market)
        current_price = price_data.get("price", pos["avg_cost"]) if "error" not in price_data else pos["avg_cost"]
        currency      = price_data.get("currency", "INR" if market == "IN" else "USD")
        result = ai_valuation.analyze_position(
            ticker, market, pos["avg_cost"], current_price, currency, news_items
        )
        if "error" in result:
            print(f"  {ticker} analysis failed: {result['error']}")
            continue
        _update_yaml_valuation(ticker, result)
        msg  = f"*AI Valuation: {ticker}* ({market})\n\n"
        msg += f"*PEG* — Fair Value: `{result.get('peg_fair_value')}` | Target: `{result.get('peg_target_12m')}`\n"
        msg += f"*DCF* — Fair Value: `{result.get('dcf_fair_value')}` | Target: `{result.get('dcf_target_12m')}`\n"
        msg += f"*Stop Loss:* `{result.get('stop_loss')}`\n\n"
        msg += f"_{result.get('thesis', '')}_\n\n"
        if result.get("risks"):
            msg += "*Risks:* " + " · ".join(f"⚠️ {r}" for r in result["risks"][:3]) + "\n"
        if result.get("opportunities"):
            msg += "*Opportunities:* " + " · ".join(f"🚀 {o}" for o in result["opportunities"][:3])
        telegram.send(msg)
        db.log_alert(ticker, "ai_valuation", json.dumps(result))
        print(f"  {ticker}: PEG FV={result.get('peg_fair_value')} DCF FV={result.get('dcf_fair_value')}")

def job_daily():
    """News scan + opportunity scan + auto-valuation for new positions."""
    bootstrap()

    # Auto-analyze any position added without a valuation
    auto_analyze_new_positions()

    # News
    for stock in db.all_positions() + db.all_watchlist():
        try:
            items = news.fetch_news(stock["ticker"], stock["market"], stock.get("thesis", "")[:50])
        except Exception as e:
            print(f"News fetch failed {stock['ticker']}: {e}")
            continue
        for n in items:
            nid = db.save_news(stock["ticker"], n["title"], n["url"], n["published"])
            if not nid:
                continue
            cls = materiality.classify(stock["ticker"], n["title"])
            summary = None
            if cls["material"]:
                summary = materiality.summarize(stock["ticker"], n["title"], stock["thesis"])
                telegram.alert_filing({"ticker": stock["ticker"], "source": "News", "title": n["title"], "url": n["url"]}, summary=summary)
                db.log_alert(stock["ticker"], "news_material", json.dumps({"title": n["title"], "summary": summary}))
            db.update_news_materiality(nid, cls["material"], summary)

    # Opportunity scan on watchlist
    for w in db.all_watchlist():
        f = prices.get_fundamentals(w["ticker"], w["market"])
        p = prices.get_price(w["ticker"], w["market"])
        if "error" in p or not f.get("pe") or not f.get("eps"):
            continue
        v = valuation.assess(
            ticker=w["ticker"], market=w["market"],
            pe=f["pe"], eps=f["eps"],
            growth_pct=w["expected_growth"],
            current_price=p["price"],
            peg_threshold=w["peg_threshold"],
            mos_pct=w["mos_pct"],
        )
        db.save_valuation(w["ticker"], v["pe"], v["growth_pct"], v["peg"], v["fair_value"], v["current_price"], v["mos"])
        if v["signal"] in ("BUY", "ACCUMULATE"):
            telegram.alert_opportunity(v)
            db.log_alert(w["ticker"], f"opportunity_{v['signal']}", json.dumps(v))

def job_weekly():
    """Sunday digest."""
    bootstrap()
    with db.conn() as c:
        recent = [dict(r) for r in c.execute(
            "SELECT * FROM alerts WHERE sent_at > datetime('now','-7 days') ORDER BY sent_at DESC"
        )]
    telegram.digest(db.all_positions(), recent)

if __name__ == "__main__":
    job = sys.argv[1] if len(sys.argv) > 1 else "hourly"
    {"hourly": job_hourly, "daily": job_daily, "weekly": job_weekly, "bootstrap": bootstrap}[job]()
