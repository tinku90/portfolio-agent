import os
import requests

BOT = os.environ.get("TELEGRAM_BOT_TOKEN")
CHAT = os.environ.get("TELEGRAM_CHAT_ID")

def send(text: str, parse_mode: str = "Markdown"):
    if not BOT or not CHAT:
        print(f"[TELEGRAM DISABLED] {text}")
        return
    url = f"https://api.telegram.org/bot{BOT}/sendMessage"
    payload = {"chat_id": CHAT, "text": text[:4000], "parse_mode": parse_mode, "disable_web_page_preview": False}
    try:
        r = requests.post(url, json=payload, timeout=10)
        r.raise_for_status()
    except Exception as e:
        print(f"Telegram send failed: {e}")

def alert_price(p):
    msg = f"*Price alert: {p['ticker']}* ({p['market']})\n"
    msg += f"Current: {p['price']:.2f} {p['currency']}\n"
    msg += f"vs prev close: {((p['price']/p['prev_close']-1)*100):+.2f}%\n"
    if p.get("trigger"):
        msg += f"\n*Trigger:* {p['trigger']}"
    send(msg)

def alert_filing(f, summary=None):
    msg = f"*Filing: {f['ticker']}* ({f['source']})\n{f['title']}\n{f['url']}"
    if summary:
        msg += f"\n\n_Impact:_ {summary}"
    send(msg)

def alert_rerating(ticker, position, rerating):
    msg = f"*RERATING: {ticker}*\n"
    msg += f"Verdict: *{rerating.get('verdict')}*\n"
    msg += f"Guidance: {rerating.get('guidance_status')}\n"
    fv_low = rerating.get("new_fair_value_low")
    fv_high = rerating.get("new_fair_value_high")
    msg += f"New FV: {fv_low} - {fv_high} (was {position['fair_value']})\n"
    msg += f"\n_Thesis delta:_ {rerating.get('thesis_delta')}\n"
    msg += f"\n_Action:_ {rerating.get('action')}\n"
    risks = rerating.get("new_risks", [])
    if risks:
        msg += f"\n_New risks:_ " + ", ".join(risks)
    send(msg)

def alert_news_rerate(ticker, market, news_title, r):
    verdict_emoji = {"ACCUMULATE": "🟢", "HOLD": "🟡", "TRIM": "🔴", "EXIT": "⛔"}.get(r.get("verdict", ""), "⚪")
    prev_g, new_g = r.get("prev_growth_pct", 0), r.get("new_growth_pct", 0)
    arrow = "⬆️" if new_g > prev_g else ("⬇️" if new_g < prev_g else "➡️")
    msg  = f"*NEWS RERATE: {ticker}* ({market})\n\n"
    msg += f"📰 _{news_title[:120]}_\n\n"
    bullets = r.get("summary_bullets", [])
    if bullets:
        msg += "*Summary:*\n" + "".join(f"• {b}\n" for b in bullets) + "\n"
    msg += f"*Growth revision:* {prev_g}% → {new_g}% {arrow}\n"
    msg += f"_{r.get('growth_change_reason', '')}_\n\n"
    msg += f"*Updated valuation:*\n"
    msg += f"PEG — FV: `{r.get('new_peg_fair_value')}` | Target: `{r.get('new_peg_target')}`\n"
    msg += f"DCF — FV: `{r.get('new_dcf_fair_value')}` | Target: `{r.get('new_dcf_target')}`\n\n"
    msg += f"*Verdict: {verdict_emoji} {r.get('verdict')}*\n"
    msg += f"_{r.get('verdict_reason', '')}_\n\n"
    msg += f"*Action:* {r.get('action', '')}"
    send(msg)

def alert_watchlist_analysis(ticker, market, current_price, result):
    msg  = f"*WATCHLIST ANALYSIS: {ticker}* ({market})\n\n"
    msg += f"*Current Price:* `{current_price:,.2f}`\n"
    msg += f"*Expected Growth:* `{result.get('expected_growth_pct', 0)}%`\n\n"
    msg += f"*Valuation:*\n"
    msg += f"PEG — FV: `{result.get('peg_fair_value',0):,.0f}` | Target: `{result.get('peg_target_12m',0):,.0f}`\n"
    msg += f"DCF — FV: `{result.get('dcf_fair_value',0):,.0f}` | Target: `{result.get('dcf_target_12m',0):,.0f}`\n"
    msg += f"*Entry price ({int(result.get('suggested_mos_pct',0.2)*100)}% MOS):* `{result.get('entry_price',0):,.0f}`\n"
    msg += f"*Max PEG to pay:* `{result.get('suggested_peg_threshold',1.0)}`\n\n"
    msg += f"_{result.get('thesis', '')}_\n\n"
    if result.get("risks"):
        msg += "*Risks:*\n" + "".join(f"⚠️ {r}\n" for r in result["risks"][:3]) + "\n"
    if result.get("opportunities"):
        msg += "*Opportunities:*\n" + "".join(f"🚀 {o}\n" for o in result["opportunities"][:3])
    send(msg)

def alert_annual_report_analysis(ticker, market, result):
    msg  = f"*ANNUAL REPORT ANALYSIS: {ticker}* ({market})\n\n"
    msg += f"*PEG* — FV: `{result.get('peg_fair_value')}` | Target: `{result.get('peg_target_12m')}`\n"
    msg += f"*DCF* — FV: `{result.get('dcf_fair_value')}` | Target: `{result.get('dcf_target_12m')}`\n"
    msg += f"*Stop Loss:* `{result.get('stop_loss')}`\n\n"
    msg += f"_{result.get('thesis', '')}_\n\n"
    if result.get("risks"):
        msg += "*Risks:*\n" + "".join(f"⚠️ {r}\n" for r in result["risks"][:3]) + "\n"
    if result.get("opportunities"):
        msg += "*Opportunities:*\n" + "".join(f"🚀 {o}\n" for o in result["opportunities"][:3])
    send(msg)

def alert_opportunity(v):
    msg = f"*OPPORTUNITY: {v['ticker']}* ({v['market']})\n"
    msg += f"Signal: *{v['signal']}*\n"
    msg += f"Price: {v['current_price']:.2f} | FV: {v['fair_value']}\n"
    msg += f"PEG: {v['peg']} | MOS: {(v['mos'] or 0)*100:.0f}%\n"
    send(msg)

def digest(positions, recent_alerts):
    msg = "*WEEKLY DIGEST*\n\n*Portfolio:*\n"
    for p in positions:
        peg_fv  = p.get("fair_value", 0)
        dcf_fv  = p.get("dcf_fair_value", 0)
        peg_tgt = p.get("target", 0)
        dcf_tgt = p.get("dcf_target", 0)
        msg += f"\n*{p['ticker']}* ({p['market']}) qty {p['qty']} @ {p['avg_cost']}\n"
        if peg_fv or dcf_fv:
            msg += f"  Fair Value — PEG: {peg_fv or '—'}  |  DCF: {dcf_fv or '—'}\n"
        if peg_tgt or dcf_tgt:
            msg += f"  Target 12m — PEG: {peg_tgt or '—'}  |  DCF: {dcf_tgt or '—'}\n"
    msg += f"\n*Alerts this week:* {len(recent_alerts)}\n"
    for a in recent_alerts[:10]:
        msg += f"- [{a['kind']}] {a['ticker']}\n"
    send(msg)
