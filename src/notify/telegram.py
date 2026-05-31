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

def alert_opportunity(v):
    msg = f"*OPPORTUNITY: {v['ticker']}* ({v['market']})\n"
    msg += f"Signal: *{v['signal']}*\n"
    msg += f"Price: {v['current_price']:.2f} | FV: {v['fair_value']}\n"
    msg += f"PEG: {v['peg']} | MOS: {(v['mos'] or 0)*100:.0f}%\n"
    send(msg)

def digest(positions, recent_alerts):
    msg = "*WEEKLY DIGEST*\n\n*Portfolio:*\n"
    for p in positions:
        msg += f"- {p['ticker']} ({p['market']}) qty {p['qty']} @ {p['avg_cost']}\n"
    msg += f"\n*Alerts this week:* {len(recent_alerts)}\n"
    for a in recent_alerts[:10]:
        msg += f"- [{a['kind']}] {a['ticker']}\n"
    send(msg)
