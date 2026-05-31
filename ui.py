import streamlit as st
import yaml
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent
PORTFOLIO_FILE = ROOT / "config" / "portfolio.yaml"
WATCHLIST_FILE = ROOT / "config" / "watchlist.yaml"

st.set_page_config(page_title="Portfolio Agent", page_icon="📈", layout="wide")

# ── helpers ──────────────────────────────────────────────────────────────────

def load():
    with open(PORTFOLIO_FILE) as f:
        pf = yaml.safe_load(f) or {}
    with open(WATCHLIST_FILE) as f:
        wl = yaml.safe_load(f) or {}
    return pf.get("positions") or [], wl.get("candidates") or []

def save(positions, watchlist):
    with open(PORTFOLIO_FILE, "w") as f:
        yaml.dump({"positions": positions}, f, default_flow_style=False, allow_unicode=True)
    with open(WATCHLIST_FILE, "w") as f:
        yaml.dump({"candidates": watchlist}, f, default_flow_style=False, allow_unicode=True)

# ── page ─────────────────────────────────────────────────────────────────────

st.title("📈 Portfolio Agent")
st.caption("Manage your holdings and watchlist. Changes are saved to config YAMLs immediately.")

positions, watchlist = load()

tab1, tab2 = st.tabs(["💼  My Portfolio", "👁️  Watchlist"])

# ═══════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Holdings  ·  {len(positions)} stocks")

    if positions:
        for i, p in enumerate(positions):
            col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1, 1, 1.2, 1.2, 1.2, 1.2, 0.8])
            col1.markdown(f"**{p['ticker']}** `{p['market']}`")
            col2.metric("Qty", int(p["qty"]))
            col3.metric("Avg Cost", f"{p['avg_cost']:,.0f}")
            col4.metric("Fair Value", f"{p['fair_value']:,.0f}")
            col5.metric("Stop Loss", f"{p['stop_loss']:,.0f}")
            col6.metric("Target", f"{p['target']:,.0f}")
            col7.caption(p.get("thesis", "")[:60] + ("…" if len(p.get("thesis","")) > 60 else ""))
            if col8.button("🗑️ Remove", key=f"del_pos_{i}"):
                positions = [x for x in positions if x["ticker"] != p["ticker"]]
                save(positions, watchlist)
                st.success(f"Removed {p['ticker']}")
                st.rerun()
            st.divider()
    else:
        st.info("No holdings yet. Add your first stock below.")

    st.subheader("➕ Add a Position")
    with st.form("add_position", clear_on_submit=True):
        c1, c2, c3 = st.columns(3)
        ticker   = c1.text_input("Ticker *", placeholder="e.g. INFY or NVDA").upper().strip()
        market   = c2.selectbox("Market *", ["IN", "US"])
        qty      = c3.number_input("Quantity *", min_value=0.01, step=1.0)

        c4, c5, c6 = st.columns(3)
        avg_cost   = c4.number_input("Avg Buy Price *", min_value=0.01, step=0.5)
        fair_value = c5.number_input("Your Fair Value *", min_value=0.01, step=0.5)
        stop_loss  = c6.number_input("Stop Loss *", min_value=0.01, step=0.5)

        c7, c8 = st.columns([1, 2])
        target = c7.number_input("Target Price *", min_value=0.01, step=0.5)
        thesis = c8.text_area("Investment Thesis *", placeholder="Why did you buy this?", height=80)

        submitted = st.form_submit_button("Add to Portfolio", type="primary")
        if submitted:
            if not ticker or not thesis:
                st.error("Ticker and thesis are required.")
            elif any(p["ticker"] == ticker for p in positions):
                st.error(f"{ticker} is already in your portfolio.")
            else:
                positions.append({
                    "ticker": ticker, "market": market, "qty": float(qty),
                    "avg_cost": float(avg_cost), "entry_date": str(date.today()),
                    "thesis": thesis, "fair_value": float(fair_value),
                    "stop_loss": float(stop_loss), "target": float(target),
                })
                save(positions, watchlist)
                st.success(f"✅ Added {ticker} to portfolio!")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════
# TAB 2 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Watchlist  ·  {len(watchlist)} candidates")

    if watchlist:
        for i, w in enumerate(watchlist):
            col1, col2, col3, col4, col5, col6 = st.columns([2, 1, 1.2, 1.2, 1.2, 0.8])
            col1.markdown(f"**{w['ticker']}** `{w['market']}`")
            col2.metric("Growth est.", f"{w['expected_growth']}%")
            col3.metric("Max PEG", w["peg_threshold"])
            col4.metric("Min MOS", f"{int(w['mos_pct']*100)}%")
            col5.caption(w.get("thesis", "")[:60] + ("…" if len(w.get("thesis","")) > 60 else ""))
            if col6.button("🗑️ Remove", key=f"del_wl_{i}"):
                watchlist = [x for x in watchlist if x["ticker"] != w["ticker"]]
                save(positions, watchlist)
                st.success(f"Removed {w['ticker']}")
                st.rerun()
            st.divider()
    else:
        st.info("No watchlist candidates yet. Add one below.")

    st.subheader("➕ Add to Watchlist")
    with st.form("add_watchlist", clear_on_submit=True):
        c1, c2 = st.columns(2)
        ticker = c1.text_input("Ticker *", placeholder="e.g. TATAMOTORS or GOOGL").upper().strip()
        market = c2.selectbox("Market *", ["IN", "US"])

        c3, c4, c5 = st.columns(3)
        growth    = c3.number_input("Expected Growth % *", min_value=1.0, max_value=200.0, value=20.0, step=1.0,
                                     help="Annual earnings growth you expect, e.g. 20 for 20%")
        peg_max   = c4.number_input("Max PEG you'd pay *", min_value=0.1, max_value=5.0, value=1.0, step=0.1,
                                     help="PEG below this = attractive. Rule of thumb: <1 is cheap")
        mos       = c5.number_input("Min Margin of Safety % *", min_value=5, max_value=60, value=20, step=5,
                                     help="How much discount to fair value you need before buying") / 100

        thesis = st.text_area("Why are you watching this? *", placeholder="e.g. EV transition play, cheap PEG", height=80)

        submitted = st.form_submit_button("Add to Watchlist", type="primary")
        if submitted:
            if not ticker or not thesis:
                st.error("Ticker and thesis are required.")
            elif any(w["ticker"] == ticker for w in watchlist):
                st.error(f"{ticker} is already in your watchlist.")
            else:
                watchlist.append({
                    "ticker": ticker, "market": market,
                    "expected_growth": float(growth), "peg_threshold": float(peg_max),
                    "mos_pct": float(mos), "thesis": thesis,
                })
                save(positions, watchlist)
                st.success(f"✅ Added {ticker} to watchlist!")
                st.rerun()
