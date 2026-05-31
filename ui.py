import streamlit as st
import yaml, os, io, json, sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

import pdfplumber
from groq import Groq
from src.data.news import fetch_news
from src.data.prices import get_price

PORTFOLIO_FILE = ROOT / "config" / "portfolio.yaml"
WATCHLIST_FILE = ROOT / "config" / "watchlist.yaml"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")

st.set_page_config(page_title="Portfolio Agent", page_icon="📈", layout="wide")

# ── helpers ───────────────────────────────────────────────────────────────────

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

def extract_text(uploaded_file) -> str:
    try:
        raw = uploaded_file.read()
        if uploaded_file.name.lower().endswith(".pdf"):
            text = ""
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages[:30]:
                    text += (page.extract_text() or "") + "\n"
            return text[:40000]
        return raw.decode("utf-8", errors="ignore")[:40000]
    except Exception as e:
        return f"[Read failed: {e}]"

ANALYSIS_PROMPT = """You are a buy-side equity analyst. Analyze {ticker} ({market}-listed) using the data below.

Recent news:
{news_text}

{docs_section}

Current price : {current_price} {currency}
Investor's avg buy cost: {avg_cost}

Your tasks:
1. Estimate intrinsic fair value per share (use DCF, P/E, or EV/EBITDA — whichever the data supports best).
2. Set a realistic 12-month target price.
3. Set an absolute stop-loss price level (not a %).
4. Write a concise 2-3 sentence investment thesis.
5. List 3-5 key risk factors specific to this stock.
6. List 3-5 growth opportunities or upcoming catalysts for this stock.
7. Briefly explain your valuation methodology in one sentence.

Return ONLY valid JSON:
{{
  "fair_value": <number>,
  "target_12m": <number>,
  "stop_loss": <number>,
  "thesis": "...",
  "risks": ["...", "...", "..."],
  "opportunities": ["...", "...", "..."],
  "valuation_basis": "..."
}}"""

def run_analysis(ticker, market, avg_cost, news_items, uploaded_texts) -> dict:
    if not GROQ_KEY:
        return {"error": "GROQ_API_KEY not set in .env"}
    price_data   = get_price(ticker, market)
    current_price = price_data.get("price", avg_cost)
    currency      = price_data.get("currency", "INR" if market == "IN" else "USD")
    news_text = "\n".join(f"- {n['title']}" for n in news_items[:10]) or "No recent news."
    docs_section  = (
        "Uploaded documents:\n" +
        "\n\n---\n\n".join(f"[{name}]:\n{text[:15000]}" for name, text in uploaded_texts)
        if uploaded_texts else
        "No documents uploaded — base analysis on news and general knowledge of this company."
    )
    prompt = ANALYSIS_PROMPT.format(
        ticker=ticker, market=market, news_text=news_text, docs_section=docs_section,
        current_price=current_price, currency=currency, avg_cost=avg_cost,
    )
    try:
        resp = Groq(api_key=GROQ_KEY).chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
            max_tokens=1000,
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        return {"error": str(e)}

# ── page ──────────────────────────────────────────────────────────────────────

st.title("📈 Portfolio Agent")
st.caption("Add stocks, then use **AI Analysis** to auto-calculate fair value, target, and risks from news and uploaded documents.")

positions, watchlist = load()
tab1, tab2 = st.tabs(["💼  My Portfolio", "👁️  Watchlist"])

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 1 — PORTFOLIO
# ═══════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader(f"Holdings · {len(positions)} stocks")

    for i, p in enumerate(positions):
        fv  = p.get("fair_value") or 0
        tgt = p.get("target")     or 0
        sl  = p.get("stop_loss")  or 0

        # ── header row ────────────────────────────────────────────────────────
        h1, h2, h3, h4, h5, h6, h7 = st.columns([2.5, 1, 1.2, 1.2, 1.2, 1.2, 0.6])
        h1.markdown(f"### {p['ticker']}  `{p['market']}`")
        h2.metric("Qty",       int(p["qty"]))
        h3.metric("Avg Cost",  f"{p['avg_cost']:,.0f}")
        h4.metric("Fair Value",f"{fv:,.0f}"  if fv  else "—")
        h5.metric("Target",    f"{tgt:,.0f}" if tgt else "—")
        h6.metric("Stop Loss", f"{sl:,.0f}"  if sl  else "—")
        if h7.button("🗑️", key=f"del_{i}", help=f"Remove {p['ticker']}"):
            save([x for x in positions if x["ticker"] != p["ticker"]], watchlist)
            st.rerun()

        thesis = p.get("thesis", "")
        if thesis and thesis != "Pending analysis":
            st.caption(f"**Thesis:** {thesis}")
        risks = p.get("risks", [])
        if risks:
            st.caption("**Risks:** " + "  ·  ".join(f"⚠️ {r}" for r in risks))
        opportunities = p.get("opportunities", [])
        if opportunities:
            st.caption("**Opportunities:** " + "  ·  ".join(f"🚀 {o}" for o in opportunities))

        # ── edit expander ─────────────────────────────────────────────────────
        with st.expander("✏️  Edit position"):
            with st.form(f"edit_{i}"):
                e1, e2, e3 = st.columns(3)
                new_qty  = e1.number_input("Quantity",   value=float(p["qty"]),             min_value=0.01, step=1.0)
                new_cost = e2.number_input("Avg Cost",   value=float(p["avg_cost"]),         min_value=0.01, step=0.5)
                new_sl   = e3.number_input("Stop Loss",  value=float(sl),                    min_value=0.0,  step=0.5)
                e4, e5   = st.columns(2)
                new_fv   = e4.number_input("Fair Value", value=float(fv),                    min_value=0.0,  step=0.5)
                new_tgt  = e5.number_input("Target",     value=float(tgt),                   min_value=0.0,  step=0.5)
                if st.form_submit_button("💾  Save changes", type="primary"):
                    positions[i].update(qty=new_qty, avg_cost=new_cost,
                                        stop_loss=new_sl, fair_value=new_fv, target=new_tgt)
                    save(positions, watchlist)
                    st.success("Saved!")
                    st.rerun()

        # ── AI analysis expander ───────────────────────────────────────────────
        with st.expander("🤖  AI Analysis  —  upload transcripts & annual report"):
            st.markdown("#### Recent news")
            with st.spinner("Fetching…"):
                try:
                    recent_news = fetch_news(p["ticker"], p["market"], "")
                    for n in recent_news[:6]:
                        st.markdown(f"- [{n['title'][:100]}]({n['url']})")
                except Exception as e:
                    st.warning(f"News fetch failed: {e}")
                    recent_news = []

            st.markdown("---")
            st.markdown("#### Upload documents *(optional)*")
            st.caption("Accepted: PDF or TXT — concall transcripts (Q1, Q2), annual report")
            uploads = st.file_uploader(
                "Drop files here",
                accept_multiple_files=True,
                type=["pdf", "txt"],
                key=f"up_{p['ticker']}",
            )

            if st.button("▶  Run AI Analysis", key=f"run_{i}", type="primary"):
                docs = [(uf.name, extract_text(uf)) for uf in (uploads or [])]
                with st.spinner("Analysing with Groq LLM — this takes ~10 seconds…"):
                    result = run_analysis(p["ticker"], p["market"], p["avg_cost"], recent_news, docs)

                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    positions[i].update(
                        fair_value    = result.get("fair_value",    fv),
                        target        = result.get("target_12m",    tgt),
                        stop_loss     = result.get("stop_loss",     sl),
                        thesis        = result.get("thesis",        thesis),
                        risks         = result.get("risks",         []),
                        opportunities = result.get("opportunities", []),
                    )
                    save(positions, watchlist)

                    st.success("✅ Analysis complete — position updated!")
                    r1, r2, r3 = st.columns(3)
                    r1.metric("Fair Value",    f"{result['fair_value']:,.0f}")
                    r2.metric("Target (12m)",  f"{result['target_12m']:,.0f}")
                    r3.metric("Stop Loss",     f"{result['stop_loss']:,.0f}")
                    st.markdown(f"**Thesis:** {result.get('thesis','')}")
                    st.markdown(f"**Valuation basis:** {result.get('valuation_basis','')}")
                    col_risk, col_opp = st.columns(2)
                    if result.get("risks"):
                        with col_risk:
                            st.markdown("**⚠️ Risk factors**")
                            for r in result["risks"]:
                                st.markdown(f"- {r}")
                    if result.get("opportunities"):
                        with col_opp:
                            st.markdown("**🚀 Growth opportunities**")
                            for o in result["opportunities"]:
                                st.markdown(f"- {o}")
                    st.rerun()

        st.divider()

    # ── add position form ──────────────────────────────────────────────────────
    st.subheader("➕  Add a Position")
    st.caption("Only enter what you know. Fair value, target, and stop-loss are auto-calculated by AI Analysis.")
    with st.form("add_pos", clear_on_submit=True):
        a1, a2, a3, a4 = st.columns(4)
        new_ticker   = a1.text_input("Ticker *",         placeholder="INFY / NVDA").upper().strip()
        new_market   = a2.selectbox("Market *",           ["IN", "US"])
        new_qty      = a3.number_input("Quantity *",      min_value=0.01, step=1.0)
        new_avg_cost = a4.number_input("Avg Buy Price *", min_value=0.01, step=0.5)

        if st.form_submit_button("Add to Portfolio", type="primary"):
            if not new_ticker:
                st.error("Ticker is required.")
            elif any(p["ticker"] == new_ticker for p in positions):
                st.error(f"{new_ticker} is already in portfolio.")
            else:
                positions.append({
                    "ticker":     new_ticker,
                    "market":     new_market,
                    "qty":        float(new_qty),
                    "avg_cost":   float(new_avg_cost),
                    "entry_date": str(date.today()),
                    "thesis":     "Pending analysis",
                    "fair_value": round(new_avg_cost * 1.3, 2),
                    "stop_loss":  round(new_avg_cost * 0.8, 2),
                    "target":     round(new_avg_cost * 1.5, 2),
                })
                save(positions, watchlist)
                st.success(f"✅ {new_ticker} added! Click **AI Analysis** to generate fair value and target.")
                st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WATCHLIST
# ═══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader(f"Watchlist · {len(watchlist)} candidates")

    for i, w in enumerate(watchlist):
        w1, w2, w3, w4, w5 = st.columns([2.5, 1.2, 1.2, 1.2, 0.6])
        w1.markdown(f"### {w['ticker']}  `{w['market']}`")
        w2.metric("Expected Growth", f"{w['expected_growth']}%")
        w3.metric("Max PEG",         w["peg_threshold"])
        w4.metric("Min MOS",         f"{int(w['mos_pct']*100)}%")
        if w5.button("🗑️", key=f"del_wl_{i}", help=f"Remove {w['ticker']}"):
            save(positions, [x for x in watchlist if x["ticker"] != w["ticker"]])
            st.rerun()

        if w.get("thesis"):
            st.caption(f"**Thesis:** {w['thesis']}")

        with st.expander("✏️  Edit"):
            with st.form(f"edit_wl_{i}"):
                we1, we2, we3 = st.columns(3)
                ng = we1.number_input("Expected Growth %", value=float(w["expected_growth"]), min_value=1.0, step=1.0)
                np = we2.number_input("Max PEG",           value=float(w["peg_threshold"]),   min_value=0.1, step=0.1)
                nm = we3.number_input("Min MOS %",         value=int(w["mos_pct"]*100),       min_value=5,   step=5) / 100
                nt = st.text_area("Thesis", value=w.get("thesis", ""))
                if st.form_submit_button("💾  Save", type="primary"):
                    watchlist[i].update(expected_growth=ng, peg_threshold=np, mos_pct=nm, thesis=nt)
                    save(positions, watchlist)
                    st.success("Saved!")
                    st.rerun()

        st.divider()

    st.subheader("➕  Add to Watchlist")
    with st.form("add_wl", clear_on_submit=True):
        b1, b2 = st.columns(2)
        wl_ticker = b1.text_input("Ticker *", placeholder="TATAMOTORS / GOOGL").upper().strip()
        wl_market = b2.selectbox("Market *", ["IN", "US"])
        b3, b4, b5 = st.columns(3)
        wl_growth = b3.number_input("Expected Growth % *", min_value=1.0, max_value=200.0, value=20.0, step=1.0,
                                     help="Annual earnings growth you expect")
        wl_peg    = b4.number_input("Max PEG you'd pay *",  min_value=0.1, max_value=5.0,  value=1.0,  step=0.1,
                                     help="PEG below this = attractive")
        wl_mos    = b5.number_input("Min Margin of Safety %*", min_value=5, max_value=60, value=20, step=5,
                                     help="Discount to fair value needed before you buy") / 100
        wl_thesis = st.text_area("Why are you watching this? *", placeholder="EV transition play, cheap PEG…", height=68)

        if st.form_submit_button("Add to Watchlist", type="primary"):
            if not wl_ticker or not wl_thesis:
                st.error("Ticker and thesis are required.")
            elif any(x["ticker"] == wl_ticker for x in watchlist):
                st.error(f"{wl_ticker} already in watchlist.")
            else:
                watchlist.append({
                    "ticker": wl_ticker, "market": wl_market,
                    "expected_growth": float(wl_growth), "peg_threshold": float(wl_peg),
                    "mos_pct": float(wl_mos), "thesis": wl_thesis,
                })
                save(positions, watchlist)
                st.success(f"✅ {wl_ticker} added to watchlist!")
                st.rerun()
