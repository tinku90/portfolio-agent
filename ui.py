import streamlit as st
import yaml, os, io, json, sys
from pathlib import Path
from datetime import date
from dotenv import load_dotenv

ROOT = Path(__file__).parent
load_dotenv(ROOT / ".env")
sys.path.insert(0, str(ROOT))

import base64
import pandas as pd
import pdfplumber
import docx
from groq import Groq
from src.data.news import fetch_news
from src.data.prices import get_price
from src.analysis.ai_valuation import analyze_position as _analyze
from src.notify.telegram import send as tg_send

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

def extract_text_from_image(raw: bytes, filename: str) -> str:
    try:
        ext  = filename.lower().rsplit(".", 1)[-1]
        mime = "image/png" if ext == "png" else "image/jpeg"
        b64  = base64.b64encode(raw).decode()
        resp = Groq(api_key=GROQ_KEY).chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
            messages=[{"role": "user", "content": [
                {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{b64}"}},
                {"type": "text", "text": (
                    "Extract all text and financial data visible in this image. "
                    "If it contains tables or charts, describe the key numbers and trends clearly."
                )},
            ]}],
            max_tokens=2000,
        )
        return resp.choices[0].message.content
    except Exception as e:
        return f"[Image extraction failed: {e}]"

def extract_text(uploaded_file) -> str:
    try:
        raw  = uploaded_file.read()
        name = uploaded_file.name.lower()
        if name.endswith(".pdf"):
            text = ""
            with pdfplumber.open(io.BytesIO(raw)) as pdf:
                for page in pdf.pages[:30]:
                    text += (page.extract_text() or "") + "\n"
            return text[:40000]
        elif name.endswith((".doc", ".docx")):
            doc = docx.Document(io.BytesIO(raw))
            return "\n".join(p.text for p in doc.paragraphs)[:40000]
        elif name.endswith(".csv"):
            df = pd.read_csv(io.BytesIO(raw))
            return df.to_string(index=False)[:40000]
        elif name.endswith((".xls", ".xlsx")):
            df = pd.read_excel(io.BytesIO(raw))
            return df.to_string(index=False)[:40000]
        elif name.endswith((".jpg", ".jpeg", ".png")):
            return extract_text_from_image(raw, uploaded_file.name)
        else:
            return raw.decode("utf-8", errors="ignore")[:40000]
    except Exception as e:
        return f"[Read failed: {e}]"

def send_analysis_to_telegram(ticker, market, avg_cost, result):
    msg  = f"*AI Valuation: {ticker}* ({market})\n\n"
    msg += f"*Avg Cost:* `{avg_cost:,.2f}`\n"
    msg += f"*PEG* — FV: `{result.get('peg_fair_value',0):,.0f}` | Target: `{result.get('peg_target_12m',0):,.0f}`\n"
    msg += f"*DCF* — FV: `{result.get('dcf_fair_value',0):,.0f}` | Target: `{result.get('dcf_target_12m',0):,.0f}`\n"
    msg += f"*Stop Loss:* `{result.get('stop_loss',0):,.0f}`\n\n"
    msg += f"_{result.get('thesis', '')}_\n\n"
    if result.get("risks"):
        msg += "*Risks:*\n" + "".join(f"⚠️ {r}\n" for r in result["risks"][:3]) + "\n"
    if result.get("opportunities"):
        msg += "*Opportunities:*\n" + "".join(f"🚀 {o}\n" for o in result["opportunities"][:3])
    tg_send(msg)

def run_analysis(ticker, market, avg_cost, news_items, uploaded_texts) -> dict:
    price_data    = get_price(ticker, market)
    current_price = price_data.get("price", avg_cost) if "error" not in price_data else avg_cost
    currency      = price_data.get("currency", "INR" if market == "IN" else "USD")
    return _analyze(ticker, market, avg_cost, current_price, currency, news_items, uploaded_texts or None)

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
        fv      = p.get("fair_value")      or 0
        tgt     = p.get("target")         or 0
        sl      = p.get("stop_loss")      or 0
        dcf_fv  = p.get("dcf_fair_value") or 0
        dcf_tgt = p.get("dcf_target")     or 0

        # ── header row ────────────────────────────────────────────────────────
        h1, h2, h3, h4, h5, h6, h7, h8 = st.columns([2.2, 1, 1.1, 1.1, 1.1, 1.1, 1.1, 0.5])
        h1.markdown(f"### {p['ticker']}  `{p['market']}`")
        h2.metric("Qty",            int(p["qty"]))
        h3.metric("Avg Cost",       f"{p['avg_cost']:,.0f}")
        h4.metric("FV (PEG)",       f"{fv:,.0f}"      if fv      else "—")
        h5.metric("Target (PEG)",   f"{tgt:,.0f}"     if tgt     else "—")
        h6.metric("FV (DCF)",       f"{dcf_fv:,.0f}"  if dcf_fv  else "—")
        h7.metric("Stop Loss",      f"{sl:,.0f}"      if sl      else "—")
        if h8.button("🗑️", key=f"del_{i}", help=f"Remove {p['ticker']}"):
            save([x for x in positions if x["ticker"] != p["ticker"]], watchlist)
            st.rerun()

        if fv and st.button(f"📤 Send {p['ticker']} to Telegram", key=f"tg_{i}"):
            send_analysis_to_telegram(p["ticker"], p["market"], p["avg_cost"], {
                "peg_fair_value": fv, "peg_target_12m": tgt,
                "dcf_fair_value": dcf_fv, "dcf_target_12m": dcf_tgt,
                "stop_loss": sl, "thesis": p.get("thesis",""),
                "risks": p.get("risks",[]), "opportunities": p.get("opportunities",[]),
            })
            st.success(f"Sent {p['ticker']} to Telegram!")

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
            st.caption("Fair value and target are set by AI Analysis only — edit trading data here.")
            with st.form(f"edit_{i}"):
                e1, e2, e3 = st.columns(3)
                new_qty  = e1.number_input("Quantity",  value=float(p["qty"]),     min_value=0.01, step=1.0)
                new_cost = e2.number_input("Avg Cost",  value=float(p["avg_cost"]),min_value=0.01, step=0.5)
                new_sl   = e3.number_input("Stop Loss", value=float(sl),           min_value=0.0,  step=0.5)
                if st.form_submit_button("💾  Save changes", type="primary"):
                    positions[i].update(qty=new_qty, avg_cost=new_cost, stop_loss=new_sl)
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
            st.caption("Accepted: PDF, Word (DOC/DOCX), TXT, JPG/PNG, XLS/XLSX, CSV — concall transcripts, annual reports, screener exports")
            uploads = st.file_uploader(
                "Drop files here",
                accept_multiple_files=True,
                type=["pdf", "txt", "jpg", "jpeg", "png", "xls", "xlsx", "csv", "doc", "docx"],
                key=f"up_{p['ticker']}",
            )
            raw_text = st.text_area(
                "Or paste raw text here (concall transcript, news article, annual report excerpt…)",
                height=150,
                key=f"raw_{p['ticker']}",
                placeholder="Paste any text you want the AI to consider…",
            )

            if st.button("▶  Run AI Analysis", key=f"run_{i}", type="primary"):
                docs = [(uf.name, extract_text(uf)) for uf in (uploads or [])]
                if raw_text.strip():
                    docs.append(("pasted_text", raw_text.strip()))
                with st.spinner("Analysing with Groq LLM — this takes ~10 seconds…"):
                    result = run_analysis(p["ticker"], p["market"], p["avg_cost"], recent_news, docs)

                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    positions[i].update(
                        fair_value    = result.get("peg_fair_value",  fv),
                        target        = result.get("peg_target_12m",  tgt),
                        dcf_fair_value= result.get("dcf_fair_value",  dcf_fv),
                        dcf_target    = result.get("dcf_target_12m",  dcf_tgt),
                        stop_loss     = result.get("stop_loss",       sl),
                        thesis        = result.get("thesis",          thesis),
                        risks         = result.get("risks",           []),
                        opportunities = result.get("opportunities",   []),
                    )
                    save(positions, watchlist)
                    send_analysis_to_telegram(p["ticker"], p["market"], p["avg_cost"], result)

                    st.success("✅ Analysis complete — sent to Telegram!")
                    v1, v2, v3, v4, v5 = st.columns(5)
                    v1.metric("FV — PEG",      f"{result.get('peg_fair_value',0):,.0f}")
                    v2.metric("Target — PEG",  f"{result.get('peg_target_12m',0):,.0f}")
                    v3.metric("FV — DCF",      f"{result.get('dcf_fair_value',0):,.0f}")
                    v4.metric("Target — DCF",  f"{result.get('dcf_target_12m',0):,.0f}")
                    v5.metric("Stop Loss",     f"{result.get('stop_loss',0):,.0f}")
                    st.markdown(f"**Thesis:** {result.get('thesis','')}")
                    st.caption(f"*Valuation basis: {result.get('valuation_basis','')}*")
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
    st.caption("Upload documents to get fair value immediately, or leave blank to analyze later.")

    a1, a2, a3, a4 = st.columns(4)
    new_ticker   = a1.text_input("Ticker *", placeholder="INFY / NVDA", key="add_ticker").upper().strip()
    new_market   = a2.selectbox("Market *", ["IN", "US"], key="add_market")
    new_qty      = a3.number_input("Quantity *", min_value=0.01, step=1.0, key="add_qty")
    new_avg_cost = a4.number_input("Avg Buy Price *", min_value=0.01, step=0.5, key="add_cost")

    add_uploads  = st.file_uploader(
        "Documents for instant valuation *(optional — PDF, TXT, JPG, PNG, XLS, CSV)*",
        accept_multiple_files=True,
        type=["pdf", "txt", "jpg", "jpeg", "png", "xls", "xlsx", "csv", "doc", "docx"],
        key="add_uploads",
    )
    add_raw_text = st.text_area(
        "Or paste text *(annual report excerpt, concall transcript…)*",
        height=100, key="add_raw_text",
        placeholder="Paste any text for the AI to use in valuation…",
    )

    if st.button("Add to Portfolio", type="primary", key="add_btn"):
        if not new_ticker:
            st.error("Ticker is required.")
        elif any(p["ticker"] == new_ticker for p in positions):
            st.error(f"{new_ticker} is already in portfolio.")
        else:
            new_pos = {
                "ticker": new_ticker, "market": new_market,
                "qty": float(new_qty), "avg_cost": float(new_avg_cost),
                "entry_date": str(date.today()), "thesis": "Pending analysis",
                "fair_value": 0, "dcf_fair_value": 0,
                "stop_loss": 0, "target": 0, "dcf_target": 0,
            }
            has_docs = bool(add_uploads or (add_raw_text or "").strip())
            if has_docs:
                docs = [(uf.name, extract_text(uf)) for uf in (add_uploads or [])]
                if (add_raw_text or "").strip():
                    docs.append(("pasted_text", add_raw_text.strip()))
                with st.spinner(f"Analysing {new_ticker} with Groq…"):
                    price_data    = get_price(new_ticker, new_market)
                    current_price = price_data.get("price", new_avg_cost) if "error" not in price_data else new_avg_cost
                    currency      = price_data.get("currency", "INR" if new_market == "IN" else "USD")
                    result = _analyze(new_ticker, new_market, new_avg_cost, current_price, currency, [], docs)
                if "error" not in result:
                    new_pos.update(
                        fair_value     = result.get("peg_fair_value", 0),
                        target         = result.get("peg_target_12m", 0),
                        dcf_fair_value = result.get("dcf_fair_value", 0),
                        dcf_target     = result.get("dcf_target_12m", 0),
                        stop_loss      = result.get("stop_loss", 0),
                        thesis         = result.get("thesis", "Pending analysis"),
                        risks          = result.get("risks", []),
                        opportunities  = result.get("opportunities", []),
                    )
                    positions.append(new_pos)
                    save(positions, watchlist)
                    send_analysis_to_telegram(new_ticker, new_market, new_avg_cost, result)
                    st.success(f"✅ {new_ticker} added with AI valuation — sent to Telegram!")
                    r1, r2, r3, r4, r5 = st.columns(5)
                    r1.metric("FV — PEG",     f"{result.get('peg_fair_value',0):,.0f}")
                    r2.metric("Target — PEG", f"{result.get('peg_target_12m',0):,.0f}")
                    r3.metric("FV — DCF",     f"{result.get('dcf_fair_value',0):,.0f}")
                    r4.metric("Target — DCF", f"{result.get('dcf_target_12m',0):,.0f}")
                    r5.metric("Stop Loss",    f"{result.get('stop_loss',0):,.0f}")
                    st.markdown(f"**Thesis:** {result.get('thesis','')}")
                    st.rerun()
                else:
                    st.warning(f"Analysis failed ({result['error']}), stock added without valuation.")
                    positions.append(new_pos)
                    save(positions, watchlist)
                    st.rerun()
            else:
                positions.append(new_pos)
                save(positions, watchlist)
                st.success(f"✅ {new_ticker} added. Upload documents via **AI Analysis** to calculate fair value.")
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
