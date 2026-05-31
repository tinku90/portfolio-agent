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
from src.analysis.watchlist_valuation import analyze_watchlist as _analyze_wl
from src.notify.telegram import send as tg_send, alert_watchlist_analysis as tg_wl

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
        wfv     = w.get("fair_value", 0)      or 0
        wdcf    = w.get("dcf_fair_value", 0)  or 0
        wtgt    = w.get("peg_target", 0)       or 0
        wdtgt   = w.get("dcf_target", 0)      or 0
        wentry  = w.get("entry_price", 0)      or 0

        # ── header row ────────────────────────────────────────────────────────
        c1,c2,c3,c4,c5,c6,c7,c8 = st.columns([2.2,1,1.1,1.1,1.1,1.1,1.1,0.5])
        c1.markdown(f"### {w['ticker']}  `{w['market']}`")
        c2.metric("Growth est.",  f"{w['expected_growth']}%")
        c3.metric("FV (PEG)",     f"{wfv:,.0f}"    if wfv    else "—")
        c4.metric("FV (DCF)",     f"{wdcf:,.0f}"   if wdcf   else "—")
        c5.metric("Entry price",  f"{wentry:,.0f}" if wentry else "—")
        c6.metric("Max PEG",      w["peg_threshold"])
        c7.metric("Min MOS",      f"{int(w['mos_pct']*100)}%")
        if c8.button("🗑️", key=f"del_wl_{i}", help=f"Remove {w['ticker']}"):
            save(positions, [x for x in watchlist if x["ticker"] != w["ticker"]])
            st.rerun()

        if w.get("thesis") and w["thesis"] != "Pending analysis":
            st.caption(f"**Thesis:** {w['thesis']}")
        if w.get("risks"):
            st.caption("**Risks:** " + "  ·  ".join(f"⚠️ {r}" for r in w["risks"]))
        if w.get("opportunities"):
            st.caption("**Opportunities:** " + "  ·  ".join(f"🚀 {o}" for o in w["opportunities"]))

        if wfv and st.button(f"📤 Send {w['ticker']} to Telegram", key=f"tg_wl_{i}"):
            price_data = get_price(w["ticker"], w["market"])
            cp = price_data.get("price", 0) if "error" not in price_data else 0
            tg_wl(w["ticker"], w["market"], cp, {
                "expected_growth_pct":    w.get("expected_growth", 0),
                "peg_fair_value":         wfv,   "peg_target_12m":  wtgt,
                "dcf_fair_value":         wdcf,  "dcf_target_12m":  wdtgt,
                "suggested_peg_threshold":w.get("peg_threshold", 1.0),
                "suggested_mos_pct":      w.get("mos_pct", 0.2),
                "entry_price":            wentry,
                "thesis":                 w.get("thesis", ""),
                "risks":                  w.get("risks", []),
                "opportunities":          w.get("opportunities", []),
            })
            st.success(f"Sent {w['ticker']} to Telegram!")

        # ── edit expander ─────────────────────────────────────────────────────
        with st.expander("✏️  Edit — manual override"):
            st.caption("AI Analysis sets these automatically. Edit only to override.")
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

        # ── AI analysis expander ───────────────────────────────────────────────
        with st.expander("🤖  AI Analysis — upload transcripts & annual report"):
            st.markdown("#### Recent news")
            with st.spinner("Fetching…"):
                try:
                    wl_news = fetch_news(w["ticker"], w["market"], "")
                    for n in wl_news[:5]:
                        st.markdown(f"- [{n['title'][:100]}]({n['url']})")
                except Exception:
                    wl_news = []

            st.markdown("---")
            st.caption("Accepted: PDF, Word, TXT, JPG/PNG, XLS/XLSX, CSV")
            wl_uploads = st.file_uploader(
                "Drop files here",
                accept_multiple_files=True,
                type=["pdf","txt","jpg","jpeg","png","xls","xlsx","csv","doc","docx"],
                key=f"wl_up_{w['ticker']}",
            )
            wl_raw = st.text_area(
                "Or paste text",
                height=120,
                key=f"wl_raw_{w['ticker']}",
                placeholder="Paste concall transcript, annual report excerpt…",
            )

            if st.button("▶  Run AI Analysis", key=f"wl_run_{i}", type="primary"):
                docs = [(uf.name, extract_text(uf)) for uf in (wl_uploads or [])]
                if (wl_raw or "").strip():
                    docs.append(("pasted_text", wl_raw.strip()))
                price_data    = get_price(w["ticker"], w["market"])
                current_price = price_data.get("price", 0) if "error" not in price_data else 0
                currency      = price_data.get("currency", "INR" if w["market"] == "IN" else "USD")
                with st.spinner("Analysing with Groq LLM…"):
                    result = _analyze_wl(w["ticker"], w["market"], current_price, currency, wl_news, docs or None)

                if "error" in result:
                    st.error(f"Analysis failed: {result['error']}")
                else:
                    watchlist[i].update(
                        expected_growth = result.get("expected_growth_pct",    w["expected_growth"]),
                        peg_threshold   = result.get("suggested_peg_threshold",w["peg_threshold"]),
                        mos_pct         = result.get("suggested_mos_pct",      w["mos_pct"]),
                        fair_value      = result.get("peg_fair_value",         0),
                        peg_target      = result.get("peg_target_12m",         0),
                        dcf_fair_value  = result.get("dcf_fair_value",         0),
                        dcf_target      = result.get("dcf_target_12m",         0),
                        entry_price     = result.get("entry_price",            0),
                        thesis          = result.get("thesis",                 w.get("thesis","")),
                        risks           = result.get("risks",                  []),
                        opportunities   = result.get("opportunities",          []),
                    )
                    save(positions, watchlist)
                    tg_wl(w["ticker"], w["market"], current_price, result)

                    st.success("✅ Analysis complete — sent to Telegram!")
                    m1,m2,m3,m4,m5,m6 = st.columns(6)
                    m1.metric("Growth",      f"{result.get('expected_growth_pct',0)}%")
                    m2.metric("FV (PEG)",    f"{result.get('peg_fair_value',0):,.0f}")
                    m3.metric("FV (DCF)",    f"{result.get('dcf_fair_value',0):,.0f}")
                    m4.metric("Entry price", f"{result.get('entry_price',0):,.0f}")
                    m5.metric("Max PEG",     result.get("suggested_peg_threshold",1.0))
                    m6.metric("Min MOS",     f"{int(result.get('suggested_mos_pct',0.2)*100)}%")
                    st.markdown(f"**Thesis:** {result.get('thesis','')}")
                    st.caption(f"*{result.get('valuation_basis','')}*")
                    col_r, col_o = st.columns(2)
                    if result.get("risks"):
                        with col_r:
                            st.markdown("**⚠️ Risks**")
                            for r in result["risks"]: st.markdown(f"- {r}")
                    if result.get("opportunities"):
                        with col_o:
                            st.markdown("**🚀 Opportunities**")
                            for o in result["opportunities"]: st.markdown(f"- {o}")
                    st.rerun()

        st.divider()

    # ── add to watchlist ──────────────────────────────────────────────────────
    st.subheader("➕  Add to Watchlist")
    st.caption("Upload documents for instant AI valuation, or leave blank to analyze later.")

    b1, b2 = st.columns(2)
    wl_ticker = b1.text_input("Ticker *", placeholder="TATAMOTORS / GOOGL", key="wl_ticker").upper().strip()
    wl_market = b2.selectbox("Market *", ["IN", "US"], key="wl_market")

    wl_uploads = st.file_uploader(
        "Documents for instant valuation *(optional)*",
        accept_multiple_files=True,
        type=["pdf","txt","jpg","jpeg","png","xls","xlsx","csv","doc","docx"],
        key="wl_add_uploads",
    )
    wl_raw_text = st.text_area(
        "Or paste text",
        height=100, key="wl_add_raw",
        placeholder="Paste annual report excerpt, concall transcript…",
    )

    if st.button("Add to Watchlist", type="primary", key="wl_add_btn"):
        if not wl_ticker:
            st.error("Ticker is required.")
        elif any(x["ticker"] == wl_ticker for x in watchlist):
            st.error(f"{wl_ticker} already in watchlist.")
        else:
            new_wl = {
                "ticker": wl_ticker, "market": wl_market,
                "expected_growth": 20.0, "peg_threshold": 1.0,
                "mos_pct": 0.20, "thesis": "Pending analysis",
            }
            has_docs = bool(wl_uploads or (wl_raw_text or "").strip())
            if has_docs:
                docs = [(uf.name, extract_text(uf)) for uf in (wl_uploads or [])]
                if (wl_raw_text or "").strip():
                    docs.append(("pasted_text", wl_raw_text.strip()))
                price_data    = get_price(wl_ticker, wl_market)
                current_price = price_data.get("price", 0) if "error" not in price_data else 0
                currency      = price_data.get("currency", "INR" if wl_market == "IN" else "USD")
                with st.spinner(f"Analysing {wl_ticker}…"):
                    result = _analyze_wl(wl_ticker, wl_market, current_price, currency, [], docs)
                if "error" not in result:
                    new_wl.update(
                        expected_growth = result.get("expected_growth_pct",    20.0),
                        peg_threshold   = result.get("suggested_peg_threshold",1.0),
                        mos_pct         = result.get("suggested_mos_pct",      0.20),
                        fair_value      = result.get("peg_fair_value",         0),
                        peg_target      = result.get("peg_target_12m",         0),
                        dcf_fair_value  = result.get("dcf_fair_value",         0),
                        dcf_target      = result.get("dcf_target_12m",         0),
                        entry_price     = result.get("entry_price",            0),
                        thesis          = result.get("thesis",                 ""),
                        risks           = result.get("risks",                  []),
                        opportunities   = result.get("opportunities",          []),
                    )
                    watchlist.append(new_wl)
                    save(positions, watchlist)
                    tg_wl(wl_ticker, wl_market, current_price, result)
                    st.success(f"✅ {wl_ticker} added with AI valuation — sent to Telegram!")
                    r1,r2,r3,r4,r5,r6 = st.columns(6)
                    r1.metric("Growth",      f"{result.get('expected_growth_pct',0)}%")
                    r2.metric("FV (PEG)",    f"{result.get('peg_fair_value',0):,.0f}")
                    r3.metric("FV (DCF)",    f"{result.get('dcf_fair_value',0):,.0f}")
                    r4.metric("Entry price", f"{result.get('entry_price',0):,.0f}")
                    r5.metric("Max PEG",     result.get("suggested_peg_threshold",1.0))
                    r6.metric("Min MOS",     f"{int(result.get('suggested_mos_pct',0.2)*100)}%")
                    st.markdown(f"**Thesis:** {result.get('thesis','')}")
                    st.rerun()
                else:
                    st.warning(f"Analysis failed ({result['error']}), stock added without valuation.")
                    watchlist.append(new_wl)
                    save(positions, watchlist)
                    st.rerun()
            else:
                watchlist.append(new_wl)
                save(positions, watchlist)
                st.success(f"✅ {wl_ticker} added. Open **AI Analysis** to calculate fair value.")
                st.rerun()
