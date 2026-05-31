# -*- coding: utf-8 -*-
"""
Pure Python financial calculations — zero LLM, zero hallucination.
LLM provides the INPUTS; this module does all the MATH.
"""

# ── Default WACC by market + business risk ────────────────────────────────────
# Used when LLM does not return a WACC estimate.
_WACC_DEFAULTS = {
    ("IN", "high"):   15.0,
    ("IN", "medium"): 12.0,
    ("IN", "low"):    10.0,
    ("US", "high"):   12.0,
    ("US", "medium"): 10.0,
    ("US", "low"):     8.0,
}
_TERMINAL_GROWTH_DEFAULT = 3.0   # % perpetual growth in terminal value


def default_wacc(market: str, business_risk: str = "medium") -> float:
    return _WACC_DEFAULTS.get((market.upper(), business_risk.lower()), 12.0)


# ── PEG valuation (Peter Lynch) ───────────────────────────────────────────────

def peg_fair_value(eps: float, growth_pct: float) -> float | None:
    """
    Peter Lynch fair value:  EPS × (1+g)² × fair_PE
    where fair_PE = growth_pct  (e.g. 20% growth → P/E of 20)
    """
    if not eps or not growth_pct or growth_pct <= 0:
        return None
    fair_pe = max(8.0, min(50.0, growth_pct))   # cap between 8 and 50
    fy2_eps = eps * (1 + growth_pct / 100) ** 2
    return round(fy2_eps * fair_pe, 2)


def peg_target(eps: float, growth_pct: float, upside_pct: float = 10.0) -> float | None:
    fv = peg_fair_value(eps, growth_pct)
    if fv is None:
        return None
    return round(fv * (1 + upside_pct / 100), 2)


def peg_ratio(pe: float, growth_pct: float) -> float | None:
    if not pe or not growth_pct or growth_pct <= 0:
        return None
    return round(pe / growth_pct, 2)


# ── DCF valuation (5-year) ────────────────────────────────────────────────────

def dcf_fair_value(
    fcf_or_eps: float,
    growth_pct: float,
    wacc_pct: float,
    terminal_growth_pct: float = _TERMINAL_GROWTH_DEFAULT,
    years: int = 5,
) -> float | None:
    """
    Standard 5-year DCF with Gordon Growth terminal value.

    Args:
        fcf_or_eps:          Free cash flow per share (or EPS as proxy)
        growth_pct:          Expected annual growth % during projection period
        wacc_pct:            Weighted average cost of capital %
        terminal_growth_pct: Perpetual growth rate for terminal value
        years:               Projection horizon

    Returns None if inputs are invalid or WACC <= terminal growth (model breaks down).
    """
    if not fcf_or_eps or fcf_or_eps <= 0:
        return None
    if not growth_pct or not wacc_pct:
        return None
    g = growth_pct / 100
    r = wacc_pct / 100
    tg = terminal_growth_pct / 100

    if r <= tg:
        return None   # Gordon Growth model undefined

    # PV of projected cash flows
    pv_cf = 0.0
    cf = float(fcf_or_eps)
    for yr in range(1, years + 1):
        cf *= (1 + g)
        pv_cf += cf / (1 + r) ** yr

    # Terminal value (Gordon Growth on year-5 CF)
    terminal_cf = cf * (1 + tg)
    pv_terminal = (terminal_cf / (r - tg)) / (1 + r) ** years

    return round(pv_cf + pv_terminal, 2)


def dcf_target(fcf_or_eps, growth_pct, wacc_pct,
               terminal_growth_pct=_TERMINAL_GROWTH_DEFAULT,
               upside_pct=8.0, years=5) -> float | None:
    fv = dcf_fair_value(fcf_or_eps, growth_pct, wacc_pct, terminal_growth_pct, years)
    if fv is None:
        return None
    return round(fv * (1 + upside_pct / 100), 2)


# ── Margin of safety ──────────────────────────────────────────────────────────

def margin_of_safety(current: float, fair_value: float) -> float | None:
    if not current or not fair_value:
        return None
    return round((fair_value - current) / fair_value, 3)


# ── Suggested stop loss ───────────────────────────────────────────────────────

def suggested_stop_loss(current_price: float, business_risk: str = "medium") -> float:
    """Stop loss as % below current price, based on risk level."""
    pct = {"low": 0.12, "medium": 0.18, "high": 0.25}.get(business_risk.lower(), 0.18)
    return round(current_price * (1 - pct), 2)


# ── Unified builder: inputs -> full valuation dict ────────────────────────────

REQUIRED_INPUTS = ["estimated_eps", "estimated_growth_pct"]   # minimum for PEG


def build_valuation(market, current_price, avg_cost, inputs: dict) -> dict:
    """
    Compute PEG + DCF valuation from a dict of inputs (LLM-estimated or
    user-supplied). Pure Python — no LLM, no hallucination.

    inputs keys: estimated_eps, estimated_fcf_per_share, estimated_growth_pct,
                 wacc_pct, terminal_growth_pct, business_risk
    Returns the valuation dict plus a 'missing_inputs' list naming which
    required fields are still null (so the UI can prompt for them).
    """
    eps    = inputs.get("estimated_eps")
    fcf    = inputs.get("estimated_fcf_per_share") or eps
    growth = inputs.get("estimated_growth_pct")
    risk   = inputs.get("business_risk", "medium")
    wacc   = inputs.get("wacc_pct") or default_wacc(market, risk)
    tg     = inputs.get("terminal_growth_pct") or 3.0

    peg_fv  = peg_fair_value(eps, growth)
    peg_tgt = peg_target(eps, growth) if peg_fv else None
    dcf_fv  = dcf_fair_value(fcf, growth, wacc, tg) if (fcf and growth) else None
    dcf_tgt = dcf_target(fcf, growth, wacc, tg)      if dcf_fv        else None
    sl      = suggested_stop_loss(current_price, risk) if current_price else None
    best_fv = peg_fv or dcf_fv
    mos     = margin_of_safety(current_price, best_fv) if best_fv else None

    # Which inputs are still missing (blocking a full valuation)?
    missing = []
    if not eps:
        missing.append("estimated_eps")
    if not growth:
        missing.append("estimated_growth_pct")
    if not fcf:
        missing.append("estimated_fcf_per_share")

    return {
        "peg_fair_value": peg_fv, "peg_target_12m": peg_tgt,
        "dcf_fair_value": dcf_fv, "dcf_target_12m": dcf_tgt,
        "stop_loss": sl, "mos": mos,
        "missing_inputs": missing,
        "_inputs": {"eps": eps, "fcf": fcf, "growth": growth,
                    "wacc": wacc, "tg": tg, "risk": risk},
        "valuation_basis": (
            f"PEG: EPS={eps}, g={growth}% -> FV={peg_fv}. "
            f"DCF: FCF={fcf}, g={growth}%, WACC={wacc}%, TG={tg}% -> FV={dcf_fv}."
        ),
    }


# ── Legacy helpers (kept for backward compatibility) ──────────────────────────

def fair_value_from_pe(eps: float, fair_pe: float, growth_pct: float) -> float | None:
    if not eps or not fair_pe:
        return None
    fy2_eps = eps * (1 + growth_pct / 100) ** 2
    return round(fy2_eps * fair_pe, 2)


def assess(ticker, market, pe, eps, growth_pct, current_price,
           peg_threshold=1.0, mos_pct=0.20):
    fair_pe = max(15, min(35, growth_pct * 0.9))
    fv  = fair_value_from_pe(eps, fair_pe, growth_pct)
    peg = peg_ratio(pe, growth_pct)
    mos = margin_of_safety(current_price, fv) if fv else None
    return {
        "ticker": ticker, "market": market, "pe": pe,
        "growth_pct": growth_pct, "peg": peg, "fair_pe": fair_pe,
        "fair_value": fv, "current_price": current_price, "mos": mos,
        "signal": _signal(peg, mos, peg_threshold, mos_pct),
    }


def _signal(peg, mos, peg_threshold, mos_pct):
    if peg is None or mos is None:
        return "INSUFFICIENT_DATA"
    if peg <= peg_threshold and mos >= mos_pct:
        return "BUY"
    if peg <= peg_threshold * 1.5 and mos >= 0:
        return "ACCUMULATE"
    if mos < -0.20:
        return "TRIM"
    return "HOLD"
