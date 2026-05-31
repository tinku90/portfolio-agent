def peg_ratio(pe: float, growth_pct: float):
    if not pe or not growth_pct or growth_pct <= 0:
        return None
    return round(pe / growth_pct, 2)

def fair_value_from_pe(eps: float, fair_pe: float, growth_pct: float):
    """Simple forward fair value: EPS * (1 + g)^2 * fair_pe"""
    if not eps or not fair_pe:
        return None
    fy2_eps = eps * (1 + growth_pct / 100) ** 2
    return round(fy2_eps * fair_pe, 2)

def margin_of_safety(current: float, fair_value: float):
    if not current or not fair_value:
        return None
    return round((fair_value - current) / fair_value, 3)

def assess(ticker, market, pe, eps, growth_pct, current_price, peg_threshold=1.0, mos_pct=0.20):
    fair_pe = max(15, min(35, growth_pct * 0.9))  # heuristic: fair P/E ~ growth, capped
    fv = fair_value_from_pe(eps, fair_pe, growth_pct)
    peg = peg_ratio(pe, growth_pct)
    mos = margin_of_safety(current_price, fv) if fv else None
    return {
        "ticker": ticker,
        "market": market,
        "pe": pe,
        "growth_pct": growth_pct,
        "peg": peg,
        "fair_pe": fair_pe,
        "fair_value": fv,
        "current_price": current_price,
        "mos": mos,
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
