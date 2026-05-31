"""
Usage:
  python manage.py list
  python manage.py add-position  TICKER MARKET QTY AVG_COST FAIR_VALUE STOP_LOSS TARGET "thesis"
  python manage.py remove-position TICKER
  python manage.py add-watchlist  TICKER MARKET EXPECTED_GROWTH PEG_THRESHOLD MOS_PCT "thesis"
  python manage.py remove-watchlist TICKER

Examples:
  python manage.py add-position INFY IN 100 1800 2200 1500 2600 "IT services, steady FCF"
  python manage.py add-watchlist TATAMOTORS IN 20 0.8 0.20 "EV transition play"
  python manage.py remove-position NVDA
  python manage.py list
"""
import sys
import yaml
from pathlib import Path
from datetime import date

ROOT = Path(__file__).parent
PORTFOLIO_FILE = ROOT / "config" / "portfolio.yaml"
WATCHLIST_FILE = ROOT / "config" / "watchlist.yaml"


def load():
    with open(PORTFOLIO_FILE, encoding='utf-8') as f:
        pf = yaml.safe_load(f) or {"positions": []}
    with open(WATCHLIST_FILE, encoding='utf-8') as f:
        wl = yaml.safe_load(f) or {"candidates": []}
    return pf["positions"] or [], wl["candidates"] or []


def save(positions, watchlist):
    with open(PORTFOLIO_FILE, "w", encoding="utf-8") as f:
        yaml.dump({"positions": positions}, f, default_flow_style=False, allow_unicode=True)
    with open(WATCHLIST_FILE, "w", encoding="utf-8") as f:
        yaml.dump({"candidates": watchlist}, f, default_flow_style=False, allow_unicode=True)


def cmd_list():
    positions, watchlist = load()
    print(f"\n{'='*55}")
    print(f"  PORTFOLIO ({len(positions)} positions)")
    print(f"{'='*55}")
    if positions:
        print(f"  {'TICKER':<12} {'MKT':<5} {'QTY':>6} {'AVG COST':>10} {'TARGET':>10} {'STOP':>10}")
        print(f"  {'-'*52}")
        for p in positions:
            print(f"  {p['ticker']:<12} {p['market']:<5} {p['qty']:>6} {p['avg_cost']:>10} {p['target']:>10} {p['stop_loss']:>10}")
    else:
        print("  (empty)")

    print(f"\n{'='*55}")
    print(f"  WATCHLIST ({len(watchlist)} candidates)")
    print(f"{'='*55}")
    if watchlist:
        print(f"  {'TICKER':<12} {'MKT':<5} {'GROWTH':>8} {'PEG':>6} {'MOS':>6}")
        print(f"  {'-'*40}")
        for w in watchlist:
            print(f"  {w['ticker']:<12} {w['market']:<5} {w['expected_growth']:>7}% {w['peg_threshold']:>6} {int(w['mos_pct']*100):>5}%")
    else:
        print("  (empty)")
    print()


def cmd_add_position(args):
    if len(args) < 8:
        print("Usage: add-position TICKER MARKET QTY AVG_COST FAIR_VALUE STOP_LOSS TARGET \"thesis\"")
        sys.exit(1)
    ticker, market, qty, avg_cost, fair_value, stop_loss, target, thesis = (
        args[0].upper(), args[1].upper(), float(args[2]), float(args[3]),
        float(args[4]), float(args[5]), float(args[6]), args[7]
    )
    positions, watchlist = load()
    if any(p["ticker"] == ticker for p in positions):
        print(f"  {ticker} already in portfolio. Remove it first to update.")
        sys.exit(1)
    positions.append({
        "ticker": ticker, "market": market, "qty": qty, "avg_cost": avg_cost,
        "entry_date": str(date.today()), "thesis": thesis,
        "fair_value": fair_value, "stop_loss": stop_loss, "target": target,
    })
    save(positions, watchlist)
    print(f"  Added {ticker} ({market}) to portfolio.")


def cmd_remove_position(args):
    if not args:
        print("Usage: remove-position TICKER")
        sys.exit(1)
    ticker = args[0].upper()
    positions, watchlist = load()
    before = len(positions)
    positions = [p for p in positions if p["ticker"] != ticker]
    if len(positions) == before:
        print(f"  {ticker} not found in portfolio.")
        sys.exit(1)
    save(positions, watchlist)
    print(f"  Removed {ticker} from portfolio.")


def cmd_add_watchlist(args):
    if len(args) < 6:
        print("Usage: add-watchlist TICKER MARKET EXPECTED_GROWTH PEG_THRESHOLD MOS_PCT \"thesis\"")
        sys.exit(1)
    ticker, market, growth, peg, mos, thesis = (
        args[0].upper(), args[1].upper(), float(args[2]),
        float(args[3]), float(args[4]), args[5]
    )
    positions, watchlist = load()
    if any(w["ticker"] == ticker for w in watchlist):
        print(f"  {ticker} already in watchlist. Remove it first to update.")
        sys.exit(1)
    watchlist.append({
        "ticker": ticker, "market": market, "expected_growth": growth,
        "peg_threshold": peg, "mos_pct": mos, "thesis": thesis,
    })
    save(positions, watchlist)
    print(f"  Added {ticker} ({market}) to watchlist.")


def cmd_remove_watchlist(args):
    if not args:
        print("Usage: remove-watchlist TICKER")
        sys.exit(1)
    ticker = args[0].upper()
    positions, watchlist = load()
    before = len(watchlist)
    watchlist = [w for w in watchlist if w["ticker"] != ticker]
    if len(watchlist) == before:
        print(f"  {ticker} not found in watchlist.")
        sys.exit(1)
    save(positions, watchlist)
    print(f"  Removed {ticker} from watchlist.")


COMMANDS = {
    "list": (cmd_list, []),
    "add-position": (cmd_add_position, sys.argv[2:]),
    "remove-position": (cmd_remove_position, sys.argv[2:]),
    "add-watchlist": (cmd_add_watchlist, sys.argv[2:]),
    "remove-watchlist": (cmd_remove_watchlist, sys.argv[2:]),
}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in COMMANDS:
        print(__doc__)
        sys.exit(1)
    fn, args = COMMANDS[sys.argv[1]]
    fn() if sys.argv[1] == "list" else fn(args)
