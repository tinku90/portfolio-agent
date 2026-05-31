import sqlite3
from contextlib import contextmanager
from pathlib import Path

DB_PATH = Path(__file__).parent.parent.parent / "data" / "agent.db"
DB_PATH.parent.mkdir(exist_ok=True)

SCHEMA = """
CREATE TABLE IF NOT EXISTS positions (ticker TEXT PRIMARY KEY, market TEXT, qty REAL, avg_cost REAL, entry_date TEXT, thesis TEXT, fair_value REAL, stop_loss REAL, target REAL);
CREATE TABLE IF NOT EXISTS watchlist (ticker TEXT PRIMARY KEY, market TEXT, thesis TEXT, expected_growth REAL, peg_threshold REAL, mos_pct REAL);
CREATE TABLE IF NOT EXISTS filings (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, source TEXT, title TEXT, url TEXT, published TEXT, processed INTEGER DEFAULT 0, UNIQUE(ticker, url));
CREATE TABLE IF NOT EXISTS news (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, title TEXT, url TEXT, published TEXT, material INTEGER DEFAULT 0, summary TEXT, UNIQUE(ticker, url));
CREATE TABLE IF NOT EXISTS alerts (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, kind TEXT, payload TEXT, sent_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE TABLE IF NOT EXISTS valuations (id INTEGER PRIMARY KEY AUTOINCREMENT, ticker TEXT, pe REAL, growth_pct REAL, peg REAL, fair_value REAL, current_price REAL, mos REAL, computed_at TEXT DEFAULT CURRENT_TIMESTAMP);
CREATE INDEX IF NOT EXISTS idx_filings_unprocessed ON filings(processed) WHERE processed=0;
CREATE INDEX IF NOT EXISTS idx_news_unprocessed ON news(material) WHERE material IS NULL;
"""

@contextmanager
def conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute("PRAGMA foreign_keys=ON")
    try:
        yield c
        c.commit()
    finally:
        c.close()

def init():
    with conn() as c:
        c.executescript(SCHEMA)

def sync_yaml(positions: list, watchlist: list):
    with conn() as c:
        c.execute("DELETE FROM positions")
        for p in positions:
            c.execute("INSERT INTO positions VALUES (:ticker,:market,:qty,:avg_cost,:entry_date,:thesis,:fair_value,:stop_loss,:target)", p)
        c.execute("DELETE FROM watchlist")
        for w in watchlist:
            c.execute("INSERT INTO watchlist VALUES (:ticker,:market,:thesis,:expected_growth,:peg_threshold,:mos_pct)", w)

def save_filing(ticker, source, title, url, published):
    with conn() as c:
        try:
            cur = c.execute("INSERT INTO filings (ticker, source, title, url, published) VALUES (?,?,?,?,?)", (ticker, source, title, url, published))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

def unprocessed_filings():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM filings WHERE processed=0 ORDER BY published DESC")]

def mark_processed(filing_id):
    with conn() as c:
        c.execute("UPDATE filings SET processed=1 WHERE id=?", (filing_id,))

def save_news(ticker, title, url, published):
    with conn() as c:
        try:
            cur = c.execute("INSERT INTO news (ticker, title, url, published) VALUES (?,?,?,?)", (ticker, title, url, published))
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None

def update_news_materiality(news_id, material, summary):
    with conn() as c:
        c.execute("UPDATE news SET material=?, summary=? WHERE id=?", (1 if material else 0, summary, news_id))

def unprocessed_news():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM news WHERE material IS NULL ORDER BY published DESC LIMIT 50")]

def log_alert(ticker, kind, payload):
    with conn() as c:
        c.execute("INSERT INTO alerts (ticker, kind, payload) VALUES (?,?,?)", (ticker, kind, payload))

def save_valuation(ticker, pe, growth_pct, peg, fair_value, current_price, mos):
    with conn() as c:
        c.execute("INSERT INTO valuations (ticker, pe, growth_pct, peg, fair_value, current_price, mos) VALUES (?,?,?,?,?,?,?)",
                  (ticker, pe, growth_pct, peg, fair_value, current_price, mos))

def all_positions():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM positions")]

def all_watchlist():
    with conn() as c:
        return [dict(r) for r in c.execute("SELECT * FROM watchlist")]
