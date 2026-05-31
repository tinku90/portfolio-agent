# Portfolio Agent — Verification & Run Spec

## Your job
The project is already scaffolded. Do NOT recreate files that exist. Your job:
1. Verify the directory structure and file contents match the spec below.
2. Report any missing or malformed files; create only what is missing.
3. Install dependencies, run bootstrap, run a daily job, fix any errors, report results.

## Project overview
A personal portfolio management agent that:
- Tracks Indian (NSE/BSE) and US (NYSE/NASDAQ) stock positions + a watchlist.
- Fetches prices (yfinance), filings (BSE/NSE via Google News RSS + SEC EDGAR), news (Google News RSS).
- Uses Groq LLM to classify materiality, summarize news, and rerate stocks after results.
- Sends alerts to Telegram.
- Runs on GitHub Actions cron (hourly, daily, weekly jobs).
- SQLite for state, committed back to the repo by the workflow.

## Stack
- Python 3.11+ in `./venv`
- LLM: Groq only
  - `llama-3.3-70b-versatile` → materiality + news summaries
  - `openai/gpt-oss-120b` → concall/results rerating
- Notifications: Telegram bot
- Free tier everything; no Anthropic, no Gemini, no Screener API.

## Expected directory structure
```
.
├── CLAUDE.md
├── README.md
├── requirements.txt
├── .env.example
├── .env                        (gitignored, holds real keys)
├── .gitignore
├── config/
│   ├── portfolio.yaml          (holdings + theses)
│   └── watchlist.yaml          (opportunity candidates)
├── src/
│   ├── __init__.py
│   ├── main.py                 (entry: hourly|daily|weekly|bootstrap)
│   ├── data/
│   │   ├── __init__.py
│   │   ├── prices.py           (yfinance wrapper, IN uses .NS suffix)
│   │   ├── filings_in.py       (BSE/NSE filings via Google News RSS)
│   │   ├── filings_us.py       (SEC EDGAR atom feed)
│   │   └── news.py             (Google News RSS per ticker)
│   ├── analysis/
│   │   ├── __init__.py
│   │   ├── valuation.py        (PEG, fair value, MOS, BUY/HOLD/TRIM signal)
│   │   ├── materiality.py      (Groq: classify + summarize)
│   │   └── rerating.py         (Groq: rerate after results, with PDF extract)
│   ├── storage/
│   │   ├── __init__.py
│   │   └── db.py               (SQLite at data/agent.db)
│   └── notify/
│       ├── __init__.py
│       └── telegram.py         (alert_price, alert_filing, alert_rerating, alert_opportunity, digest)
├── .github/workflows/
│   ├── hourly.yml              (cron */30 4-11 * * 1-5)
│   ├── daily.yml               (cron 30 3 * * 1-5)
│   └── weekly.yml              (cron 0 4 * * 0)
└── data/
    ├── .gitkeep
    └── agent.db                (created on first bootstrap)
```

## Key contracts (do not break these)

### `.env` keys required
```
GROQ_API_KEY=
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
```
If absent, the telegram module prints to stdout instead of sending — that's intentional for local testing.

### `main.py` invocation
```
python -m src.main bootstrap   # init DB + sync YAMLs
python -m src.main hourly      # prices + filings scan
python -m src.main daily       # news + opportunity scan
python -m src.main weekly      # digest
```

### Groq model strings (confirmed available on free tier)
```python
# materiality.py
MODEL = "llama-3.3-70b-versatile"

# rerating.py
MODEL = "openai/gpt-oss-120b"
```
If `models.list()` shows neither, fall back to `qwen/qwen3-32b` for rerating and report the change.

### SQLite path
`data/agent.db` — created by `db.init()` on bootstrap. Schema in `src/storage/db.py`.

### Materiality JSON shape
```json
{"material": true|false, "reason": "...", "category": "results|guidance|ma|rating|regulatory|promoter|contract|other"}
```

### Rerating JSON shape
```json
{
  "verdict": "HOLD|ACCUMULATE|TRIM|EXIT",
  "guidance_status": "BEAT|MEET|MISS|MIXED",
  "new_fair_value_low": <number>,
  "new_fair_value_high": <number>,
  "thesis_delta": "...",
  "key_numbers": {"revenue_growth_pct": <number>, "ebitda_margin_pct": <number>, "pat_growth_pct": <number>},
  "new_risks": ["..."],
  "action": "..."
}
```

### Indian ticker convention
yfinance uses `<TICKER>.NS` suffix for NSE-listed stocks. US tickers used as-is.

## Verification steps (run in order)

1. **Structure check** — confirm every file in the expected tree exists. List anything missing. Do NOT overwrite existing files.

2. **Syntax check** — for each `.py` file, run `python -m py_compile <file>`. Report any syntax errors.

3. **Imports check** — run `python -c "from src.main import bootstrap, job_hourly, job_daily, job_weekly"` and confirm no import errors.

4. **Install deps** — `pip install -r requirements.txt`. Expected packages: `yfinance, feedparser, requests, beautifulsoup4, pyyaml, python-dotenv, groq, pdfplumber, python-dateutil`.

5. **Env check** — confirm `.env` exists and contains `GROQ_API_KEY`. If missing or empty, prompt user to add it before continuing.

6. **Groq model check** — run this and confirm both required models appear in the list:
```python
   from groq import Groq
   import os
   c = Groq(api_key=os.environ["GROQ_API_KEY"])
   print([m.id for m in c.models.list().data])
```
   Required: `llama-3.3-70b-versatile`, `openai/gpt-oss-120b`. If `openai/gpt-oss-120b` missing, replace `MODEL = "openai/gpt-oss-120b"` in `src/analysis/rerating.py` with `MODEL = "qwen/qwen3-32b"` and report.

7. **Bootstrap** — `python -m src.main bootstrap`. Expect: "Loaded N positions, M watchlist". Confirm `data/agent.db` was created.

8. **Smoke test daily job** — `python -m src.main daily`. This will:
   - Fetch news for every position + watchlist ticker
   - Run materiality classification through Groq
   - Run opportunity scan with valuation
   - Print results to console (telegram disabled if no creds)

   Fix any errors. Common ones:
   - `yfinance` rate limits → add retry with backoff
   - Empty `info` dict from yfinance → already handled by `.get()`
   - `pdfplumber` failing on non-PDF URLs → already handled, returns error string
   - Groq rate limit → free tier is 30 req/min, 14400/day; should be fine

9. **Report** — at the end, summarize:
   - Files verified vs files created
   - Tests that passed
   - Errors encountered and fixes applied
   - Whether `data/agent.db` was created and how many rows in each table
   - Any model substitutions made

## What NOT to do
- Do not add new dependencies beyond `requirements.txt` without explicit user approval.
- Do not change the LLM provider (Groq only).
- Do not commit `.env` or `data/agent.db` if `.gitignore` excludes them — but note the workflows DO commit `data/agent.db` back to the repo intentionally.
- Do not modify `config/portfolio.yaml` or `config/watchlist.yaml` content — that's user data.
- Do not refactor working code "for cleanliness". Only fix what is broken.

## If files are missing
Recreate from the full spec in the conversation history (the prior turn before this CLAUDE.md was generated). Match exactly — same file paths, same code.

## Success criteria
- `python -m src.main bootstrap` runs without error
- `python -m src.main daily` completes (errors on individual stocks OK, but no fatal crash)
- `data/agent.db` exists with rows in `positions`, `watchlist`, and at least one of `news` / `valuations`
- Report delivered with checklist of what passed.