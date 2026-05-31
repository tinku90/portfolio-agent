# Portfolio Agent

Personal portfolio management agent tracking Indian (NSE/BSE) and US stocks.

## Setup

```bash
cp .env.example .env
# fill in GROQ_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
pip install -r requirements.txt
python -m src.main bootstrap
```

## Jobs

```bash
python -m src.main hourly    # prices + filings scan
python -m src.main daily     # news + opportunity scan
python -m src.main weekly    # digest
```
# portfolio-agent
