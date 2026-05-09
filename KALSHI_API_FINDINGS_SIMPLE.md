# Kalshi API Research — What I Found

**Date:** April 24, 2026
**By:** Syam

---

## What is the Kalshi API?

Kalshi is a prediction market — people bet on whether things will happen ("Will Trump say 'tariff' in his speech?"). Their API is a way for our code to ask Kalshi questions and get answers back, like:

- What contracts are open right now?
- What words is Trump betting markets watching?
- How did past contracts settle?
- What was the price history minute-by-minute?

Every request goes to this address:

**`https://api.elections.kalshi.com/trade-api/v2`**

You don't need a login or password to *read* data. You only need credentials if you want to actually place trades.

---

## The big "gotcha": Live data vs. Historical data

Kalshi keeps the most recent ~3 months of data in a "live" section, and everything older in a "historical" section. They're at different URLs.

- Recent stuff → `/markets`, `/events`, `/trades`
- Old stuff → `/historical/markets`, `/historical/trades`, etc.

If you ask the live endpoint for an old event, you'll get an empty response. You have to use the `/historical/...` version.

You can check today's cutoff date by hitting:
`/historical/cutoff`

Right now the cutoff is **February 23, 2026** — anything before that is "historical."

---

## What we can pull from the API

### 1. List of contract series (categories)
A "series" is a recurring contract type. Examples I found in the Politics category:

| Series Ticker | What it is |
|---|---|
| `KXVANCEINGRAHAM` | Words J.D. Vance will say on Ingraham Angle |
| `KXELONDJTSAY` | Will Elon Musk say "Trump" |
| `KXWHBRIEFING` | Words mentioned in White House press briefings |

These are exactly the kind of "mention markets" our word counter is designed for.

### 2. List of past events under a series
For example, the March 14, 2025 Vance/Ingraham appearance had its own event ticker: `KXVANCEINGRAHAM-25MAR14`. Under that one event, Kalshi listed **22 separate word markets** — one for each word like "Ukraine," "Meme," "Couch," "St. Patrick's," etc.

### 3. Per-market detail
For each individual word market, we get:

- The **target word** itself (e.g. "Meme")
- How it **settled** (yes / no)
- The **final price** and the **payout amount**
- **Trading volume** and **open interest** (how active it was)
- The **resolution rules** in plain English
- The **exact word-matching logic Kalshi used** to settle it

### 4. Price history (candlesticks)
Minute-by-minute (or any interval we want) chart data: open/high/low/close price, bid/ask, volume, open interest. We can replay how the market moved second by second during the actual speech.

---

## The most important discovery

When I pulled a real settled market (`KXVANCEINGRAHAM-25MAR14-MEME`), the API returned the official Kalshi rule for how they decided whether the word counted:

> "The exact phrase/word, **or a plural or possessive form**, must be used. Grammatical inflections are NOT included. Hyphenated or compound words are NOT included. A compound word group separated by spaces (e.g. 'fire truck') WILL count if it includes the word, but a single compound word (e.g. 'fireman') will NOT."

**This is exactly what our `kalshi_word_counter.py` already does.** Our counter was built to match these rules — and now we have proof straight from the API that they match. We can also pull this rule per market and automatically check if Kalshi ever changes the wording.

---

## Why this matters for our pipeline

Right now, `kalshi_word_counter.py` has a hardcoded list of ~35 target words (`DEFAULT_TARGETS`). With the API, we can:

1. **Stop hardcoding words.** Pull the actual word list directly from a real Kalshi event by ticker.
2. **Backtest our counter against reality.** Take a past speech transcript (from `transcript_bot.py`), run the word counter on it, then compare our counts to how the Kalshi markets actually settled. We'd quickly see if our counter is calling words right or wrong.
3. **Find where the alpha is.** Use the minute-by-minute price data to see *when* during a speech each word's market moved, and see if our counter could have predicted it in real time.
4. **Auto-detect rule changes.** Pull `rules_secondary` per market; alert if Kalshi ever changes how they count.

---

## What I actually tested (all worked)

1. ✅ Got the live/historical cutoff date
2. ✅ Listed all political contract series
3. ✅ Found the Vance/Ingraham event from March 14, 2025
4. ✅ Pulled all 22 settled word markets for that event with results, prices, volumes, rules
5. ✅ Pulled minute-by-minute candlestick data for one of the markets

---

## Still unknown / to follow up

- **What's the SOTU series ticker?** I tried `KXSOTUMENTION` — empty. Need to find the real one (probably visible on news.kalshi.com or by browsing markets around February 24, 2026).
- **Do we ever need to actually trade?** Reading is free and unauthenticated. Trading requires an API key + RSA signature setup.
- **WebSocket feed** for real-time price ticks during a live speech — exists but I haven't tested it yet.

---

## Suggested next steps for the meeting

1. Build a small `kalshi_api.py` helper file that wraps the API calls.
2. Add a function `get_target_words(event_ticker)` so our counter pulls the live word list automatically.
3. Build a `backtest.py` that runs our counter on a past transcript and scores it against real settled markets.
4. Find the actual SOTU series ticker.
5. Decide whether we eventually want trade access (separate setup).

---

## Reference links

- Kalshi API documentation: https://docs.kalshi.com/welcome
- Markets endpoint: https://docs.kalshi.com/api-reference/market/get-markets
- Historical markets: https://docs.kalshi.com/api-reference/historical/get-historical-markets
- Live vs historical overview: https://docs.kalshi.com/getting_started/historical_data
