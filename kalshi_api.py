"""
kalshi_api.py
-------------
"""

from __future__ import annotations

import os
import re
import time
import requests
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Optional

try:
    from bs4 import BeautifulSoup
    _BS4 = True
except ImportError:
    _BS4 = False

BASE_URL      = "https://api.elections.kalshi.com/trade-api/v2"
DEMO_BASE_URL = "https://demo-api.kalshi.co/trade-api/v2"
_LAST_HIT = 0.0
_MIN_GAP = 0.4   # seconds between requests — keeps us under Kalshi's rate cap


@dataclass
class KalshiMarket:
    ticker: str               # e.g. "KXVANCEINGRAHAM-25MAR14-MEME"
    word: Optional[str]       # the target word, e.g. "Meme"
    result: str               # "yes" / "no" / "" (unsettled)
    status: str               # "open" / "closed" / "finalized"
    last_price: float         # last traded price, 0.00–1.00
    yes_ask: float            # cheapest YES you can buy right now
    no_ask: float             # cheapest NO you can buy right now
    yes_bid: float            # best bid for YES (what you'd get selling)
    no_bid: float             # best bid for NO
    settlement_value: float   # final payout per share (post-resolution)
    volume: float
    open_interest: float
    close_time: Optional[str] # ISO timestamp when trading closes
    # "previous" prices = price before the most recent change.
    # On settled markets these are typically the realistic mid-range prices
    # before the market converged to 0.99/0.01 — useful for backtesting.
    previous_yes_ask: float
    previous_yes_bid: float
    previous_price:   float
    rules_primary: str
    rules_secondary: str

    @property
    def is_open(self) -> bool:
        return self.status == "open"

    @property
    def best_yes_price(self) -> float:
        """Price you actually pay to BUY YES — use this for EV, not last_price."""
        return self.yes_ask if self.yes_ask > 0 else self.last_price

    @property
    def best_no_price(self) -> float:
        """Price you actually pay to BUY NO — use this for EV, not last_price."""
        return self.no_ask if self.no_ask > 0 else (1.0 - self.last_price)

    @property
    def spread_yes(self) -> float:
        """Bid-ask spread on YES side. Wide spread = thin market = bad fills."""
        if self.yes_ask <= 0 or self.yes_bid <= 0:
            return 1.0  # treat as max spread when missing
        return max(0.0, self.yes_ask - self.yes_bid)

    @property
    def spread_no(self) -> float:
        """Bid-ask spread on NO side."""
        if self.no_ask <= 0 or self.no_bid <= 0:
            return 1.0
        return max(0.0, self.no_ask - self.no_bid)

    def seconds_to_close(self) -> Optional[float]:
        """Seconds until market closes. Returns None if no close_time."""
        if not self.close_time:
            return None
        try:
            close_dt = datetime.fromisoformat(self.close_time.replace("Z", "+00:00"))
            return (close_dt - datetime.now(timezone.utc)).total_seconds()
        except (ValueError, AttributeError):
            return None


def _get(path: str, params: dict | None = None,
         _retries: int = 6, _backoff: float = 5.0) -> dict:
    """GET with exponential backoff on 429 / 5xx."""
    global _LAST_HIT
    for attempt in range(_retries):
        # Honour minimum gap between requests
        gap = time.time() - _LAST_HIT
        if gap < _MIN_GAP:
            time.sleep(_MIN_GAP - gap)
        _LAST_HIT = time.time()

        r = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)

        if r.status_code == 429 or r.status_code >= 500:
            wait = _backoff * (2 ** attempt)   # 5, 10, 20, 40, 80, 160 s
            print(f"  [rate-limit] {r.status_code} — waiting {wait:.0f}s "
                  f"(attempt {attempt+1}/{_retries}) ...")
            time.sleep(wait)
            continue

        r.raise_for_status()
        return r.json()

    # Final attempt — let it raise naturally
    r = requests.get(f"{BASE_URL}{path}", params=params, timeout=20)
    r.raise_for_status()
    return r.json()


# ---------------------------------------------------------------------------
# Authenticated write client (demo + live)
# ---------------------------------------------------------------------------

def _auth_headers(demo: bool = True) -> dict:
    """
    Return Authorization + Content-Type headers for authenticated Kalshi requests.

    Reads KALSHI_DEMO_API_KEY (demo=True) or KALSHI_API_KEY (demo=False) from
    the environment / .env file. Raises RuntimeError if the key is absent so
    callers get a clear error instead of a silent 401.
    """
    env_var = "KALSHI_DEMO_API_KEY" if demo else "KALSHI_API_KEY"
    key = os.getenv(env_var, "").strip()
    if not key:
        env_name = "demo" if demo else "live"
        raise RuntimeError(
            f"Missing {env_var} environment variable. "
            f"Add it to your .env file to use the {env_name} Kalshi API.\n"
            f"  Get your key at: {'https://demo.kalshi.com' if demo else 'https://kalshi.com'} "
            f"→ Account → API Access"
        )
    return {
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
    }


def _post(path: str, payload: dict, demo: bool = True) -> dict:
    """
    Authenticated POST to Kalshi API (demo by default) with rate-limiting.

    Uses the same _MIN_GAP throttle as _get() to stay within Kalshi's rate cap.
    Raises requests.HTTPError on non-2xx responses.
    """
    global _LAST_HIT
    gap = time.time() - _LAST_HIT
    if gap < _MIN_GAP:
        time.sleep(_MIN_GAP - gap)
    _LAST_HIT = time.time()

    base = DEMO_BASE_URL if demo else BASE_URL
    r = requests.post(
        f"{base}{path}",
        json=payload,
        headers=_auth_headers(demo=demo),
        timeout=20,
    )
    r.raise_for_status()
    return r.json()


def get_balance(demo: bool = True) -> float:
    """
    Fetch the current account balance from Kalshi in dollars.

    Kalshi reports balances in cents; this converts to dollars automatically.
    Returns 0.0 on any error (prints a warning) so the caller can still run
    with the default bankroll.

    demo=True  → uses DEMO_BASE_URL + KALSHI_DEMO_API_KEY
    demo=False → uses BASE_URL + KALSHI_API_KEY
    """
    global _LAST_HIT
    try:
        gap = time.time() - _LAST_HIT
        if gap < _MIN_GAP:
            time.sleep(_MIN_GAP - gap)
        _LAST_HIT = time.time()

        base = DEMO_BASE_URL if demo else BASE_URL
        r = requests.get(
            f"{base}/portfolio/balance",
            headers=_auth_headers(demo=demo),
            timeout=20,
        )
        r.raise_for_status()
        data = r.json()
        cents = data.get("balance", 0)
        return float(cents) / 100.0
    except RuntimeError as exc:
        print(f"  [balance] {exc}")
        return 0.0
    except Exception as exc:
        env_label = "demo" if demo else "live"
        print(f"  [balance] Could not fetch {env_label} balance: {exc}")
        return 0.0


def place_order(
    ticker: str,
    side: str,
    count: int,
    action: str = "buy",
    demo: bool = True,
) -> dict:
    """
    Place a market order on Kalshi (demo by default).

    Parameters
    ----------
    ticker : str
        The *market* ticker (e.g. "KXTRUMPSOTU-26FEB25-MEME"), NOT the event
        ticker. These are the `ticker` fields on KalshiMarket objects.
    side : str
        "yes" or "no"
    count : int
        Number of contracts to buy/sell.
    action : str
        "buy" (default) or "sell"
    demo : bool
        True → Kalshi demo environment (no real money).
        False → live production (real money — use with caution).

    Returns the full Kalshi API response dict, which contains:
        {"order": {"order_id": "...", "status": "...", ...}}

    Raises requests.HTTPError on rejection (e.g. insufficient funds, bad ticker).
    """
    if count <= 0:
        raise ValueError(f"place_order: count must be ≥ 1, got {count}")
    if side not in ("yes", "no"):
        raise ValueError(f"place_order: side must be 'yes' or 'no', got {side!r}")
    if action not in ("buy", "sell"):
        raise ValueError(f"place_order: action must be 'buy' or 'sell', got {action!r}")

    return _post(
        "/portfolio/orders",
        {
            "ticker": ticker,
            "action": action,
            "side":   side,
            "count":  count,
            "type":   "market",
        },
        demo=demo,
    )


def get_event_markets(event_ticker: str, historical: bool | None = None) -> list[KalshiMarket]:
    """
    Fetch all word markets under a Kalshi event.

    historical=None  → auto-detect: try historical first, then live
    historical=True  → only historical (older events)
    historical=False → only live (~3 month window)
    """
    if historical is None:
        # Try live first (current/upcoming events), then historical
        for path in ("/markets", "/historical/markets"):
            try:
                data = _get(path, {"event_ticker": event_ticker, "limit": 1000})
                if data.get("markets"):
                    break
            except Exception:
                continue
        else:
            data = {"markets": []}
    else:
        path = "/historical/markets" if historical else "/markets"
        data = _get(path, {"event_ticker": event_ticker, "limit": 1000})

    out: list[KalshiMarket] = []
    for m in data.get("markets", []):
        word = (m.get("custom_strike") or {}).get("Word")
        out.append(KalshiMarket(
            ticker=m["ticker"],
            word=word,
            result=m.get("result", ""),
            status=m.get("status", ""),
            last_price=float(m.get("last_price_dollars")      or 0),
            yes_ask=float(m.get("yes_ask_dollars")            or 0),
            no_ask=float(m.get("no_ask_dollars")              or 0),
            yes_bid=float(m.get("yes_bid_dollars")            or 0),
            no_bid=float(m.get("no_bid_dollars")              or 0),
            settlement_value=float(m.get("settlement_value_dollars") or 0),
            volume=float(m.get("volume_fp")                   or 0),
            open_interest=float(m.get("open_interest_fp")     or 0),
            close_time=m.get("close_time"),
            previous_yes_ask=float(m.get("previous_yes_ask_dollars") or 0),
            previous_yes_bid=float(m.get("previous_yes_bid_dollars") or 0),
            previous_price=float(m.get("previous_price_dollars")     or 0),
            rules_primary=m.get("rules_primary", ""),
            rules_secondary=m.get("rules_secondary", ""),
        ))
    return out


def get_target_words(event_ticker: str, historical: bool | None = None) -> list[str]:
    """Convenience: just the word list, ready to feed KalshiCounter(targets=...)."""
    return [m.word for m in get_event_markets(event_ticker, historical) if m.word]


def get_event_meta(event_ticker: str) -> dict:
    """Fetch the event metadata (title, sub_title, series_ticker, etc.)."""
    data = _get(f"/events/{event_ticker}")
    return data.get("event", {}) or {}


def get_pre_event_mid_price(market_ticker: str, event_date: str,
                            window_days: int = 7) -> float:
    """
    Return median pre-event YES mid-price from historical hourly candlesticks.

    For finalized markets, the trades API returns 0 trades (purged), but the
    historical candlesticks endpoint still has hourly OHLC + bid/ask data.
    We pull a `window_days` window ending the day BEFORE event_date, take
    candles with valid trade prices or tight spreads, and return the median.

    Returns 0.0 if no usable candle data.
    """
    import statistics
    if not event_date:
        return 0.0
    try:
        end_dt   = datetime.strptime(event_date, "%Y-%m-%d")
        start_dt = end_dt - timedelta(days=window_days)
        end_ts   = int(end_dt.timestamp())          # event-day 00:00 = pre-event cutoff
        start_ts = int(start_dt.timestamp())

        url = f"{BASE_URL}/historical/markets/{market_ticker}/candlesticks"
        r = requests.get(
            url,
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60},
            timeout=15,
        )
        if r.status_code != 200:
            return 0.0
        candles = r.json().get("candlesticks", []) or []
        if not candles:
            return 0.0

        prices: list[float] = []
        for c in candles:
            ts = int(c.get("end_period_ts") or 0)
            if ts >= end_ts:
                continue   # skip event-day or later candles

            # Prefer mean trade price when the bar had volume
            mean_p = (c.get("price") or {}).get("mean")
            if mean_p is not None:
                try:
                    p = float(mean_p)
                    if 0.04 < p < 0.96:
                        prices.append(p)
                        continue
                except (TypeError, ValueError):
                    pass

            # Fall back to bid/ask mid when spread is reasonable
            ya = (c.get("yes_ask") or {}).get("close")
            yb = (c.get("yes_bid") or {}).get("close")
            if ya is None or yb is None:
                continue
            try:
                ya = float(ya); yb = float(yb)
            except (TypeError, ValueError):
                continue
            if 0.0 < yb < 1.0 and 0.0 < ya < 1.0 and (ya - yb) < 0.30:
                prices.append((ya + yb) / 2.0)

        if prices:
            return float(statistics.median(prices))

        # Historical candlestick endpoint had no data (common for recent events).
        # Fall back to the live candlestick endpoint which covers ~3 months.
        url_live = f"{BASE_URL}/markets/{market_ticker}/candlesticks"
        r2 = requests.get(
            url_live,
            params={"start_ts": start_ts, "end_ts": end_ts, "period_interval": 60},
            timeout=15,
        )
        if r2.status_code != 200:
            return 0.0
        candles2 = r2.json().get("candlesticks", []) or []
        for c in candles2:
            ts = int(c.get("end_period_ts") or 0)
            if ts >= end_ts:
                continue
            mean_p = (c.get("price") or {}).get("mean")
            if mean_p is not None:
                try:
                    p = float(mean_p)
                    if 0.04 < p < 0.96:
                        prices.append(p)
                        continue
                except (TypeError, ValueError):
                    pass
            ya = (c.get("yes_ask") or {}).get("close")
            yb = (c.get("yes_bid") or {}).get("close")
            if ya is None or yb is None:
                continue
            try:
                ya = float(ya); yb = float(yb)
            except (TypeError, ValueError):
                continue
            if 0.0 < yb < 1.0 and 0.0 < ya < 1.0 and (ya - yb) < 0.30:
                prices.append((ya + yb) / 2.0)

        if not prices:
            return 0.0
        return float(statistics.median(prices))
    except Exception:
        return 0.0


def list_series(category: str = "Politics", limit: int = 200) -> list[dict]:
    """All series in a category."""
    return _get("/series", {"category": category, "limit": limit}).get("series", []) or []


def list_events(series_ticker: str, limit: int = 200) -> list[dict]:
    """
    All events under a series — paginates through every page so no events
    are missed even when a series has more than `limit` entries per page.
    """
    all_events: list[dict] = []
    cursor: str | None = None
    while True:
        params: dict = {"series_ticker": series_ticker, "limit": limit}
        if cursor:
            params["cursor"] = cursor
        data   = _get("/events", params)
        batch  = data.get("events") or []
        all_events.extend(batch)
        cursor = data.get("cursor") or data.get("next_cursor")
        # Stop when the batch is smaller than the page size or no cursor
        if not batch or not cursor or len(batch) < limit:
            break
        time.sleep(0.3)   # respect rate limit between pages
    return all_events


def find_speaker_events(
    speaker_name: str,
    categories: tuple[str, ...] = ("Politics", "Economics"),
    max_events: int = 10,
) -> list[dict]:
    """
    Discover past Kalshi mention-style events that mention this speaker.

    Returns up to `max_events` event dicts ({event_ticker, title, series_ticker, ...}),
    newest first.
    """
    parts = [p.lower() for p in speaker_name.split() if len(p) > 1]
    surname = parts[-1] if parts else ""
    full = " ".join(parts)
    needles = {n for n in (full, surname) if n}

    # SAY-style series (e.g. KXTRUMPSAY) are weekly aggregates — a market
    # covers everything said across an entire week, so one transcript can
    # never match reliably. Only MENTION/SOTU/specific-event series are
    # scoped to a single speech and useful for per-event backtesting.
    _WEEKLY_PATTERNS = ("SAY",)

    def _is_weekly(ticker: str) -> bool:
        t = ticker.upper()
        return any(t.endswith(p) or f"{p}-" in t for p in _WEEKLY_PATTERNS)

    # Hand-curated map of speaker → known per-event series.
    KNOWN: dict[str, list[str]] = {
        "trump":     ["KXTRUMPMENTION", "KXTRUMPSOTU"],
        "vance":     ["KXVANCEMENTION", "KXVANCEINGRAHAM"],
        "musk":      ["KXELONMENTION"],
        "powell":    ["KXFOMCMENTION", "KXPOWELLMENTION"],
        "biden":     ["KXBIDENMENTION"],
        "bessent":   ["KXBESSENTMENTION"],
        "bowman":    ["KXBOWMANMENTION"],
        "williams":  ["KXWILLIAMSMENTION"],
        "cook":      ["KXCOOKMENTION"],
        "jefferson": ["KXJEFFERSONMENTION"],
        "rubio":     ["KXRUBIOMENTION"],
        "barr":      ["KXBARRMENTION"],
        "hegseth":   ["KXHEGSETHMENTION"],
    }

    # Speculative ticker patterns to probe for *any* speaker. Kalshi follows
    # a consistent KX<NAME><SUFFIX> convention for mention markets, so for an
    # unknown speaker we try the surname (and full last+first) with common
    # suffixes. Misses 404 quietly.
    speculative: list[str] = []
    upper_surname = surname.upper()
    if upper_surname:
        for suffix in ("MENTION", "SOTU", "SAY"):  # SAY filtered later as weekly
            speculative.append(f"KX{upper_surname}{suffix}")

    candidate_series: list[dict] = []

    # Direct-probe known + speculative series for this speaker
    probe_tickers = list(dict.fromkeys(KNOWN.get(surname, []) + speculative))
    for tk in probe_tickers:
        try:
            s = _get(f"/series/{tk}").get("series", {}) or {}
            if s and not _is_weekly(s.get("ticker", "")):
                candidate_series.append(s)
        except Exception:
            continue

    # Then scan the listed series in each category for any other matches.
    # We match the speaker by surname appearing inside the ticker code itself
    # (e.g. KXVANCEINGRAHAM contains "vance"), which catches series whose
    # public titles don't repeat the speaker name.
    for cat in categories:
        for s in list_series(cat):
            ticker = s.get("ticker", "")
            if _is_weekly(ticker):
                continue
            blob = (ticker + " " + s.get("title", "")).lower()
            mention_like = any(k in blob for k in
                               ("mention", "word", "speech", "sotu", "briefing", "remark"))
            speaker_hit = any(n in blob for n in needles)
            if mention_like and speaker_hit:
                candidate_series.append(s)

    seen: set[str] = set()
    events: list[dict] = []
    for s in candidate_series:
        try:
            for e in list_events(s["ticker"]):
                tk = e.get("event_ticker")
                if not tk or tk in seen:
                    continue
                seen.add(tk)
                events.append(e)
        except Exception:
            continue

    # Newest first by event_ticker (which encodes the date suffix)
    events.sort(key=lambda e: e.get("event_ticker", ""), reverse=True)
    return events[:max_events]


# Map a Kalshi series_ticker to a likely speaker name for transcript_bot.
_SERIES_TO_SPEAKER = {
    "KXVANCEINGRAHAM": "J.D. Vance",
    "KXELONDJTSAY":    "Elon Musk",
    "KXWHBRIEFING":    "Karoline Leavitt",
    "KXWHBRIEFINGEY":  "Karoline Leavitt",
}


_NAME_MAP = {
    "trump": "Donald Trump", "biden": "Joe Biden",
    "powell": "Jerome Powell", "vance": "J.D. Vance",
    "obama": "Barack Obama", "harris": "Kamala Harris",
    "musk": "Elon Musk",
}

def guess_speaker(event_ticker: str) -> Optional[str]:
    """Best-effort speaker guess from event metadata. Returns None if unknown."""
    series = event_ticker.split("-", 1)[0]
    if series in _SERIES_TO_SPEAKER:
        return _SERIES_TO_SPEAKER[series]

    # Try parsing speaker name directly from the ticker string
    ticker_lower = event_ticker.lower()
    for name, full in _NAME_MAP.items():
        if name in ticker_lower:
            return full

    # Fall back to event metadata title
    try:
        meta = get_event_meta(event_ticker)
    except Exception:
        return None
    title = (meta.get("title") or "").lower()
    for name, full in _NAME_MAP.items():
        if name in title:
            return full
    return None


# ---------------------------------------------------------------------------
# Event timing — Kalshi API has no time fields, so we scrape externally
# ---------------------------------------------------------------------------

@dataclass
class EventTiming:
    event_ticker: str
    event_date:   Optional[str]       # ISO date e.g. "2026-03-26"
    start_time:   Optional[datetime]  # UTC-aware if found, else None
    source:       str                 # where we got the time from
    confidence:   str                 # "high" | "medium" | "low"


_WH_SCHEDULE_URL = "https://www.whitehouse.gov/briefing-room/statements-releases/"
_WH_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}

# Month abbreviation → number (matches Kalshi ticker format)
_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5,  "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def _ticker_to_date(event_ticker: str) -> Optional[str]:
    """Extract ISO date string from a Kalshi event ticker e.g. 26MAR26 → 2026-03-26."""
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{2})[A-Z]?(?:-|$)', event_ticker)
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    month = _MONTH_MAP.get(mon)
    if not month:
        return None
    return f"20{yy}-{month:02d}-{int(dd):02d}"


def _scrape_whitehouse_time(event_name: str, event_date: str) -> Optional[datetime]:
    """
    Try to find the exact start time for a presidential event from the
    White House briefing room. Matches by date and event name similarity.
    Returns a UTC-aware datetime or None.
    """
    if not _BS4:
        return None
    try:
        resp = requests.get(_WH_SCHEDULE_URL, headers=_WH_HEADERS, timeout=10)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")

        # Look for time tags near matching event titles
        for article in soup.find_all(["article", "div"], limit=50):
            text = article.get_text(" ", strip=True).lower()
            # Check date match
            if event_date not in text and event_date.replace("-", "/") not in text:
                continue
            # Check name similarity — at least 2 words from event_name appear
            name_words = [w.lower() for w in event_name.split() if len(w) > 3]
            if sum(1 for w in name_words if w in text) < 2:
                continue
            # Try to find a time string like "2:00 PM EST" or "14:00"
            time_match = re.search(
                r'(\d{1,2}:\d{2}\s*(?:AM|PM|am|pm)?(?:\s*[A-Z]{2,4})?)',
                article.get_text()
            )
            if time_match:
                time_str = time_match.group(1).strip()
                try:
                    # Parse and attach date
                    dt = datetime.strptime(f"{event_date} {time_str}", "%Y-%m-%d %I:%M %p")
                    return dt.replace(tzinfo=timezone.utc)
                except ValueError:
                    pass
    except Exception:
        pass
    return None


# Trusted channels for political/presidential event livestreams.
# Values are YouTube @handles — get_channel_id() resolves them to UC... IDs.
_TRUSTED_CHANNELS = [
    "@AssociatedPress",
    "@cspan",
    "@WhiteHouse",
    "@newshour",          # PBS NewsHour
    "@ABCNews",
    "@NBCNews",
]


def _yt_build():
    """Build a YouTube Data API client. Returns None if key not configured."""
    try:
        from googleapiclient.discovery import build as _build
        import os
        # Try youtube_key module first, then env var
        try:
            from youtube_key import YOUTUBE_API_KEY
            key = YOUTUBE_API_KEY
        except ImportError:
            key = os.getenv("YOUTUBE_API_KEY", "")
        if not key:
            return None
        return _build("youtube", "v3", developerKey=key)
    except Exception:
        return None


def _yt_get_channel_id(youtube, identifier: str) -> Optional[str]:
    """Resolve a @handle or channel name to a UC... channel ID."""
    try:
        if identifier.startswith("UC"):
            return identifier
        resp = youtube.search().list(
            part="snippet", q=identifier, type="channel", maxResults=3
        ).execute()
        items = resp.get("items", [])
        return items[0]["snippet"]["channelId"] if items else None
    except Exception:
        return None


def _scrape_youtube_time(event_name: str, event_date: str) -> Optional[datetime]:
    """
    Search trusted YouTube channels (AP, C-SPAN, White House, etc.) for an
    upcoming livestream matching this event and return its scheduledStartTime.

    Uses the official YouTube Data API v3 — requires YOUTUBE_API_KEY env var
    or a youtube_key.py file with YOUTUBE_API_KEY defined.
    Returns a UTC-aware datetime or None.
    """
    youtube = _yt_build()
    if not youtube:
        return None

    # Keywords from event name for title matching (words > 3 chars)
    keywords = [w.lower() for w in event_name.split() if len(w) > 3]

    for handle in _TRUSTED_CHANNELS:
        try:
            channel_id = _yt_get_channel_id(youtube, handle)
            if not channel_id:
                continue

            # Search for upcoming streams on this channel
            search_resp = youtube.search().list(
                part     = "id,snippet",
                channelId = channel_id,
                eventType = "upcoming",
                type      = "video",
                maxResults = 10,
                order     = "date",
            ).execute()

            video_ids = [
                item["id"]["videoId"]
                for item in search_resp.get("items", [])
            ]
            if not video_ids:
                continue

            # Get full video details including liveStreamingDetails
            video_resp = youtube.videos().list(
                part = "snippet,liveStreamingDetails",
                id   = ",".join(video_ids),
            ).execute()

            for item in video_resp.get("items", []):
                title = item["snippet"].get("title", "").lower()

                # Must match the event date
                if event_date not in item["snippet"].get("publishedAt", "") \
                   and event_date not in (
                       item.get("liveStreamingDetails", {})
                           .get("scheduledStartTime", "")
                   ):
                    # Also check title contains at least 2 event keywords
                    if sum(1 for k in keywords if k in title) < 2:
                        continue

                details = item.get("liveStreamingDetails", {})
                scheduled = details.get("scheduledStartTime")
                if scheduled:
                    return datetime.fromisoformat(
                        scheduled.replace("Z", "+00:00")
                    )

        except Exception:
            continue

    return None


def _get_market_open_time(event_ticker: str) -> Optional[datetime]:
    """
    Return the earliest open_time across all markets for this event.

    The Kalshi markets endpoint already includes open_time on every market row —
    no extra scraper needed. The earliest open_time is when trading (and the
    event itself) begins.
    """
    try:
        markets = get_event_markets(event_ticker)
        times = []
        for m in markets:
            # open_time is on the raw API dict, not on KalshiMarket dataclass —
            # re-fetch the raw response to grab it.
            pass
        # Re-fetch raw to access open_time field not stored on KalshiMarket
        for path in ("/markets", "/historical/markets"):
            try:
                data = _get(path, {"event_ticker": event_ticker, "limit": 1000})
                if data.get("markets"):
                    break
            except Exception:
                continue
        else:
            return None

        for m in data.get("markets", []):
            ot = m.get("open_time")
            if ot:
                try:
                    times.append(datetime.fromisoformat(ot.replace("Z", "+00:00")))
                except ValueError:
                    pass
        return min(times) if times else None
    except Exception:
        return None


def get_event_time(event_ticker: str) -> EventTiming:
    """
    Best-effort event start time lookup for a Kalshi event ticker.

    Strategy:
      1. Kalshi market open_time (always available, high confidence)
      2. White House schedule (cross-check for presidential events)
      3. YouTube livestream metadata (fallback for non-WH events)
      4. Return date-only with low confidence if nothing found

    Returns an EventTiming dataclass. start_time is None if we can't find it.
    """
    event_date = _ticker_to_date(event_ticker)

    # Get event name from metadata for matching
    event_name = ""
    try:
        meta = get_event_meta(event_ticker)
        event_name = (
            meta.get("sub_title") or meta.get("subtitle") or meta.get("title") or ""
        ).strip()
    except Exception:
        pass

    if not event_date:
        return EventTiming(
            event_ticker = event_ticker,
            event_date   = None,
            start_time   = None,
            source       = "none",
            confidence   = "low",
        )

    # 1 — Kalshi market open_time (no extra scraper — already in the API response)
    kalshi_time = _get_market_open_time(event_ticker)
    if kalshi_time:
        return EventTiming(
            event_ticker = event_ticker,
            event_date   = event_date,
            start_time   = kalshi_time,
            source       = "kalshi_market",
            confidence   = "high",
        )

    # 2 — White House schedule (best for Trump/presidential events)
    if event_name:
        wh_time = _scrape_whitehouse_time(event_name, event_date)
        if wh_time:
            return EventTiming(
                event_ticker = event_ticker,
                event_date   = event_date,
                start_time   = wh_time,
                source       = "whitehouse.gov",
                confidence   = "high",
            )

    # 3 — YouTube livestream metadata
    if event_name:
        yt_time = _scrape_youtube_time(event_name, event_date)
        if yt_time:
            return EventTiming(
                event_ticker = event_ticker,
                event_date   = event_date,
                start_time   = yt_time,
                source       = "youtube",
                confidence   = "medium",
            )

    # 4 — Date only fallback
    return EventTiming(
        event_ticker = event_ticker,
        event_date   = event_date,
        start_time   = None,
        source       = "ticker_only",
        confidence   = "low",
    )


if __name__ == "__main__":
    import sys
    ticker = sys.argv[1] if len(sys.argv) > 1 else "KXTRUMPMENTION-26MAR26B"
    markets = get_event_markets(ticker, historical=False)
    print(f"\n{ticker}  —  {len(markets)} markets\n")
    print(f"{'WORD':<25} {'STATUS':<11} {'YES_ASK':<9} {'NO_ASK':<9} {'LAST':<7} {'VOL':>10}")
    print("-" * 75)
    for m in markets:
        print(f"{(m.word or '?'):<25} {m.status:<11} "
              f"{m.yes_ask:<9.2f} {m.no_ask:<9.2f} "
              f"{m.last_price:<7.2f} {m.volume:>10,.0f}")
