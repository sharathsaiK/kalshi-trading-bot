"""
news_scraper.py
---------------
"""

from __future__ import annotations

import math
import os
import re
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from typing import Optional
from urllib.parse import quote_plus

import requests

import db

# ---------------------------------------------------------------------------
# HTTP
# ---------------------------------------------------------------------------

_TIMEOUT = 12
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}

# ---------------------------------------------------------------------------
# Source authority scores (0–1). Keyed by lowercase source name or domain.
# ---------------------------------------------------------------------------

_SOURCE_AUTHORITY: dict[str, float] = {
    # Tier 1 — primary newswires / public broadcasters
    "ap": 1.0, "apnews.com": 1.0, "associated press": 1.0,
    "reuters": 1.0, "reuters.com": 1.0,
    "bbc": 0.95, "bbc.com": 0.95, "bbc.co.uk": 0.95, "bbc news": 0.95,
    "nytimes.com": 0.95, "new york times": 0.95,
    "guardian": 0.92, "theguardian.com": 0.92,
    "npr": 0.92, "npr.org": 0.92,
    "pbs": 0.90, "pbs.org": 0.90, "pbs newshour": 0.90,
    "wsj": 0.92, "wall street journal": 0.92,
    "washingtonpost.com": 0.90, "washington post": 0.90,
    "bloomberg": 0.90, "bloomberg.com": 0.90,
    "c-span": 0.88, "cspan.org": 0.88,
    # Tier 2 — major outlets
    "abcnews.go.com": 0.78, "abc news": 0.78,
    "cbsnews.com": 0.78, "cbs news": 0.78,
    "nbcnews.com": 0.78, "nbc news": 0.78,
    "cnn": 0.72, "cnn.com": 0.72,
    "fox news": 0.68, "foxnews.com": 0.68,
    "msnbc": 0.65, "msnbc.com": 0.65,
    "politico": 0.78, "politico.com": 0.78,
    "thehill.com": 0.75, "the hill": 0.75,
    "axios": 0.75, "axios.com": 0.75,
    "the atlantic": 0.75, "theatlantic.com": 0.75,
    "roll call": 0.68, "rollcall.com": 0.68,
    "time": 0.72, "time.com": 0.72,
    # Tier 3 — secondary
    "huffpost": 0.55, "huffpost.com": 0.55,
    "vox": 0.60, "vox.com": 0.60,
    "slate": 0.58, "slate.com": 0.58,
    "breitbart": 0.42, "breitbart.com": 0.42,
    "daily mail": 0.38, "dailymail.co.uk": 0.38,
    "the intercept": 0.55,
    "buzzfeed": 0.45,
    "msn": 0.40, "msn.com": 0.40,
}
_DEFAULT_AUTHORITY = 0.35


def _source_authority(source: str) -> float:
    s = source.strip().lower()
    # Try exact match first, then domain substring match
    if s in _SOURCE_AUTHORITY:
        return _SOURCE_AUTHORITY[s]
    for key, score in _SOURCE_AUTHORITY.items():
        if key in s or s in key:
            return score
    return _DEFAULT_AUTHORITY


# ---------------------------------------------------------------------------
# Article / event type classification
# ---------------------------------------------------------------------------

_OPINION_RE = re.compile(
    r'\b(opinion|op-ed|editorial|commentary|perspective|column|letters? to)\b',
    re.IGNORECASE,
)
_PRESS_RELEASE_RE = re.compile(
    r'\b(press release|announces?|announcement|statement by|fact sheet)\b',
    re.IGNORECASE,
)
_ANALYSIS_RE = re.compile(r'\b(analysis|explainer|fact.?check|in depth)\b', re.IGNORECASE)

_EVENT_TYPE_PATTERNS: list[tuple[re.Pattern, str]] = [
    (re.compile(r'\b(fomc|federal reserve|rate decision|rate hike|rate cut|interest rate)\b', re.I), "fomc"),
    (re.compile(r'\b(sotu|state of the union)\b', re.I), "sotu"),
    (re.compile(r'\b(earnings|quarterly results?|q[1-4] 20\d\d|revenue miss|eps)\b', re.I), "earnings"),
    (re.compile(r'\bdebate\b', re.I), "debate"),
    (re.compile(r'\bpress conference\b|\bpresser\b', re.I), "press_conf"),
    (re.compile(r'\bunited nations\b|\b\bun general assembly\b|\bunga\b', re.I), "un_speech"),
    (re.compile(r'\baddress|speech\b', re.I), "speech"),
]


def classify_article_type(title: str, source: str = "") -> str:
    if _OPINION_RE.search(title):
        return "opinion"
    if _PRESS_RELEASE_RE.search(title):
        return "press_release"
    if _ANALYSIS_RE.search(title):
        return "analysis"
    return "news"


def infer_event_type(event_name: Optional[str], query: str = "") -> str:
    text = f"{event_name or ''} {query}".strip()
    if not text:
        return "speech"
    for pattern, label in _EVENT_TYPE_PATTERNS:
        if pattern.search(text):
            return label
    return "speech"


# ---------------------------------------------------------------------------
# Relevancy scoring
# ---------------------------------------------------------------------------

def compute_relevancy(
    title: str,
    source: str,
    published_at: Optional[str],
    speaker: str,
    word: Optional[str] = None,
    event_name: Optional[str] = None,
) -> float:
    """
    Return a relevancy score in [0, 1] for a news article relative to
    the (speaker, word, event) context.
    """
    # Recency: half-life of 7 days
    recency = 0.5  # default when date unknown
    if published_at:
        try:
            pub = datetime.fromisoformat(published_at.rstrip("Z")).replace(tzinfo=timezone.utc)
            age_days = max(0.0, (datetime.now(tz=timezone.utc) - pub).total_seconds() / 86400)
            recency = math.exp(-age_days / 7.0)
        except ValueError:
            pass

    authority = _source_authority(source)

    tl = title.lower()
    surname = speaker.strip().split()[-1].lower()
    speaker_match = 1.0 if surname in tl else 0.0

    word_match = 0.0
    if word:
        word_match = 1.0 if word.lower() in tl else 0.0

    event_match = 0.0
    if event_name:
        keywords = [w for w in event_name.lower().split() if len(w) > 3]
        if keywords:
            hits = sum(1 for k in keywords if k in tl)
            event_match = hits / len(keywords)

    score = (
        0.25 * recency
        + 0.20 * authority
        + 0.25 * speaker_match
        + 0.20 * word_match
        + 0.10 * event_match
    )
    return round(min(1.0, max(0.0, score)), 4)


def aggregate_relevancy_features(articles: list[dict]) -> dict:
    """
    Aggregate per-article relevancy scores into LightGBM feature dict.

    Keys: rel_max, rel_mean, rel_top3_mean, rel_count_hi (>= 0.5), rel_n
    """
    scores = [a["relevancy"] for a in articles if a.get("relevancy") is not None]
    if not scores:
        return {"rel_max": 0.0, "rel_mean": 0.0, "rel_top3_mean": 0.0,
                "rel_count_hi": 0, "rel_n": 0}
    top3 = sorted(scores, reverse=True)[:3]
    return {
        "rel_max":      round(max(scores), 4),
        "rel_mean":     round(sum(scores) / len(scores), 4),
        "rel_top3_mean": round(sum(top3) / len(top3), 4),
        "rel_count_hi": sum(1 for s in scores if s >= 0.5),
        "rel_n":        len(scores),
    }


# ---------------------------------------------------------------------------
# Query builder (unchanged logic, kept here for imports)
# ---------------------------------------------------------------------------

_DATE_ONLY_RE = re.compile(
    r'^\s*(on|before|after|by)?\s*'
    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'
    r'\s+\d{1,2},?\s+\d{4}\s*$',
    re.IGNORECASE,
)


def build_query(
    speaker: str,
    event_name: Optional[str] = None,
    word: Optional[str] = None,
) -> str:
    parts: list[str] = [speaker.strip().split()[-1].lower()]
    if event_name and not _DATE_ONLY_RE.match(event_name):
        words = [w for w in event_name.split() if len(w) > 2][:3]
        parts.extend(w.lower() for w in words)
    if word:
        w = word.strip().lower()
        if w and w not in parts:
            parts.append(w)
    return " ".join(parts)


# ---------------------------------------------------------------------------
# RSS parser (shared by Google News + Bing)
# ---------------------------------------------------------------------------

def _parse_pub_date(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        return parsedate_to_datetime(raw.strip())
    except Exception:
        return None


def _parse_rss(
    content: bytes,
    query: str,
    cutoff: datetime,
    max_results: int,
) -> list[dict]:
    try:
        root = ET.fromstring(content)
    except ET.ParseError:
        return []
    channel = root.find("channel")
    if channel is None:
        return []

    articles: list[dict] = []
    for item in channel.findall("item"):
        title_el = item.find("title")
        link_el  = item.find("link")
        if title_el is None or link_el is None:
            continue
        title = (title_el.text or "").strip()
        url   = (link_el.text or "").strip()
        if not title or not url:
            continue

        pub_el   = item.find("pubDate")
        pub_date = _parse_pub_date(pub_el.text if pub_el is not None else None)
        if pub_date is not None and pub_date < cutoff:
            continue

        src_el = item.find("source")
        source = src_el.text.strip() if src_el is not None and src_el.text else ""
        pub_str = pub_date.isoformat() if pub_date else None

        articles.append({
            "query": query, "title": title, "url": url,
            "published_at": pub_str, "source": source, "snippet": "",
        })
        if len(articles) >= max_results:
            break
    return articles


# ---------------------------------------------------------------------------
# Source adapters
# ---------------------------------------------------------------------------

_GOOGLE_RSS = "https://news.google.com/rss/search"
_BING_RSS   = "https://www.bing.com/news/search"
_GDELT_URL  = "https://api.gdeltproject.org/api/v2/doc/doc"
_NYT_URL    = "https://api.nytimes.com/svc/search/v2/articlesearch.json"

# Google/Bing are limited to ~14 days regardless of what we ask.
# Flag them so callers can skip them when doing historical backfills.
_LIMITED_WINDOW_SOURCES = {"google_news", "bing_news"}


def _fetch_google_news(
    query: str, date_from: datetime, date_to: datetime, max_results: int
) -> list[dict]:
    # Google News RSS ignores explicit date ranges — only serves recent ~14 days.
    params = {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}
    try:
        resp = requests.get(_GOOGLE_RSS, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[news_scraper/google] {exc}")
        return []
    return _parse_rss(resp.content, query, date_from, max_results)


def _fetch_bing_news(
    query: str, date_from: datetime, date_to: datetime, max_results: int
) -> list[dict]:
    # Bing News RSS also caps at ~14 days; no date param accepted.
    params = {"q": query, "format": "rss"}
    try:
        resp = requests.get(_BING_RSS, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
    except requests.RequestException as exc:
        print(f"[news_scraper/bing] {exc}")
        return []
    articles = _parse_rss(resp.content, query, date_from, max_results)
    for a in articles:
        if not a["source"]:
            try:
                from urllib.parse import urlparse
                a["source"] = urlparse(a["url"]).netloc.replace("www.", "")
            except Exception:
                pass
    return articles


_GDELT_TIMEOUT = 30   # GDELT responds slowly under load; 12s was timing out often
_GDELT_RETRIES = 5    # more patience — was giving up too fast on 429 and timeouts


def _gdelt_request(params: dict, retries: int = _GDELT_RETRIES) -> Optional[dict]:
    """GET GDELT with exponential backoff on 429 AND on timeout (previously
    timeouts gave up immediately with no retry — only 429s were retried)."""
    for attempt in range(retries):
        try:
            resp = requests.get(_GDELT_URL, params=params, headers=_HEADERS,
                                 timeout=_GDELT_TIMEOUT)
            if resp.status_code == 429:
                wait = 2 ** attempt
                print(f"[news_scraper/gdelt] rate limited, retrying in {wait}s…")
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.Timeout:
            wait = 2 ** attempt
            print(f"[news_scraper/gdelt] timed out, retrying in {wait}s…")
            time.sleep(wait)
            continue
        except requests.RequestException as exc:
            print(f"[news_scraper/gdelt] {exc}")
            return None
    print("[news_scraper/gdelt] giving up after retries")
    return None


def _fetch_gdelt(
    query: str, date_from: datetime, date_to: datetime, max_results: int
) -> list[dict]:
    """GDELT GKG — historical depth from 2013, full timestamps."""
    params = {
        "query":         query,
        "mode":          "ArtList",
        "maxrecords":    str(min(max_results * 2, 250)),
        "format":        "json",
        "sort":          "DateDesc",
        "startdatetime": date_from.strftime("%Y%m%d%H%M%S"),
        "enddatetime":   date_to.strftime("%Y%m%d%H%M%S"),
    }
    data = _gdelt_request(params)
    if not data:
        return []

    articles: list[dict] = []
    for art in (data.get("articles") or []):
        title = (art.get("title") or "").strip()
        url   = (art.get("url") or "").strip()
        if not title or not url:
            continue

        pub_str: Optional[str] = None
        raw_date = art.get("seendate") or art.get("pubdate") or ""
        try:
            pub = datetime.strptime(raw_date[:15], "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
            if pub < date_from or pub > date_to:
                continue
            pub_str = pub.isoformat()
        except ValueError:
            pass

        source = art.get("domain") or ""
        articles.append({
            "query": query, "title": title, "url": url,
            "published_at": pub_str, "source": source, "snippet": "",
        })
        if len(articles) >= max_results:
            break
    return articles


def _fetch_guardian(
    query: str, date_from: datetime, date_to: datetime, max_results: int
) -> list[dict]:
    """Guardian Open Platform — historical from 1999. Requires GUARDIAN_API_KEY."""
    api_key = os.getenv("GUARDIAN_API_KEY")
    if not api_key:
        return []
    params = {
        "q":           query,
        "api-key":     api_key,
        "from-date":   date_from.strftime("%Y-%m-%d"),
        "to-date":     date_to.strftime("%Y-%m-%d"),
        "page-size":   min(max_results, 50),
        "order-by":    "newest",
        "show-fields": "headline,trailText",
    }
    try:
        resp = requests.get(
            "https://content.guardianapis.com/search",
            params=params, headers=_HEADERS, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("response", {}).get("results", [])
    except Exception as exc:
        print(f"[news_scraper/guardian] {exc}")
        return []

    articles: list[dict] = []
    for r in results:
        title = r.get("webTitle") or (r.get("fields") or {}).get("headline") or ""
        url   = r.get("webUrl") or ""
        if not title or not url:
            continue
        pub_str: Optional[str] = None
        raw = r.get("webPublicationDate") or ""
        try:
            pub = datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=timezone.utc)
            pub_str = pub.isoformat()
        except ValueError:
            pass
        snippet = (r.get("fields") or {}).get("trailText") or ""
        articles.append({
            "query": query, "title": title, "url": url,
            "published_at": pub_str, "source": "The Guardian", "snippet": snippet,
        })
    return articles


def _fetch_nyt(
    query: str, date_from: datetime, date_to: datetime, max_results: int
) -> list[dict]:
    """NYT Article Search — historical from 1851. Requires NYT_API_KEY."""
    api_key = os.getenv("NYT_API_KEY")
    if not api_key:
        return []
    params = {
        "q":          query,
        "api-key":    api_key,
        "begin_date": date_from.strftime("%Y%m%d"),
        "end_date":   date_to.strftime("%Y%m%d"),
        "sort":       "newest",
        "page":       0,
    }
    try:
        resp = requests.get(_NYT_URL, params=params, headers=_HEADERS, timeout=_TIMEOUT)
        resp.raise_for_status()
        docs = resp.json().get("response", {}).get("docs", [])
    except Exception as exc:
        print(f"[news_scraper/nyt] {exc}")
        return []

    articles: list[dict] = []
    for doc in docs[:max_results]:
        title = (doc.get("headline") or {}).get("main") or ""
        url   = doc.get("web_url") or ""
        if not title or not url:
            continue
        pub_str: Optional[str] = None
        raw = doc.get("pub_date") or ""
        try:
            pub = datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=timezone.utc)
            pub_str = pub.isoformat()
        except ValueError:
            pass
        snippet = doc.get("abstract") or doc.get("snippet") or ""
        articles.append({
            "query": query, "title": title, "url": url,
            "published_at": pub_str, "source": "The New York Times", "snippet": snippet,
        })
    return articles


def _fetch_newsapi(
    query: str, date_from: datetime, date_to: datetime, max_results: int
) -> list[dict]:
    """NewsAPI — free tier: 30 days / paid: full history. Requires NEWS_API_KEY."""
    api_key = os.getenv("NEWS_API_KEY")
    if not api_key:
        return []
    params = {
        "q":        query,
        "from":     date_from.strftime("%Y-%m-%dT%H:%M:%S"),
        "to":       date_to.strftime("%Y-%m-%dT%H:%M:%S"),
        "sortBy":   "publishedAt",
        "language": "en",
        "pageSize": min(max_results, 100),
        "apiKey":   api_key,
    }
    try:
        resp = requests.get(
            "https://newsapi.org/v2/everything",
            params=params, headers=_HEADERS, timeout=_TIMEOUT,
        )
        resp.raise_for_status()
        results = resp.json().get("articles") or []
    except Exception as exc:
        print(f"[news_scraper/newsapi] {exc}")
        return []

    articles: list[dict] = []
    for r in results:
        title = (r.get("title") or "").strip()
        url   = (r.get("url") or "").strip()
        if not title or not url or title == "[Removed]":
            continue
        pub_str: Optional[str] = None
        raw = r.get("publishedAt") or ""
        try:
            pub = datetime.fromisoformat(raw.rstrip("Z")).replace(tzinfo=timezone.utc)
            pub_str = pub.isoformat()
        except ValueError:
            pass
        source = ((r.get("source") or {}).get("name") or "").strip()
        snippet = (r.get("description") or "").strip()
        articles.append({
            "query": query, "title": title, "url": url,
            "published_at": pub_str, "source": source, "snippet": snippet,
        })
    return articles


# Adapter registry. Sources marked in _LIMITED_WINDOW_SOURCES are skipped
# automatically when historical=True is passed to fetch_news().
_ADAPTERS = [
    ("gdelt",       _fetch_gdelt),       # 2013–present, free, no key
    ("guardian",    _fetch_guardian),    # 1999–present, free key (GUARDIAN_API_KEY)
    ("nyt",         _fetch_nyt),         # 1851–present, free key (NYT_API_KEY)
    ("newsapi",     _fetch_newsapi),     # 30d free / full paid (NEWS_API_KEY)
    ("google_news", _fetch_google_news), # ~14d only
    ("bing_news",   _fetch_bing_news),   # ~14d only
]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def fetch_news(
    speaker: str,
    event_name: Optional[str] = None,
    word: Optional[str] = None,
    max_age_days: int = 14,
    max_results: int = 20,
    persist: bool = True,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> list[dict]:
    """
    Fetch news from all configured sources, score each article, deduplicate
    by URL, and return sorted by relevancy descending.

    date_from / date_to  — explicit UTC window for historical backfills.
                           When supplied, sources limited to ~14 days
                           (Google News, Bing) are automatically skipped.
                           When omitted, defaults to (now - max_age_days, now).

    Each returned dict contains: query, title, url, published_at, source,
    snippet, relevancy, event_type, article_type.
    """
    query = build_query(speaker, event_name, word)
    now   = datetime.now(tz=timezone.utc)

    if date_from is None:
        date_from = now - timedelta(days=max_age_days)
    if date_to is None:
        date_to = now

    # Historical query: skip sources that can't serve past ~14 days
    historical = (now - date_to).days > 14
    active = [
        (name, fn) for name, fn in _ADAPTERS
        if not (historical and name in _LIMITED_WINDOW_SOURCES)
    ]

    et = infer_event_type(event_name, query)
    per_source = max(max_results, 15)

    # Fan out to all active adapters concurrently
    raw: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(active)) as pool:
        futures = {
            pool.submit(fn, query, date_from, date_to, per_source): name
            for name, fn in active
        }
        for fut in as_completed(futures):
            try:
                raw.extend(fut.result())
            except Exception as exc:
                print(f"[news_scraper/{futures[fut]}] crashed: {exc}")

    # Deduplicate by URL (keep first seen)
    seen: set[str] = set()
    deduped: list[dict] = []
    for a in raw:
        if a["url"] not in seen:
            seen.add(a["url"])
            deduped.append(a)

    # Score + classify
    for a in deduped:
        a["event_type"]   = et
        a["article_type"] = classify_article_type(a["title"], a["source"])
        a["relevancy"]    = compute_relevancy(
            title=a["title"],
            source=a["source"],
            published_at=a["published_at"],
            speaker=speaker,
            word=word,
            event_name=event_name,
        )

    # Sort best-first, cap
    deduped.sort(key=lambda a: a["relevancy"], reverse=True)
    deduped = deduped[:max_results]

    if persist:
        for a in deduped:
            db.insert_news_cache(
                query=a["query"],
                title=a["title"],
                url=a["url"],
                published_at=a["published_at"],
                source=a["source"],
                snippet=a["snippet"],
                relevancy=a["relevancy"],
                event_type=a["event_type"],
                article_type=a["article_type"],
            )

    return deduped


def fetch_news_for_words(
    speaker: str,
    words: list[str],
    event_name: Optional[str] = None,
    max_age_days: int = 14,
    max_results_per_word: int = 10,
    date_from: Optional[datetime] = None,
    date_to: Optional[datetime] = None,
) -> dict[str, list[dict]]:
    """
    Fetch and score news for each target word in parallel.
    Returns word → list[article_dict].
    Pass date_from/date_to for historical backfills.
    """
    def _fetch_one(word: str):
        return word, fetch_news(
            speaker=speaker,
            event_name=event_name,
            word=word,
            max_age_days=max_age_days,
            max_results=max_results_per_word,
            persist=True,
            date_from=date_from,
            date_to=date_to,
        )

    results: dict[str, list[dict]] = {}
    with ThreadPoolExecutor(max_workers=min(len(words), 8)) as pool:
        futures = {pool.submit(_fetch_one, w): w for w in words}
        for fut in as_completed(futures):
            word, articles = fut.result()
            results[word] = articles
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python news_scraper.py <speaker> [event_name] [word]")
        print('  e.g. python news_scraper.py "Donald Trump" "SOTU" "tariff"')
        sys.exit(1)

    speaker_arg = sys.argv[1]
    event_arg   = sys.argv[2] if len(sys.argv) > 2 else None
    word_arg    = sys.argv[3] if len(sys.argv) > 3 else None

    arts = fetch_news(speaker_arg, event_name=event_arg, word=word_arg)
    q = build_query(speaker_arg, event_arg, word_arg)
    print(f"\nQuery: {q!r}  →  {len(arts)} article(s)\n")
    for a in arts:
        pub = a["published_at"] or "unknown date"
        rel = a.get("relevancy", "?")
        at  = a.get("article_type", "")
        print(f"  [{a['source'] or '?'}] [{at}] (rel={rel}) {a['title']}")
        print(f"    {pub}")
        print()
