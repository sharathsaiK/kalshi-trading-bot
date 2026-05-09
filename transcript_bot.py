"""
transcript_bot.py
-----------------
Transcript scraper bot.
"""

from __future__ import annotations

import hashlib
import logging
import os
import re
import sys
import threading
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError as _FuturesTimeout
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Callable, Optional
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1.  QUERY SCHEMA
# ---------------------------------------------------------------------------

@dataclass
class TranscriptQuery:
    """
    Required
    --------
    speaker_name : str

    Optional — narrow the search
    ----------------------------
    event_name    : str | None            # "Q4 2024 Earnings Call", "FOMC Dec 2024"
    event_type    : str | None            # earnings_call, speech, press_conference,
                                          # congressional_hearing, debate, interview,
                                          # podcast, conference
    date_from     : date | None
    date_to       : date | None
    topic_hint    : str | None
    ticker        : str | None            # stock ticker for earnings_call
    organization  : str | None            # "Federal Reserve", "White House" — disambiguation
    speaker_role  : str | None            # "Fed Chair", "CEO", "President"
    language      : str                   # ISO code, default "en"
    min_length    : int                   # min chars for a result to be kept
    exclude_sources: list[str]            # adapter labels to skip
    max_results   : int
    """
    speaker_name: str
    event_name: Optional[str] = None
    event_type: Optional[str] = None
    date_from: Optional[date] = None
    date_to: Optional[date] = None
    topic_hint: Optional[str] = None
    ticker: Optional[str] = None
    organization: Optional[str] = None
    speaker_role: Optional[str] = None
    language: str = "en"
    min_length: int = 200
    exclude_sources: list[str] = field(default_factory=list)
    max_results: int = 20
    strict_date: bool = False  # if True, never fall back to undated results
                               # when date_from/date_to are set
    timeout: Optional[float] = None  # per-call override; falls back to TRANSCRIPT_BOT_TIMEOUT env var


# ---------------------------------------------------------------------------
# 2.  RESULT SCHEMA
# ---------------------------------------------------------------------------

@dataclass
class TranscriptResult:
    source: str
    source_url: str
    content_hash: str
    full_text: str

    speaker_name: Optional[str] = None
    event_name: Optional[str] = None
    event_date: Optional[str] = None
    event_type: Optional[str] = None
    title: Optional[str] = None
    description: Optional[str] = None
    language: str = "en"
    match_confidence: float = 1.0            # 0.0 - 1.0 adapter-reported

    fetched_at: str = field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    scrape_notes: Optional[str] = None


def _hash(text: str) -> str:
    return hashlib.sha256(text.strip().encode()).hexdigest()


# ---------------------------------------------------------------------------
# 3.  HTTP:  per-host rate limit + retry/backoff
# ---------------------------------------------------------------------------

_SESSION = requests.Session()
_DEFAULT_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
_SESSION.headers.update({
    "User-Agent": _DEFAULT_UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
})

# SEC blocks anything without a contact-style UA.
_SEC_UA = "TranscriptBot research-pipeline contact@example.com"

_HOST_DELAY = 1.0
_host_locks: dict[str, threading.Lock] = {}
_host_last_hit: dict[str, float] = {}
_locks_lock = threading.Lock()


def _host_lock(host: str) -> threading.Lock:
    with _locks_lock:
        lock = _host_locks.get(host)
        if lock is None:
            lock = threading.Lock()
            _host_locks[host] = lock
        return lock


def _get(
    url: str,
    params: dict | None = None,
    headers: dict | None = None,
    timeout: int = 15,
    max_attempts: int = 3,
) -> requests.Response:
    host = urlparse(url).netloc
    lock = _host_lock(host)
    attempt = 0
    while True:
        attempt += 1
        with lock:
            wait = _HOST_DELAY - (time.time() - _host_last_hit.get(host, 0))
            if wait > 0:
                time.sleep(wait)
            _host_last_hit[host] = time.time()
        try:
            resp = _SESSION.get(url, params=params, headers=headers, timeout=timeout)
            resp.raise_for_status()
            return resp
        except (requests.HTTPError, requests.ConnectionError, requests.Timeout) as exc:
            if attempt >= max_attempts:
                raise
            time.sleep(1.5 ** attempt)
            logger.debug("retry %s (%s)", url, exc)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

_NAV_TOKENS = ("privacy policy", "terms of service", "cookie", "subscribe", "sign up")


def _is_meaningful_transcript(text: str, min_len: int) -> bool:
    if not text or len(text) < min_len:
        return False
    non_empty_lines = [l for l in text.split("\n") if l.strip()]
    if len(non_empty_lines) < 5:
        return False
    lower = text.lower()
    nav_hits = sum(1 for t in _NAV_TOKENS if t in lower)
    if nav_hits >= 3 and len(text) < 2000:
        return False
    return True


def _speaker_matches(haystack: str, speaker: str) -> float:
    """0..1 confidence that `haystack` identifies `speaker` as the speaker.

    Uses mention frequency so a single incidental name-drop in a news article
    about someone else doesn't score the same as a proper speech transcript
    where the speaker label appears dozens of times.
    """
    if not haystack or not speaker:
        return 0.0
    h = haystack.lower()
    parts = [p for p in speaker.lower().split() if len(p) > 1]
    if not parts:
        return 0.0
    full = " ".join(parts)
    surname = parts[-1]

    full_count = h.count(full)
    surname_count = h.count(surname)

    if full_count >= 3 or surname_count >= 6:
        return 1.0
    if full_count >= 1:
        return 0.7
    if surname_count >= 2:
        return 0.6
    if surname_count >= 1:
        return 0.4
    return 0.0


def _parse_year(text: str) -> Optional[int]:
    m = re.search(r"\b(19|20)\d{2}\b", text or "")
    return int(m.group()) if m else None


def _extract_date_from_text(text: str) -> Optional[str]:
    patterns = (
        r"\b\d{4}-Q[1-4]\b",
        r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},?\s+\d{4}\b",
        r"\b\d{4}-\d{2}-\d{2}\b",
        r"\b(?:Q[1-4])\s+\d{4}\b",
    )
    for pat in patterns:
        m = re.search(pat, text or "", re.IGNORECASE)
        if m:
            return m.group()
    # URL path dates: /YYYY/MM/DD/ — common in whitehouse.gov, WH archive, etc.
    m = re.search(r"/(\d{4})/(\d{2})/(\d{2})/", text or "")
    if m:
        return f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
    return None


def _best_date(*sources: Optional[str]) -> Optional[str]:
    """First hit wins; lets callers pass title then body as fallbacks."""
    for s in sources:
        if not s:
            continue
        d = _extract_date_from_text(s)
        if d:
            return d
    return None


# ---------------------------------------------------------------------------
# Normalization — runs after adapters, before dedup.
#
# Goal: make the same transcript from two different sources hash to the same
# value (so _deduplicate catches it), and make `event_date` / `title` consistent
# across sources.  full_text is left untouched — downstream (EV calc, word
# counting) wants the raw source.
# ---------------------------------------------------------------------------

# Common mojibake artifacts seen in scraped content (UTF-8 bytes misread as
# Windows-1252).  Example seen in the wild: "Fed's" arriving as "Fedâs".
_MOJIBAKE_FIXES = {
    "â€™": "'", "â€˜": "'",
    "â€œ": '"', "â€\x9d": '"', "â€": '"',
    "â€”": "-", "â€“": "-",
    "Â ": " ",
}

# Unicode smart punctuation → ASCII equivalents.
_PUNCT_TRANSLATE = {
    0x2018: 0x27, 0x2019: 0x27,      # ‘ ’  → '
    0x201C: 0x22, 0x201D: 0x22,      # “ ”  → "
    0x2013: 0x2D, 0x2014: 0x2D,      # – —  → -
    0x00A0: 0x20,                    # nbsp → space
}


def _normalize_for_hash(text: str) -> str:
    """Aggressive normalization used ONLY to compute the dedup hash."""
    if not text:
        return ""
    for bad, good in _MOJIBAKE_FIXES.items():
        text = text.replace(bad, good)
    text = text.translate(_PUNCT_TRANSLATE)
    text = text.lower()
    # ASCII-only: drops any residual mojibake / accented chars.  Aggressive
    # but fine for dedup since we only compare hashes here.
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


_MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}


def _normalize_date(d: Optional[str]) -> Optional[str]:
    """Return ISO-8601 `YYYY-MM-DD` or quarter `YYYY-Qn`.  Leaves unknown
    formats unchanged rather than discarding them."""
    if not d:
        return d
    d = d.strip()
    if re.match(r"^\d{4}-\d{2}-\d{2}$", d):
        return d
    if re.match(r"^\d{4}-Q[1-4]$", d):
        return d
    m = re.match(r"^Q([1-4])\s+(\d{4})$", d, re.IGNORECASE)
    if m:
        return f"{m.group(2)}-Q{m.group(1)}"
    m = re.match(r"^([A-Za-z]+)\s+(\d{1,2}),?\s+(\d{4})$", d)
    if m:
        mon = _MONTHS.get(m.group(1)[:3].lower())
        if mon:
            return f"{m.group(3)}-{mon:02d}-{int(m.group(2)):02d}"
    return d


_SITE_SUFFIX_PAT = re.compile(
    r"\s*[\|\-\u2013\u2014]\s*"
    r"(rev\.com|rev|miller center|the white house|"
    r"federal reserve(?: board)?|c-span|youtube|sec\.gov|govinfo)"
    r"\s*$",
    re.IGNORECASE,
)


def _normalize_title(title: Optional[str]) -> Optional[str]:
    if not title:
        return title
    cleaned = _SITE_SUFFIX_PAT.sub("", title)
    cleaned = " ".join(cleaned.split()).strip()
    return cleaned or None


def _normalize_result(r: TranscriptResult) -> TranscriptResult:
    """Normalize metadata and recompute content_hash.  full_text unchanged."""
    r.title = _normalize_title(r.title)
    r.event_date = _normalize_date(r.event_date)
    r.content_hash = hashlib.sha256(
        _normalize_for_hash(r.full_text).encode()
    ).hexdigest()
    return r


# ---------------------------------------------------------------------------
# 4.  ADAPTERS
#
# Contract:  def fetch(query) -> list[TranscriptResult]
#            Must never raise.  Log at WARNING and return [] on errors.
# ---------------------------------------------------------------------------


# ── 4a.  Rev.com public transcript library ─────────────────────────────────
def _fetch_rev(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = query.event_name or query.speaker_name
    if query.event_name and query.speaker_name not in term:
        term = f"{query.speaker_name} {term}"
    if query.topic_hint:
        term += f" {query.topic_hint}"
    if query.date_from:
        term += f" {query.date_from.strftime('%B %Y')}"

    # Try both the transcripts index and the blog search — Rev has migrated
    # content between these and we want to be tolerant.
    search_urls = [
        ("https://www.rev.com/transcripts", {"q": term}),
        ("https://www.rev.com/blog", {"s": term}),
    ]

    seen_hrefs: set[str] = set()
    for search_url, params in search_urls:
        try:
            resp = _get(search_url, params=params)
        except Exception as exc:
            logger.warning("rev: search failed (%s) — %s", search_url, exc)
            continue

        soup = BeautifulSoup(resp.text, "html.parser")
        anchors = soup.select("a[href*='/transcripts/'], a[href*='/blog/transcripts/']")
        for a in anchors[:20]:
            href = a.get("href", "")
            if not href or href in seen_hrefs or href.rstrip("/") in (
                "/transcripts", "/blog/transcripts"
            ):
                continue
            seen_hrefs.add(href)
            full_url = urljoin("https://www.rev.com", href)
            try:
                page = _get(full_url)
                psoup = BeautifulSoup(page.text, "html.parser")
                title_tag = psoup.find("h1")
                title = title_tag.get_text(strip=True) if title_tag else None
                body = (
                    psoup.find("div", class_=re.compile(
                        r"fl-rich-text|transcript|entry-content|post-content", re.I))
                    or psoup.find("article")
                    or psoup.find("main")
                )
                if not body:
                    continue
                text = body.get_text(separator="\n", strip=True)
                if not _is_meaningful_transcript(text, query.min_length):
                    continue
                conf = _speaker_matches(
                    f"{title or ''} {text[:3000]}", query.speaker_name)
                if conf == 0.0:
                    continue
                results.append(TranscriptResult(
                    source="rev",
                    source_url=full_url,
                    content_hash=_hash(text),
                    full_text=text,
                    speaker_name=query.speaker_name,
                    event_name=query.event_name or title,
                    event_date=_best_date(title, text[:500]),
                    event_type=query.event_type,
                    title=title,
                    match_confidence=conf,
                ))
            except Exception as exc:
                logger.warning("rev: failed %s — %s", full_url, exc)
    return results


# ── 4b.  GovInfo API — replaces the scrape-based congress.gov adapter ──────
def _fetch_govinfo(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    api_key = os.getenv("GOVINFO_API_KEY")
    if not api_key:
        logger.debug("govinfo: no GOVINFO_API_KEY, skipping")
        return results

    congressional = {"congressional_hearing", "speech", "debate", "press_conference", None}
    if query.event_type not in congressional:
        return results

    q_text = query.speaker_name
    if query.topic_hint:
        q_text += f" {query.topic_hint}"
    if query.event_name:
        q_text += f" {query.event_name}"

    try:
        body = {
            "query": q_text,
            "pageSize": 10,
            "offsetMark": "*",
            "sorts": [{"field": "relevancy", "sortOrder": "DESC"}],
            "resultLevel": "default",
        }
        resp = _SESSION.post(
            "https://api.govinfo.gov/search",
            json=body,
            headers={"X-Api-Key": api_key, "Content-Type": "application/json"},
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("govinfo: search failed — %s", exc)
        return results

    for item in (data.get("results") or [])[:10]:
        package_id = item.get("packageId") or ""
        title = item.get("title")
        pkg_url = (
            item.get("packageLink")
            or f"https://www.govinfo.gov/app/details/{package_id}"
        )
        download = item.get("download") or {}
        txt_link = download.get("txtLink") if isinstance(download, dict) else None
        full_text = ""
        if txt_link:
            try:
                txt = _SESSION.get(
                    txt_link,
                    params={"api_key": api_key},
                    timeout=20,
                )
                txt.raise_for_status()
                full_text = txt.text
            except Exception as exc:
                logger.debug("govinfo: txt fetch failed — %s", exc)
        if not full_text:
            full_text = item.get("summary") or ""
        if not _is_meaningful_transcript(full_text, query.min_length):
            continue
        conf = _speaker_matches(
            f"{title or ''} {full_text[:3000]}", query.speaker_name)
        if conf == 0.0:
            continue
        results.append(TranscriptResult(
            source="govinfo",
            source_url=pkg_url,
            content_hash=_hash(full_text),
            full_text=full_text,
            speaker_name=query.speaker_name,
            event_name=query.event_name or title,
            event_date=item.get("dateIssued") or _best_date(title, full_text[:500]),
            event_type=query.event_type or "congressional_hearing",
            title=title,
            match_confidence=conf,
        ))
    return results


# ── 4c.  Miller Center — presidential speeches ─────────────────────────────
_PRESIDENTS = {
    "biden", "trump", "obama", "bush", "clinton", "reagan", "carter",
    "ford", "nixon", "johnson", "kennedy", "roosevelt", "eisenhower", "truman",
}


def _fetch_miller_center(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    parts = query.speaker_name.strip().split()
    if not parts:
        return results
    last = parts[-1].lower()
    if last not in _PRESIDENTS:
        return results
    if query.event_type not in (None, "speech", "press_conference", "debate", "interview"):
        return results

    # Miller Center speech URLs are date-based ("/presidential-speeches/
    # november-4-2008-remarks-election-night"), so the surname isn't in the
    # slug.  Use the per-president profile page — it lists that president's
    # speeches only — as the source of truth.
    profile_url = f"https://millercenter.org/president/{last}"
    try:
        resp = _get(profile_url)
    except Exception as exc:
        logger.warning("miller_center: profile %s failed — %s", profile_url, exc)
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.select("a[href*='/the-presidency/presidential-speeches/']"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or not title or href in seen:
            continue
        seen.add(href)
        candidates.append((href, title))

    # If caller gave a date window, restrict to speeches from those years only.
    allowed_years: set[str] = set()
    if query.date_from and query.date_to:
        for yr in range(query.date_from.year, query.date_to.year + 1):
            allowed_years.add(str(yr))
        candidates = [
            (h, t) for h, t in candidates
            if any(yr in h or yr in t for yr in allowed_years)
        ]

    for href, link_title in candidates[:8]:
        full_url = urljoin("https://millercenter.org", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")

            # Verify the page credits this speaker.
            byline_el = psoup.find(class_=re.compile(r"president|speaker|byline", re.I))
            byline = byline_el.get_text(" ", strip=True) if byline_el else ""
            conf = _speaker_matches(f"{byline} {link_title}", query.speaker_name)
            if conf == 0.0:
                # We got here from the president's own profile page, so the
                # provenance is strong even if the page's byline doesn't hit.
                conf = 0.8

            transcript_div = (
                psoup.find("div", class_=re.compile(
                    r"transcript|speech-text|view-transcript", re.I))
                or psoup.find("div", {"id": re.compile(r"transcript", re.I)})
            )
            if not transcript_div:
                continue
            text = transcript_div.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue

            results.append(TranscriptResult(
                source="miller_center",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or link_title,
                event_date=_best_date(link_title, byline, text[:500]),
                event_type=query.event_type or "speech",
                title=link_title,
                match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("miller_center: failed %s — %s", full_url, exc)
    return results


# ── 4d.  Federal Reserve speeches ──────────────────────────────────────────
_FED_SPEAKERS = {
    "powell", "yellen", "bernanke", "williams", "waller", "jefferson",
    "barr", "cook", "bowman", "kugler",
}


def _fetch_federal_reserve(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    parts = query.speaker_name.strip().split()
    last = parts[-1].lower() if parts else ""
    is_fed = (
        last in _FED_SPEAKERS
        or (query.organization and "federal reserve" in query.organization.lower())
    )
    if not is_fed:
        return results

    years = (
        list(range(query.date_from.year, query.date_to.year + 1))
        if query.date_from and query.date_to
        else [date.today().year, date.today().year - 1]
    )

    speech_links: list[tuple[str, str]] = []
    seen_hrefs: set[str] = set()
    # Fed filenames follow a fixed convention: <surname><YYYYMMDD><letter>.htm
    # That's a tighter filter than row-level text matching, which was catching
    # other speakers whose rows shared an HTML container.
    href_pat = re.compile(
        rf"/newsevents/speech/{re.escape(last)}\d{{8}}[a-z]?\.htm$", re.I)
    for y in years[:3]:
        index_url = f"https://www.federalreserve.gov/newsevents/speech/{y}-speeches.htm"
        try:
            resp = _get(index_url)
        except Exception as exc:
            logger.debug("federal_reserve: %s index failed — %s", y, exc)
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=href_pat):
            href = a["href"]
            if href in seen_hrefs:
                continue
            seen_hrefs.add(href)
            speech_links.append((href, a.get_text(strip=True)))

    for href, title in speech_links[:8]:
        full_url = urljoin("https://www.federalreserve.gov", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            content = (
                psoup.find("div", id="article")
                or psoup.find("div", class_=re.compile(r"col-(?:xs|sm|md)-\d+"))
            )
            if not content:
                continue
            text = content.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            results.append(TranscriptResult(
                source="federal_reserve",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=1.0,
            ))
        except Exception as exc:
            logger.warning("federal_reserve: failed %s — %s", full_url, exc)
    return results


# ── 4e.  White House archives ──────────────────────────────────────────────
_WH_ARCHIVES = {
    "biden": "https://bidenwhitehouse.archives.gov/briefing-room/speeches-remarks/",
    "trump": "https://trumpwhitehouse.archives.gov/briefings-statements/",
}


def _fetch_white_house(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    parts = query.speaker_name.strip().split()
    last = parts[-1].lower() if parts else ""
    archive_root = _WH_ARCHIVES.get(last)
    if not archive_root:
        return results

    try:
        resp = _get(archive_root)
    except Exception as exc:
        logger.warning("white_house: listing failed — %s", exc)
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    anchor_filter = "speeches-remarks" if last == "biden" else "briefings-statements"
    links: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.select("article a[href], h2 a[href], h3 a[href]"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not title or len(title) < 10:
            continue
        if anchor_filter not in href:
            continue
        if href in seen:
            continue
        seen.add(href)
        links.append((href, title))
        if len(links) >= 10:
            break

    for href, title in links:
        full_url = urljoin(archive_root, href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = (
                psoup.find("div", class_=re.compile(
                    r"body-content|page-content|entry-content|field--name-body", re.I))
                or psoup.find("article")
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            results.append(TranscriptResult(
                source="white_house",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=0.9,
            ))
        except Exception as exc:
            logger.warning("white_house: failed %s — %s", full_url, exc)
    return results


# ── 4e2.  Current whitehouse.gov — Trump 2nd term (2025+) ──────────────────
# The archive sites (4e) only cover Biden + Trump's 1st term.  For the
# current administration we hit the live whitehouse.gov.
_CURRENT_WH_SPEAKERS = {
    "trump":   "President Trump",
    "vance":   "Vice President Vance",
    "leavitt": "Press Secretary Leavitt",
    "rubio":   "Secretary Rubio",
    "hegseth": "Secretary Hegseth",
    "noem":    "Secretary Noem",
    "bondi":   "Attorney General Bondi",
}


def _fetch_current_whitehouse(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    parts = query.speaker_name.strip().split()
    last = parts[-1].lower() if parts else ""
    if last not in _CURRENT_WH_SPEAKERS:
        return results

    # WH speech URLs follow: /briefing-room/speeches-remarks/YYYY/MM/DD/slug/
    # Press briefings:       /briefing-room/press-briefings/YYYY/MM/DD/slug/
    # Build year/month/day prefixes to filter links to the event date window.
    date_prefixes: set[str] = set()
    date_listing_urls: list[str] = []
    if query.date_from and query.date_to:
        from datetime import timedelta as _td
        d = query.date_from
        while d <= query.date_to:
            date_prefixes.add(f"/{d.year}/{d.month:02d}/{d.day:02d}/")
            date_prefixes.add(f"/{d.year}/{d.month:02d}/")
            # Also probe date-scoped sub-paths directly — these load faster
            # and sidestep paginated listing pages.
            for base in ("speeches-remarks", "press-briefings"):
                date_listing_urls.append(
                    f"https://www.whitehouse.gov/briefing-room/{base}/"
                    f"{d.year}/{d.month:02d}/{d.day:02d}/"
                )
            d += _td(days=1)

    listing_urls = date_listing_urls + [
        "https://www.whitehouse.gov/briefings-statements/",
        "https://www.whitehouse.gov/briefing-room/speeches-remarks/",  # Biden-era fallback
        "https://www.whitehouse.gov/briefing-room/press-briefings/",
    ]

    links: list[tuple[str, str]] = []
    seen: set[str] = set()

    # Cap total links to keep serial page-fetching within the timeout budget.
    # 2 RSS feeds (~3s) + 6 page fetches (~8s) ≈ 11s, under 25s per event.
    _MAX_LINKS = 6

    # --- Primary: RSS feeds (fast, returns newest content first) ---
    # Only the two most specific feeds; the generic /feed/ duplicates everything.
    _WH_RSS = [
        "https://www.whitehouse.gov/briefing-room/speeches-remarks/feed/",
        "https://www.whitehouse.gov/briefing-room/press-briefings/feed/",
    ]
    _RSS_NS = "http://www.w3.org/2005/Atom"
    for feed_url in _WH_RSS:
        if len(links) >= _MAX_LINKS:
            break
        try:
            resp = _get(feed_url)
            root = ET.fromstring(resp.content)
            # Support both RSS 2.0 (<item>) and Atom (<entry>) formats
            items = root.findall(".//item") or root.findall(f".//{{{_RSS_NS}}}entry")
            for item in items:
                # RSS 2.0
                link_el = item.find("link")
                title_el = item.find("title")
                # Atom fallback
                if link_el is None:
                    link_el = item.find(f"{{{_RSS_NS}}}link")
                if title_el is None:
                    title_el = item.find(f"{{{_RSS_NS}}}title")
                href = (link_el.get("href") or link_el.text or "").strip() if link_el is not None else ""
                title_text = title_el.text.strip() if title_el is not None and title_el.text else ""
                if not href:
                    continue
                path = urlparse(href).path
                if path in seen:
                    continue
                if date_prefixes and not any(p in path for p in date_prefixes):
                    continue
                seen.add(path)
                links.append((path, title_text))
                if len(links) >= _MAX_LINKS:
                    break
        except Exception as exc:
            logger.debug("whitehouse_gov rss: %s — %s", feed_url, exc)

    # --- Secondary: sitemap (only when RSS didn't give enough links) ---
    if len(links) < 8:
        _WH_SITEMAPS = [
            "https://www.whitehouse.gov/post-sitemap.xml",
            "https://www.whitehouse.gov/post-sitemap2.xml",
        ]
        _SM_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
        for sitemap_url in _WH_SITEMAPS:
            if len(links) >= _MAX_LINKS:
                break
            try:
                resp = _get(sitemap_url)
                root = ET.fromstring(resp.content)
                for url_el in root.findall(f".//{{{_SM_NS}}}url"):
                    loc = url_el.findtext(f"{{{_SM_NS}}}loc") or ""
                    path = urlparse(loc).path
                    if path in seen:
                        continue
                    if date_prefixes and not any(p in path for p in date_prefixes):
                        continue
                    if not re.search(r"/briefings-statements/|/briefing-room/", path):
                        continue
                    seen.add(path)
                    links.append((path, ""))
                    if len(links) >= _MAX_LINKS:
                        break
            except Exception as exc:
                logger.debug("whitehouse_gov sitemap: %s — %s", sitemap_url, exc)

    # --- Fallback: scrape listing pages if RSS + sitemap both came up empty ---
    if not links:
        for listing_url in listing_urls:
            try:
                resp = _get(listing_url)
            except Exception:
                continue
            soup = BeautifulSoup(resp.text, "html.parser")
            href_pat = re.compile(r"/releases/|/briefings-statements/|/briefing-room/|/remarks/\d{4}/")
            for a in soup.find_all("a", href=href_pat):
                href = a.get("href", "")
                title = a.get_text(strip=True)
                if not title or href in seen:
                    continue
                if date_prefixes and not any(p in href for p in date_prefixes):
                    continue
                seen.add(href)
                links.append((href, title))
                if len(links) >= _MAX_LINKS:
                    break
            if len(links) >= _MAX_LINKS:
                break

    for href, title in links:
        full_url = urljoin("https://www.whitehouse.gov", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            # WH page body — try multiple selectors in priority order.
            body = (
                psoup.select_one(
                    ".body-content, .entry-content, "
                    ".wp-block-post-content, "
                    "article .wp-block-group, "
                    ".briefing-statement__text"
                )
                or psoup.find("div", class_=re.compile(r"body|content|post", re.I))
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            sample = text[:5000].lower()
            speaker_marker = _CURRENT_WH_SPEAKERS[last].lower()
            conf = 0.95 if speaker_marker in sample or last in sample else 0.6
            results.append(TranscriptResult(
                source="whitehouse_gov",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(href, title, text[:500]),
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("current_whitehouse: failed %s — %s", full_url, exc)
    return results


# ── 4f.  SEC EDGAR — 8-K filings (earnings exhibits) ───────────────────────
def _fetch_sec_edgar(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    if query.event_type != "earnings_call" and not query.ticker:
        return results
    ticker = (query.ticker or query.speaker_name).upper().strip().split()[0]
    if not ticker:
        return results

    sec_headers = {"User-Agent": _SEC_UA, "Accept": "application/json"}
    try:
        resp = _SESSION.get(
            "https://efts.sec.gov/LATEST/search-index",
            params={
                "q": f'"{ticker}" "earnings call"',
                "forms": "8-K",
                "hits": "10",
            },
            headers=sec_headers,
            timeout=20,
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        logger.warning("sec_edgar: search failed — %s", exc)
        return results

    for h in (data.get("hits") or {}).get("hits", [])[:6]:
        src = h.get("_source") or {}
        _id = h.get("_id") or ""
        ciks = src.get("ciks") or []
        if not ciks:
            continue
        cik = int(ciks[0])
        # _id format:  "ACCESSION:filename"  — accession has dashes, folder has none.
        accession_raw = _id.split(":", 1)[0]
        accession_no_dash = accession_raw.replace("-", "")
        file_date = src.get("file_date")
        display = (src.get("display_names") or [""])[0]
        description = src.get("file_description") or ""
        filing_index_url = (
            f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany"
            f"&CIK={cik:010d}&type=8-K&dateb=&owner=include&count=40"
        )
        try:
            archive_url = (
                f"https://www.sec.gov/Archives/edgar/data/{cik}/{accession_no_dash}/"
            )
            page = _SESSION.get(archive_url, headers=sec_headers, timeout=20)
            page.raise_for_status()
            psoup = BeautifulSoup(page.text, "html.parser")
            text = psoup.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            results.append(TranscriptResult(
                source="sec_edgar",
                source_url=archive_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or f"{ticker} 8-K",
                event_date=file_date,
                event_type="earnings_call",
                title=f"{display} — {description}".strip(" —"),
                match_confidence=0.7,
                scrape_notes=(
                    "filing index page; full transcript typically in a "
                    "linked exhibit (ex-99.*)"
                ),
            ))
        except Exception as exc:
            logger.warning("sec_edgar: fetch failed — %s", exc)
    _ = filing_index_url  # referenced for future enrichment
    return results


# ── 4g.  FMP earnings (ticker-aware) ───────────────────────────────────────
def _fetch_fmp_earnings(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return results
    is_earnings = (
        query.event_type == "earnings_call"
        or (query.event_name and "earning" in query.event_name.lower())
        or bool(query.ticker)
    )
    if not is_earnings:
        return results
    ticker = (query.ticker or query.speaker_name).upper().strip().split()[0]
    try:
        resp = _get(
            "https://financialmodelingprep.com/stable/earning-call-transcript",
            params={"symbol": ticker, "apikey": api_key},
        )
        data = resp.json()
        if not isinstance(data, list):
            return results
        for item in data[:5]:
            text = item.get("content", "")
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            year = item.get("year", "")
            quarter = item.get("quarter", "")
            if year and quarter:
                event_date = f"{year}-Q{quarter}"
                label = f"{ticker} {year} Q{quarter} Earnings Call"
            else:
                event_date = item.get("date")
                label = f"{ticker} earnings call"
            results.append(TranscriptResult(
                source="fmp_earnings",
                source_url=f"https://financialmodelingprep.com/financial-transcripts/{ticker}",
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=label,
                event_date=event_date,
                event_type="earnings_call",
                title=label,
                match_confidence=0.9,
            ))
    except Exception as exc:
        logger.warning("fmp_earnings: failed — %s", exc)
    return results


# ── 4h.  YouTube — auto-search + caption pull ──────────────────────────────
# Strategy:
#   1. If the caller passed a YouTube video ID/URL in event_name, use it.
#   2. Otherwise, search YouTube via yt-dlp using the speaker name (+ topic
#      hint / event_name / event_type as keyword padding).  Top N hits are
#      pulled via youtube_transcript_api for auto-captions.
#
# This is the catch-all adapter for any speaker whose transcripts aren't
# hosted as text on a scrape-friendly site (cable interviews, podcasts,
# rallies, recent admin speeches, etc.).
_YT_MAX_RESULTS = 5


def _yt_search_video_ids(query_text: str, n: int = _YT_MAX_RESULTS) -> list[tuple[str, str]]:
    """Return [(video_id, title), ...] for the top N YouTube search hits."""
    if _YT_DLP_DEAD:
        return []
    try:
        import yt_dlp  # type: ignore
    except ImportError:
        logger.debug("yt-dlp not installed; YouTube search disabled")
        return []
    opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": "in_playlist",
        "socket_timeout": 6,
        "retries": 0,
        "extractor_retries": 0,
    }
    try:
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(f"ytsearch{n}:{query_text}", download=False)
        out = []
        for e in (info or {}).get("entries", []) or []:
            vid = e.get("id")
            if vid:
                out.append((vid, e.get("title") or ""))
        return out
    except Exception as exc:
        logger.warning("youtube search failed for %r — %s", query_text, exc)
        return []


# Process-wide kill-switches so a single 429 / IP block doesn't make every
# subsequent event spend minutes retrying the same dead path.
_YT_TRANSCRIPT_API_DEAD = False
_YT_DLP_DEAD = False


def _yt_fetch_captions(video_id: str, language: str = "en") -> Optional[str]:
    """Pull auto-captions for a video. Returns plain text or None.

    Tries youtube_transcript_api first (fast, no download). Falls back to
    yt-dlp subtitle download when the IP is banned or captions are restricted.
    Both paths are short-circuited for the rest of the process once they
    hit IP throttling — retrying just wastes minutes per event.
    """
    global _YT_TRANSCRIPT_API_DEAD, _YT_DLP_DEAD

    # --- Fast path: youtube_transcript_api ---
    if not _YT_TRANSCRIPT_API_DEAD:
        try:
            from youtube_transcript_api import YouTubeTranscriptApi  # type: ignore
            api = YouTubeTranscriptApi()
            snippets = api.fetch(video_id, languages=[language, "en"])
            text = " ".join(s.text for s in snippets)
            if text.strip():
                return text
        except ImportError:
            _YT_TRANSCRIPT_API_DEAD = True
        except Exception as exc:
            msg = str(exc).lower()
            if "ipblocked" in msg or "requestblocked" in msg or "too many requests" in msg or "blocking requests" in msg:
                logger.warning("youtube_transcript_api IP-banned — disabling for this run")
                _YT_TRANSCRIPT_API_DEAD = True
            else:
                logger.debug("youtube_transcript_api failed %s — %s", video_id, exc)

    if _YT_DLP_DEAD:
        return None

    # Suppress yt-dlp's noisy stderr ("ERROR: ..." lines that bypass logging)
    import contextlib as _ctx
    import io as _io
    _stderr_sink = _io.StringIO()

    # --- Fallback: yt-dlp VTT subtitle download (uses Node.js, harder to block) ---
    try:
        import yt_dlp as _ydlp  # type: ignore
        import tempfile as _tmp
        import os as _os
        import shutil as _shutil
        # Find node binary so yt-dlp can use it as its JS runtime
        node_bin = _shutil.which("node")
        js_runtimes = {"node": {"path": node_bin} if node_bin else None}
        # YouTube throttles the timedtext endpoint by IP. Only probe cookie
        # sources if the user explicitly opted in via env var — blind probing
        # of every browser adds 30+ seconds per event for ~zero gain.
        cookie_file = _os.environ.get("YT_COOKIE_FILE")
        env_browser = _os.environ.get("YT_COOKIES_FROM_BROWSER")
        attempts: list[dict] = [{}]
        if cookie_file:
            attempts.append({"cookiefile": cookie_file})
        if env_browser:
            attempts.append({"cookiesfrombrowser": (env_browser,)})

        with _tmp.TemporaryDirectory() as tmp:
            for extra_opts in attempts:
                for f in _os.listdir(tmp):
                    try:
                        _os.remove(_os.path.join(tmp, f))
                    except OSError:
                        pass
                opts = {
                    "quiet": True,
                    "no_warnings": True,
                    "skip_download": True,
                    "writeautomaticsub": True,
                    "writesubtitles": True,
                    "subtitleslangs": [language, f"{language}-orig", "en", "en-orig"],
                    "subtitlesformat": "vtt",
                    "outtmpl": _os.path.join(tmp, "%(id)s"),
                    "js_runtimes": js_runtimes,
                    "retries": 0,
                    "fragment_retries": 0,
                    "extractor_retries": 0,
                    "socket_timeout": 6,
                }
                opts.update(extra_opts)
                try:
                    with _ctx.redirect_stderr(_stderr_sink):
                        with _ydlp.YoutubeDL(opts) as ydl:
                            ydl.download([f"https://www.youtube.com/watch?v={video_id}"])
                except Exception as exc:
                    msg = str(exc).lower()
                    if "429" in msg or "too many requests" in msg:
                        logger.warning("yt-dlp IP-throttled (429) — disabling YouTube for this run")
                        _YT_DLP_DEAD = True
                        return None
                    logger.debug("yt-dlp attempt %s failed: %s", extra_opts, exc)
                    continue
                # Even when yt-dlp doesn't raise, a 429 in stderr means we
                # got nothing — kill the chain so other events skip YouTube.
                stderr_text = _stderr_sink.getvalue().lower()
                if "429" in stderr_text or "too many requests" in stderr_text:
                    logger.warning("yt-dlp IP-throttled (429) — disabling YouTube for this run")
                    _YT_DLP_DEAD = True
                    return None
                if any(f.endswith(".vtt") for f in _os.listdir(tmp)):
                    break
            for fname in sorted(_os.listdir(tmp)):
                if not fname.endswith(".vtt"):
                    continue
                vtt = open(_os.path.join(tmp, fname), encoding="utf-8").read()
                # Strip VTT headers, timestamps, and tags; deduplicate adjacent lines
                seen_lines: set[str] = set()
                words: list[str] = []
                for line in vtt.splitlines():
                    line = line.strip()
                    if not line or line.startswith("WEBVTT") or "-->" in line:
                        continue
                    line = re.sub(r"<[^>]+>", "", line)  # strip <c>, </c> etc
                    if line and line not in seen_lines:
                        seen_lines.add(line)
                        words.append(line)
                result = " ".join(words)
                if result.strip():
                    return result
    except Exception as exc:
        logger.debug("yt-dlp caption download failed %s — %s", video_id, exc)

    return None


def _extract_video_id(text: str) -> Optional[str]:
    if text.startswith("yt:"):
        return text[3:]
    m = re.search(r"(?:v=|youtu\.be/|/shorts/)([A-Za-z0-9_-]{11})", text)
    return m.group(1) if m else None


def _fetch_youtube(query: TranscriptQuery) -> list[TranscriptResult]:
    if _YT_DLP_DEAD and _YT_TRANSCRIPT_API_DEAD:
        return []
    results: list[TranscriptResult] = []

    # Path 1: explicit video ID supplied
    explicit_id = _extract_video_id(query.event_name or "")

    # Path 2: search by speaker name + hints
    candidates: list[tuple[str, str]] = []
    if explicit_id:
        candidates.append((explicit_id, f"YouTube {explicit_id}"))
    else:
        terms = [query.speaker_name]
        if query.event_name:
            terms.append(query.event_name)
        if query.topic_hint:
            terms.append(query.topic_hint)
        if query.event_type and query.event_type != "interview":
            terms.append(query.event_type.replace("_", " "))
        # Use the target date window so YouTube surfaces the right week's content.
        if query.date_from:
            terms.append(query.date_from.strftime("%B %Y"))
        else:
            terms.append(str(date.today().year))
        search_q = " ".join(terms)
        candidates = _yt_search_video_ids(search_q, _YT_MAX_RESULTS)

    for video_id, title in candidates:
        # If a previous attempt killed the YouTube path, bail out — no point
        # iterating remaining candidates when the IP is throttled.
        if _YT_DLP_DEAD and _YT_TRANSCRIPT_API_DEAD:
            break
        try:
            full_text = _yt_fetch_captions(video_id, query.language)
            if not full_text:
                continue
            if not _is_meaningful_transcript(
                    full_text, max(100, query.min_length // 2)):
                continue
            speaker_conf = _speaker_matches(title, query.speaker_name) or 0.4
            # Boost confidence when the video title also matches the event name
            # (e.g. C-SPAN SOTU upload beats a random news clip).
            if query.event_name:
                ev_rel = _event_relevance(
                    type("_R", (), {"title": title, "event_name": title, "full_text": full_text})(),
                    query.event_name,
                )
                speaker_conf = min(1.0, speaker_conf + 0.3 * ev_rel)
            conf = 0.7 if explicit_id else speaker_conf
            results.append(TranscriptResult(
                source="youtube",
                source_url=f"https://www.youtube.com/watch?v={video_id}",
                content_hash=_hash(full_text),
                full_text=full_text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_type=query.event_type or "interview",
                event_date=_best_date(title),
                title=title,
                match_confidence=conf,
                scrape_notes="auto-caption; may contain errors",
            ))
        except Exception as exc:
            logger.warning("youtube: failed %s — %s", video_id, exc)
    return results


# ── 4i.  American Presidency Project — full dated archive of presidential docs ─
# Covers every Trump speech, press conference, executive order signing, etc.
# Person ID 200301 = Donald Trump (2nd term included).
_APP_PERSON_IDS = {
    "trump":   "200301",
    "biden":   "200300",
    "obama":   "200288",
    "clinton": "200299",  # Bill Clinton
    "bush":    "200296",  # George W. Bush
    "harris":  "200302",  # Kamala Harris (VP archive)
    "vance":   "200303",  # JD Vance (VP, may not be indexed yet)
}


def _fetch_presidency_project(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    parts = query.speaker_name.strip().split()
    last = parts[-1].lower() if parts else ""
    person_id = _APP_PERSON_IDS.get(last)
    if not person_id:
        return results

    # Combine event_name + topic_hint into the keyword search so APP returns
    # the specific speech (not every speech the person ever gave). Strip the
    # speaker name out of event_name first — it's redundant with `person2`.
    keywords = query.topic_hint or ""
    if query.event_name:
        ev = query.event_name
        for part in query.speaker_name.split():
            ev = re.sub(rf"\b{re.escape(part)}\b", "", ev, flags=re.IGNORECASE)
        keywords = (ev + " " + keywords).strip()
    params: dict = {
        "field-keywords": keywords,
        "person2": person_id,
        "items_per_page": "25",
        "submit": "Submit",
    }
    if query.date_from:
        params["from[date]"] = query.date_from.strftime("%m-%d-%Y")
    if query.date_to:
        params["to[date]"] = query.date_to.strftime("%m-%d-%Y")

    try:
        resp = _get(
            "https://www.presidency.ucsb.edu/advanced-search",
            params=params,
        )
    except Exception as exc:
        logger.warning("presidency_project: search failed — %s", exc)
        return results

    soup = BeautifulSoup(resp.text, "html.parser")

    # APP uses several different markup patterns across page types
    rows = (
        soup.select("td.views-field-title a")
        or soup.select(".view-content .views-row a")
        or soup.select("h3.node__title a")
        or soup.select("span.field-content a")
    )
    seen: set[str] = set()

    for a in rows[:15]:
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen:
            continue
        seen.add(href)
        full_url = urljoin("https://www.presidency.ucsb.edu", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")

            date_el = psoup.select_one(
                ".date-display-single, .field--name-field-docs-start-date, "
                ".pres-docs__date, time"
            )
            event_date = _best_date(
                date_el.get("datetime") or date_el.get_text(strip=True) if date_el else None,
                title,
            )

            body = psoup.select_one(
                ".field-docs-content, .field--name-field-docs-body, "
                ".pres-docs__content, article .field"
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue

            results.append(TranscriptResult(
                source="presidency_project",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=event_date,
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=0.95,
            ))
        except Exception as exc:
            logger.warning("presidency_project: failed %s — %s", full_url, exc)
    return results


# ── 4j.  Factba.se — Trump-specific transcript archive ────────────────────
def _fetch_factbase(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    parts = query.speaker_name.strip().split()
    last = parts[-1].lower() if parts else ""
    if last != "trump":
        return results

    # Build search candidates. Factba.se has date-based transcript pages and
    # a keyword search endpoint — use both so we hit the right speech even for
    # backtesting dates months in the past.
    candidate_urls: list[str] = []

    # 1. Date-scoped pages: factba.se/transcript?date=YYYY-MM-DD
    if query.date_from and query.date_to:
        from datetime import timedelta as _td2
        d = query.date_from
        while d <= query.date_to:
            candidate_urls.append(
                f"https://factba.se/transcript?date={d.strftime('%Y-%m-%d')}"
            )
            d += _td2(days=1)

    # 2. Keyword search: event name + year gives a targeted result set
    search_terms: list[str] = []
    if query.event_name:
        search_terms.append(query.event_name)
    if query.date_from:
        search_terms.append(str(query.date_from.year))
    if search_terms:
        q_str = " ".join(search_terms)
        candidate_urls.append(f"https://factba.se/results?q={requests.utils.quote(q_str)}")

    # 3. Fallback: generic listing (newest first)
    if not candidate_urls:
        candidate_urls.append("https://factba.se/transcripts")

    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for listing_url in candidate_urls:
        try:
            resp = _get(listing_url)
        except Exception:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.find_all("a", href=re.compile(r"/topic/|/transcript/")):
            href = a.get("href", "")
            title = a.get_text(strip=True)
            if not title or href in seen or len(title) < 10:
                continue
            seen.add(href)
            candidates.append((href, title))
        if len(candidates) >= 15:
            break

    for href, title in candidates:
        full_url = urljoin("https://factba.se", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = (
                psoup.select_one(".transcript-text, .transcript, #transcript, article")
                or psoup.find("main")
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            results.append(TranscriptResult(
                source="factbase",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(href, title, text[:500]),
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=0.9,
            ))
        except Exception as exc:
            logger.warning("factbase: failed %s — %s", full_url, exc)
    return results


# ── 4k.  CNN transcripts — news show transcripts ──────────────────────────
def _fetch_cnn_transcripts(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    if query.event_type and query.event_type not in (
        "interview", "speech", "press_conference", "debate", None
    ):
        return results

    # CNN transcript index — flat list of show transcripts by date
    index_url = "http://transcripts.cnn.com/TRANSCRIPTS/"
    try:
        resp = _get(index_url)
    except Exception:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=re.compile(r"\.html$")):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 20:
            break

    for href, link_title in candidates:
        full_url = urljoin(index_url, href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.find("body")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(text[:5000], query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="cnn_transcripts",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or link_title,
                event_date=_best_date(href, text[:500]),
                event_type=query.event_type or "interview",
                title=link_title,
                match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("cnn: failed %s — %s", full_url, exc)
    return results


# ── 4l.  Singju Post — political speech / rally transcripts ───────────────
def _fetch_singju_post(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = query.event_name or query.speaker_name
    if query.event_name and query.speaker_name not in term:
        term = f"{query.speaker_name} {term}"
    if query.date_from:
        term += f" {query.date_from.strftime('%B %Y')}"

    try:
        resp = _get("https://singjupost.com/", params={"s": term})
    except Exception:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    # When the caller specified an event, require candidate titles to share at
    # least one distinctive token from the event name. Without this, a search
    # like "Donald Trump Cabinet Meeting March 2026" returns whatever generic
    # Trump speech singjupost has indexed most prominently — which then gets
    # reused for every unrelated event in the same month.
    event_kw = _event_keywords(query.event_name) if query.event_name else set()
    for a in soup.select("h2.entry-title a, h3.entry-title a, article a[rel='bookmark']"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen:
            continue
        seen.add(href)
        if event_kw:
            haystack = f"{title} {href}".lower()
            if not any(k in haystack for k in event_kw):
                continue
        candidates.append((href, title))
        if len(candidates) >= 10:
            break

    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = (
                psoup.select_one(".entry-content, .post-content, article")
                or psoup.find("main")
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title or ''} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="singju_post",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("singju_post: failed %s — %s", full_url, exc)
    return results


# ── 4m.  Roll Call — political speech transcripts ─────────────────────────
def _fetch_roll_call(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query) + " transcript"

    try:
        resp = _get("https://rollcall.com/", params={"s": term})
    except Exception:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    candidates: list[tuple[str, str]] = []
    seen: set[str] = set()
    for a in soup.select("h2 a, h3 a, .entry-title a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "transcript" not in (href + title).lower():
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 8:
            break

    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one(".entry-content, .post-content, article, main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title or ''} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="roll_call",
                source_url=full_url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title,
                match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("roll_call: failed %s — %s", full_url, exc)
    return results


# ── 4n.  NPR transcripts ───────────────────────────────────────────────────
def _build_search_term(query: TranscriptQuery) -> str:
    """Build the best search term from query fields."""
    if query.event_name:
        term = query.event_name
        if query.speaker_name.lower() not in term.lower():
            term = f"{query.speaker_name} {term}"
    else:
        term = query.speaker_name
    if query.date_from:
        term += f" {query.date_from.strftime('%B %Y')}"
    return term


def _fetch_npr(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query)
    try:
        resp = _get("https://www.npr.org/search/", params={"query": term})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("h2.title a, .item-info h2 a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "npr.org" not in href:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 10:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one("#storytext, .transcript, article") or psoup.find("main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="npr", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "interview",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("npr: failed %s — %s", full_url, exc)
    return results


# ── 4o.  PBS NewsHour transcripts ──────────────────────────────────────────
def _fetch_pbs(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query)
    try:
        resp = _get("https://www.pbs.org/newshour/search-results", params={"q": term})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("a.card-search__title, h3 a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "pbs.org" not in href:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 10:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = (
                psoup.select_one(".transcript, .body-text, .video-single__transcript")
                or psoup.find("article") or psoup.find("main")
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="pbs", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "interview",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("pbs: failed %s — %s", full_url, exc)
    return results


# ── 4p.  Fox News transcripts ──────────────────────────────────────────────
def _fetch_fox_news(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    try:
        resp = _get(f"https://www.foxnews.com/category/transcript/{query.speaker_name.lower().replace(' ', '-')}")
    except Exception:
        try:
            resp = _get("https://www.foxnews.com/search-results/search", params={"q": _build_search_term(query) + " transcript"})
        except Exception:
            return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("h2.title a, h3.title a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "foxnews.com" not in href:
            continue
        if "transcript" not in (href + title).lower():
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 8:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one(".article-body, article, main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="fox_news", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "interview",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("fox_news: failed %s — %s", full_url, exc)
    return results


# ── 4q.  C-SPAN ────────────────────────────────────────────────────────────
def _fetch_cspan(query: TranscriptQuery) -> list[TranscriptResult]:
    """
    C-SPAN transcript search.

    C-SPAN loads video transcripts via JS so scraping video pages gives nothing.
    Instead we use their *Transcripts* search mode, which returns pages whose
    HTML contains the actual caption/transcript text inside .result-transcript
    or similar elements, and we follow each result to its full transcript page.
    """
    results: list[TranscriptResult] = []
    term = _build_search_term(query)
    # "Transcripts" search mode returns full-text searchable caption results.
    for search_type in ("Transcripts", "Videos"):
        try:
            resp = _get(
                "https://www.c-span.org/search/",
                params={"query": term, "searchtype": search_type},
            )
            break
        except Exception:
            continue
    else:
        return results

    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select(
        ".search-result a, .result-title a, h3 a, "
        ".video-thumbnail-wrapper a, .video__title a"
    ):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or len(title) < 5:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 10:
            break

    for href, title in candidates:
        full_url = urljoin("https://www.c-span.org", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            # C-SPAN transcript text lives in several possible containers.
            body = (
                psoup.select_one(
                    ".transcript-container, .transcript-text, "
                    "#transcript, .video-transcript, "
                    ".captions, [data-transcript]"
                )
                or psoup.find("div", class_=re.compile(r"transcript", re.I))
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="cspan", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("cspan: failed %s — %s", full_url, exc)
    return results


# ── 4r.  The Hill ──────────────────────────────────────────────────────────
def _fetch_the_hill(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    try:
        resp = _get("https://thehill.com/", params={"s": _build_search_term(query) + " transcript"})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("h2.entry-title a, h3.entry-title a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "thehill.com" not in href:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 8:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one(".article-body, .entry-content, article")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="the_hill", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("the_hill: failed %s — %s", full_url, exc)
    return results


# ── 4s.  Politico ──────────────────────────────────────────────────────────
def _fetch_politico(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    try:
        resp = _get("https://www.politico.com/search", params={"q": _build_search_term(query) + " transcript"})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("h3 a, .summary h3 a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "politico.com" not in href:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 8:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one(".story-text, article, main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="politico", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("politico: failed %s — %s", full_url, exc)
    return results


# ── 4t.  Internet Archive (Wayback) — full-text political transcript search ─
def _fetch_archive_org(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = f'"{query.speaker_name}" {query.event_name or "transcript"}'
    try:
        resp = _get(
            "https://archive.org/advancedsearch.php",
            params={
                "q": term,
                "fl[]": "identifier,title,date",
                "rows": "10",
                "output": "json",
                "sort[]": "date desc",
            },
        )
        data = resp.json()
    except Exception:
        return results
    docs = (data.get("response") or {}).get("docs", []) or []
    for doc in docs[:10]:
        ident = doc.get("identifier")
        title = doc.get("title", "")
        if not ident:
            continue
        try:
            details = _get(f"https://archive.org/details/{ident}")
            psoup = BeautifulSoup(details.text, "html.parser")
            body = psoup.select_one("#descript, .description, .item-description")
            text = body.get_text(separator="\n", strip=True) if body else ""
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="archive_org",
                source_url=f"https://archive.org/details/{ident}",
                content_hash=_hash(text), full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=doc.get("date") or _best_date(title),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("archive_org: failed %s — %s", ident, exc)
    return results


# ── 4u.  AP News ───────────────────────────────────────────────────────────
def _fetch_ap_news(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query) + " transcript"
    try:
        resp = _get("https://apnews.com/search", params={"q": term})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("a.Component-headline-0-2-105, h3 a, .CardHeadline a, article a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen:
            continue
        if not href.startswith("http"):
            href = urljoin("https://apnews.com", href)
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 10:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one(".RichTextStoryBody, article, main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="ap_news", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("ap_news: failed %s — %s", full_url, exc)
    return results


# ── 4v.  Axios ─────────────────────────────────────────────────────────────
def _fetch_axios(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query) + " transcript"
    try:
        resp = _get("https://www.axios.com/search", params={"q": term})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.select("h2 a, h3 a, article a, .search-result a"):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or "axios.com" not in href:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 8:
            break
    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.select_one(".gtm-story-text, article, main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="axios", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("axios: failed %s — %s", full_url, exc)
    return results


# ── 4w.  GDELT — free global news database with public API ─────────────────
_GDELT_BLOCKED = {
    "seekingalpha.com", "politico.com", "wsj.com", "nytimes.com",
    "washingtonpost.com", "bloomberg.com", "ft.com", "theatlantic.com",
    "thedispatch.com", "axios.com", "foreignpolicy.com",
}


def _fetch_gdelt(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = f'"{query.speaker_name}" {query.event_name or "transcript"}'
    params: dict = {
        "query": term,
        "mode": "artlist",
        "maxrecords": "25",
        "format": "json",
        "sort": "datedesc",
    }
    if query.date_from:
        params["startdatetime"] = query.date_from.strftime("%Y%m%d") + "000000"
    if query.date_to:
        params["enddatetime"] = query.date_to.strftime("%Y%m%d") + "235959"
    try:
        resp = _get("https://api.gdeltproject.org/api/v2/doc/doc", params=params)
        data = resp.json()
    except Exception:
        return results
    articles = data.get("articles") or []
    for art in articles[:15]:
        url = art.get("url", "")
        title = art.get("title", "")
        pub_date = art.get("seendate", "")
        if not url:
            continue
        # GDELT date format: 20260123T120000Z
        event_date = None
        dm = re.match(r"(\d{4})(\d{2})(\d{2})", pub_date or "")
        if dm:
            event_date = f"{dm.group(1)}-{dm.group(2)}-{dm.group(3)}"
        try:
            page = _get(url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = psoup.find("article") or psoup.find("main")
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="gdelt", source_url=url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=event_date or _best_date(title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("gdelt: failed %s — %s", url, exc)
    return results


# ── 4y.  Real Clear Politics — speech & transcript archive ────────────────
# Works for any politician. RCP indexes presidential, congressional, and
# campaign speech transcripts. Search is generic across the site.
def _fetch_rcp(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query)
    try:
        resp = _get("https://www.realclearpolitics.com/", params={"s": term})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    # RCP search lists results as article links; transcripts usually live
    # under /video/ or /articles/. We accept both and let the body extractor
    # decide whether the page actually contains a transcript.
    for a in soup.find_all("a", href=re.compile(r"realclearpolitics\.com/(video|articles|transcripts)/")):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or len(title) < 8:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 12:
            break

    for full_url, title in candidates:
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = (
                psoup.select_one(".article-body, .transcript, #article-body, article")
                or psoup.find("main")
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="rcp", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(full_url, title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("rcp: failed %s — %s", full_url, exc)
    return results


# ── 4z.  Grabien — political video transcript marketplace ─────────────────
# Free article previews include enough text for keyword counting on many
# political speeches. Generic across speakers.
def _fetch_grabien(query: TranscriptQuery) -> list[TranscriptResult]:
    results: list[TranscriptResult] = []
    term = _build_search_term(query)
    try:
        resp = _get("https://news.grabien.com/", params={"q": term})
    except Exception:
        return results
    soup = BeautifulSoup(resp.text, "html.parser")
    seen: set[str] = set()
    candidates: list[tuple[str, str]] = []
    for a in soup.find_all("a", href=re.compile(r"/story/|/article/|/transcripts/")):
        href = a.get("href", "")
        title = a.get_text(strip=True)
        if not href or href in seen or len(title) < 8:
            continue
        seen.add(href)
        candidates.append((href, title))
        if len(candidates) >= 12:
            break

    for href, title in candidates:
        full_url = urljoin("https://news.grabien.com", href)
        try:
            page = _get(full_url)
            psoup = BeautifulSoup(page.text, "html.parser")
            body = (
                psoup.select_one(".story-body, .article-body, .transcript, article")
                or psoup.find("main")
            )
            if not body:
                continue
            text = body.get_text(separator="\n", strip=True)
            if not _is_meaningful_transcript(text, query.min_length):
                continue
            conf = _speaker_matches(f"{title} {text[:3000]}", query.speaker_name)
            if conf == 0.0:
                continue
            results.append(TranscriptResult(
                source="grabien", source_url=full_url, content_hash=_hash(text),
                full_text=text, speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=_best_date(href, title, text[:500]),
                event_type=query.event_type or "speech",
                title=title, match_confidence=conf,
            ))
        except Exception as exc:
            logger.warning("grabien: failed %s — %s", full_url, exc)
    return results


# ---------------------------------------------------------------------------
# 5.  DEDUP / FILTER / SORT
# ---------------------------------------------------------------------------

def _deduplicate(results: list[TranscriptResult]) -> list[TranscriptResult]:
    seen: set[str] = set()
    unique: list[TranscriptResult] = []
    for r in results:
        if r.content_hash in seen:
            continue
        seen.add(r.content_hash)
        unique.append(r)
    return unique


def _parse_event_date(d: Optional[str]) -> Optional[tuple[int, int, int]]:
    """Parse a normalized event_date into a (Y, M, D) tuple for comparison.
    Returns None when the date is unparseable or missing."""
    if not d:
        return None
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})$", d)
    if m:
        return (int(m.group(1)), int(m.group(2)), int(m.group(3)))
    # Quarter form "2024-Q3" → treat as the first day of the quarter's middle month
    m = re.match(r"^(\d{4})-Q([1-4])$", d)
    if m:
        q = int(m.group(2))
        return (int(m.group(1)), {1: 2, 2: 5, 3: 8, 4: 11}[q], 15)
    y = _parse_year(d)
    return (y, 1, 1) if y else None


def _in_date_range(r: TranscriptResult, q: TranscriptQuery) -> bool:
    """Full-date comparison (not just year).
    When both date_from and date_to are set we are doing a targeted lookup —
    drop results with no parseable date so stale undated transcripts don't
    crowd out the right week's content."""
    parsed = _parse_event_date(r.event_date)
    if parsed is None:
        # Strict mode: if caller gave a date window, require a date.
        if q.date_from and q.date_to:
            return False
        return True
    y, m, d = parsed
    if q.date_from and (y, m, d) < (q.date_from.year, q.date_from.month, q.date_from.day):
        return False
    if q.date_to and (y, m, d) > (q.date_to.year, q.date_to.month, q.date_to.day):
        return False
    return True


def _recency_key(r: TranscriptResult) -> tuple:
    """Sort key that puts newest transcripts first and pushes undated ones
    to the bottom.  Confidence is a tiebreaker for same-day results."""
    parsed = _parse_event_date(r.event_date)
    # Undated → treat as year 0 so they sort last under `reverse=True`.
    ymd = parsed or (0, 0, 0)
    return (ymd, r.match_confidence)


# Generic words that don't help distinguish events. Stripped before computing
# event-name relevance so we focus on the distinctive parts of an event title.
_EVENT_STOPWORDS = frozenset({
    "a", "an", "the", "of", "in", "on", "at", "for", "to", "and", "or", "with",
    "his", "her", "their", "during", "after", "before", "remarks", "speech",
    "address", "meeting", "ceremony", "what", "will", "say", "speak", "speaks",
    "trump", "biden", "obama", "harris", "vance", "musk", "powell",
    "donald", "joe", "kamala", "barack", "elon", "jerome",
    "mr", "mrs", "ms", "president", "vice", "chairman", "fed", "secretary",
    "kalshi", "watching",
})


def _event_keywords(event_name: str) -> set[str]:
    """Distinctive lowercase tokens that identify a specific event."""
    if not event_name:
        return set()
    raw = re.findall(r"[A-Za-z][A-Za-z'-]{2,}", event_name.lower())
    return {w for w in raw if w not in _EVENT_STOPWORDS and len(w) >= 3}


def _event_relevance(r: TranscriptResult, event_name: str) -> float:
    """0..1 score for how well a transcript matches the requested event.

    Higher = more likely the right speech. We look at title and a transcript
    snippet for distinctive event keywords. If event_name has no keywords
    (e.g. a date-only subtitle), return 1.0 so we don't penalize anything.
    """
    keywords = _event_keywords(event_name)
    if not keywords:
        return 1.0
    title = (r.title or r.event_name or "").lower()
    snippet = (r.full_text or "")[:2000].lower()
    title_hits = sum(1 for k in keywords if k in title)
    snippet_hits = sum(1 for k in keywords if k in snippet)
    # Title match is authoritative: all keywords in title → score 1.0.
    # Snippet hits are a bonus that can push a partial title match higher.
    score = (2 * title_hits + snippet_hits) / (2 * len(keywords))
    return min(1.0, score)


# ---------------------------------------------------------------------------
# 6.  ENTRYPOINT
# ---------------------------------------------------------------------------

_DDG_TRANSCRIPT_DOMAINS = (
    "pbs.org", "apnews.com", "time.com", "cbsnews.com", "nbcnews.com",
    "abcnews.go.com", "cnn.com", "foxnews.com", "npr.org", "reuters.com",
    "rev.com", "factba.se", "rollcall.com", "c-span.org",
)


def _fetch_ddg_news(query: TranscriptQuery) -> list[TranscriptResult]:
    """Generic last-resort: DuckDuckGo HTML search for `<speaker> <event> transcript`,
    then scrape <p> text from the top results that look like full transcripts."""
    if not query.speaker_name:
        return []
    # Strip "Speaker Name - " prefix Kalshi adds to event names before searching.
    raw_event = (query.event_name or "").strip()
    speaker_prefix = re.compile(
        r"^" + re.escape(query.speaker_name) + r"\s*[-–]\s*", re.IGNORECASE
    )
    clean_event = speaker_prefix.sub("", raw_event).strip()
    terms = [query.speaker_name, clean_event or raw_event, "transcript"]
    if query.date_from:
        terms.append(query.date_from.strftime("%B %Y"))
    q = " ".join(t for t in terms if t)
    try:
        r = requests.get(
            "https://html.duckduckgo.com/html/",
            params={"q": q},
            headers={"User-Agent": "Mozilla/5.0"},
            timeout=12,
        )
        if r.status_code != 200:
            return []
    except Exception as exc:
        logger.debug("ddg search failed: %s", exc)
        return []

    import urllib.parse as _up
    candidates: list[str] = []
    for raw in re.findall(r'uddg=([^&"\s]+)', r.text):
        url = _up.unquote(raw)
        if any(d in url for d in _DDG_TRANSCRIPT_DOMAINS):
            if url not in candidates:
                candidates.append(url)
        if len(candidates) >= 5:
            break

    # If event-specific search returned nothing, try broader: just speaker + date
    if not candidates and query.date_from:
        broad_q = f"{query.speaker_name} speech remarks transcript {query.date_from.strftime('%B %Y')}"
        try:
            r2 = requests.get("https://html.duckduckgo.com/html/", params={"q": broad_q},
                              headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            for raw in re.findall(r'uddg=([^&"\s]+)', r2.text):
                url2 = _up.unquote(raw)
                if any(d in url2 for d in _DDG_TRANSCRIPT_DOMAINS) and url2 not in candidates:
                    candidates.append(url2)
                if len(candidates) >= 5:
                    break
        except Exception:
            pass

    results: list[TranscriptResult] = []
    for url in candidates:
        try:
            page = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=12)
            if page.status_code != 200:
                continue
            soup = BeautifulSoup(page.text, "html.parser")
            text = " ".join(p.get_text(" ", strip=True) for p in soup.find_all("p"))
            if len(text) < 5_000:
                continue
            title_tag = soup.find("title")
            title = title_tag.get_text(strip=True) if title_tag else url
            results.append(TranscriptResult(
                source="ddg_news",
                source_url=url,
                content_hash=_hash(text),
                full_text=text,
                speaker_name=query.speaker_name,
                event_name=query.event_name or title,
                event_date=query.date_from.isoformat() if query.date_from else None,
                title=title,
                match_confidence=0.7,
            ))
        except Exception as exc:
            logger.debug("ddg fetch %s failed: %s", url, exc)
            continue
    return results


_ADAPTERS: list[tuple[str, Callable[[TranscriptQuery], list[TranscriptResult]]]] = [
    ("ddg_news", _fetch_ddg_news),
    ("presidency_project", _fetch_presidency_project),
    ("factbase", _fetch_factbase),
    ("rev", _fetch_rev),
    ("singju_post", _fetch_singju_post),
    ("roll_call", _fetch_roll_call),
    ("cnn_transcripts", _fetch_cnn_transcripts),
    ("npr", _fetch_npr),
    ("pbs", _fetch_pbs),
    ("fox_news", _fetch_fox_news),
    ("cspan", _fetch_cspan),
    ("the_hill", _fetch_the_hill),
    ("politico", _fetch_politico),
    ("archive_org", _fetch_archive_org),
    ("ap_news", _fetch_ap_news),
    ("axios", _fetch_axios),
    ("gdelt", _fetch_gdelt),
    ("govinfo", _fetch_govinfo),
    ("miller_center", _fetch_miller_center),
    ("federal_reserve", _fetch_federal_reserve),
    ("white_house", _fetch_white_house),
    ("whitehouse_gov", _fetch_current_whitehouse),
    ("sec_edgar", _fetch_sec_edgar),
    ("fmp_earnings", _fetch_fmp_earnings),
    # ("youtube", _fetch_youtube),  # disabled: IP-banned frequently, adds noise
]


def fetch_transcripts(query: TranscriptQuery) -> list[TranscriptResult]:
    """
    Public API.  Never raises.  Returns deduplicated results sorted newest-
    first by event_date; undated results sink to the bottom.  Confidence is
    a tiebreaker for same-day results.
    """
    excluded = set(query.exclude_sources)
    active = [(n, f) for n, f in _ADAPTERS if n not in excluded]

    # Per-adapter hard timeout (seconds) so a single slow source can't
    # blow the entire event's budget. Override with TRANSCRIPT_BOT_TIMEOUT.
    timeout_s = query.timeout if query.timeout is not None else float(os.environ.get("TRANSCRIPT_BOT_TIMEOUT", "45"))

    all_results: list[TranscriptResult] = []
    pool = ThreadPoolExecutor(max_workers=max(1, len(active)))
    futures = {pool.submit(f, query): name for name, f in active}
    try:
        for fut in as_completed(futures, timeout=timeout_s):
            name = futures[fut]
            try:
                batch = fut.result()
                logger.info("%s returned %d result(s)", name, len(batch))
                all_results.extend(batch)
            except Exception as exc:
                logger.error("adapter %s crashed: %s", name, exc)
    except _FuturesTimeout:
        done_names = [futures[f] for f in futures if f.done()]
        straggler_names = [futures[f] for f in futures if not f.done()]
        # Drain any results that DID complete but weren't yet processed.
        for fut in [f for f in futures if f.done()]:
            try:
                all_results.extend(fut.result())
            except Exception:
                pass
        if straggler_names:
            logger.warning("adapter timeout — abandoning %d straggler(s): %s",
                           len(straggler_names), ", ".join(straggler_names[:6]))
    finally:
        # Don't wait — abandoned threads will finish in the background.
        pool.shutdown(wait=False, cancel_futures=True)

    all_results = [_normalize_result(r) for r in all_results]
    all_results = _deduplicate(all_results)

    dated = [r for r in all_results if _in_date_range(r, query)]
    # If strict date filtering wiped everything out, fall back to undated
    # results so callers always get something to work with — UNLESS the
    # caller asked for strict_date matching (e.g. per-event backtesting,
    # where a wrong-day transcript is worse than no transcript).
    if not dated and all_results:
        if query.strict_date and query.date_from and query.date_to:
            print(
                f"  [transcript_bot] strict_date: dropping {len(all_results)} "
                f"undated result(s); no transcript matches the date window",
                file=sys.stderr,
            )
        else:
            print(
                f"  [transcript_bot] no dated results found — using {len(all_results)} undated",
                file=sys.stderr,
            )
            dated = all_results

    # Fold event-name relevance into match_confidence so the right speech
    # ranks above generic same-speaker transcripts. e.g. for the Memphis
    # Roundtable, a transcript whose title mentions "Memphis" wins over a
    # random Trump speech from the same week.
    if query.event_name:
        for r in dated:
            relevance = _event_relevance(r, query.event_name)
            r.match_confidence = r.match_confidence * (0.3 + 0.7 * relevance)

    # Sort newest-first, then by (now relevance-adjusted) confidence.
    dated.sort(key=_recency_key, reverse=True)
    dated = dated[: query.max_results]

    # Show caller which sources contributed
    sources = sorted({r.source for r in dated})
    print(f"  [transcript_bot] sources: {', '.join(sources) or 'none'}", file=sys.stderr)

    logger.info("fetch_transcripts: returning %d result(s)", len(dated))
    return dated
