"""
profile_agent.py
----------------   
"""

from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Optional

import db
from kalshi_word_counter import DEFAULT_TARGETS, KalshiCounter

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_PROFILE_MAX_AGE_DAYS   = 30    # profile older than this triggers a refresh
_TRANSCRIPT_MAX_AGE_DAYS = 14   # transcript older than this is considered stale
_MIN_TRANSCRIPT_CHARS   = 5_000 # discard transcripts shorter than this
_DEFAULT_RECENCY_HALFLIFE = 30  # days; used for exponential recency decay


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _parse_dt(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp string into a UTC-aware datetime."""
    if not ts:
        return None
    try:
        # Handle both "Z" suffix and "+00:00" offset
        return datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None


def _days_since(ts: Optional[str]) -> float:
    """Return how many days ago a timestamp was. Returns inf if unparseable."""
    dt = _parse_dt(ts)
    if dt is None:
        return float("inf")
    return (datetime.now(tz=timezone.utc) - dt).total_seconds() / 86_400


def _recency_score(ts: Optional[str], halflife: float = _DEFAULT_RECENCY_HALFLIFE) -> float:
    """
    Exponential decay recency score in [0, 1].
    1.0 = just now, ~0.5 at halflife days, ~0.0 at 3× halflife.
    """
    days = _days_since(ts)
    return math.exp(-days * math.log(2) / halflife)


def _compute_all_stats(
    transcripts: list[dict],
    words: list[str],
) -> dict[str, dict]:
    """
    Single-pass stats for all target words at once.

    Builds KalshiCounter once with every word, then runs all transcripts
    through it in one loop — O(transcripts) instead of O(transcripts × words).

    Returns:
        { word: { hit_rate_lifetime, avg_freq, recency, n_samples } }
    """
    empty = {
        w: {"hit_rate_lifetime": 0.0, "avg_freq": 0.0, "recency": 0.0, "n_samples": 0}
        for w in words
    }
    if not transcripts or not words:
        return empty

    counter    = KalshiCounter(targets=list(words))
    word_lower = {w: w.lower() for w in words}
    hits       = {w: 0   for w in words}
    totals     = {w: 0.0 for w in words}
    newest_ts: Optional[str] = None
    n_used = 0

    for t in transcripts:
        text = t.get("full_text", "")
        if len(text) < _MIN_TRANSCRIPT_CHARS:
            continue
        n_used += 1
        result    = counter.count(text)
        cr_lower  = {k.lower(): v for k, v in result.counts.items()}
        for w, key in word_lower.items():
            c = cr_lower.get(key, 0)
            if c > 0:
                hits[w]   += 1
                totals[w] += c

        ts = t.get("fetched_at") or t.get("event_date")
        if ts and (newest_ts is None or ts > newest_ts):
            newest_ts = ts

    if n_used == 0:
        return empty

    recency = _recency_score(newest_ts) if newest_ts else 0.0
    return {
        w: {
            "hit_rate_lifetime": hits[w]   / n_used,
            "avg_freq":          totals[w] / n_used,
            "recency":           recency,
            "n_samples":         n_used,
        }
        for w in words
    }


# ---------------------------------------------------------------------------
# research_transcripts()
# ---------------------------------------------------------------------------

def research_transcripts(
    speaker: str,
    event_type: Optional[str] = None,
    event_name: Optional[str] = None,
    event_ticker: Optional[str] = None,
    date_from=None,
    date_to=None,
    max_results: int = 5,
) -> list[dict]:
    """
    Fetch fresh transcripts for a speaker via transcript_bot, then cache
    each result in db.transcripts.

    Returns a list of transcript dicts (same shape as db.get_transcripts()).
    """
    from transcript_bot import TranscriptQuery, fetch_transcripts

    print(f"  [research] fetching transcripts for {speaker!r} "
          f"{('event=' + event_name) if event_name else ''} ...")

    results = fetch_transcripts(TranscriptQuery(
        speaker_name  = speaker,
        event_type    = event_type,
        event_name    = event_name,
        date_from     = date_from,
        date_to       = date_to,
        max_results   = max_results,
        strict_date   = bool(date_from and date_to),
    ))

    cached: list[dict] = []
    for r in results:
        if len(r.full_text) < _MIN_TRANSCRIPT_CHARS:
            continue
        row_id = db.insert_transcript(
            speaker      = speaker,
            full_text    = r.full_text,
            content_hash = r.content_hash,
            event_type   = event_type or r.event_type or "",
            event_ticker = event_ticker or "",
            source       = r.source,
            source_url   = r.source_url,
        )
        # Return in the same shape as db.get_transcripts()
        cached.append({
            "id":           row_id,
            "speaker":      speaker,
            "event_type":   event_type or r.event_type or "",
            "event_ticker": event_ticker or "",
            "content_hash": r.content_hash,
            "full_text":    r.full_text,
            "source":       r.source,
            "source_url":   r.source_url,
            "event_date":   r.event_date,
            "fetched_at":   r.fetched_at,
        })
        print(f"    → cached {r.source} ({len(r.full_text):,} chars)")

    print(f"  [research] {len(cached)} transcript(s) cached")
    return cached


# ---------------------------------------------------------------------------
# obtain_transcripts()
# ---------------------------------------------------------------------------

def obtain_transcripts(
    speaker: str,
    event_type: Optional[str] = None,
    event_ticker: Optional[str] = None,
    event_name: Optional[str] = None,
    date_from=None,
    date_to=None,
    max_age_days: int = _TRANSCRIPT_MAX_AGE_DAYS,
) -> list[dict]:
    """
    DB-first transcript lookup. Returns cached transcripts if they exist and
    are fresh enough; otherwise falls back to research_transcripts().

    This is the single point of entry for "I need transcripts for speaker X."
    """
    # 1 — Check what's in the DB already
    cached = db.get_transcripts(
        speaker      = speaker,
        event_type   = event_type,
        event_ticker = event_ticker or "",
    )

    # 2 — Filter to fresh, long-enough transcripts
    fresh = [
        t for t in cached
        if len(t.get("full_text", "")) >= _MIN_TRANSCRIPT_CHARS
        and _days_since(t.get("fetched_at")) <= max_age_days
    ]

    if fresh:
        print(f"  [obtain] using {len(fresh)} cached transcript(s) from db")
        return fresh

    # 3 — Nothing usable in DB — go to the web
    print(f"  [obtain] no fresh transcripts in db — researching ...")
    return research_transcripts(
        speaker      = speaker,
        event_type   = event_type,
        event_name   = event_name,
        event_ticker = event_ticker,
        date_from    = date_from,
        date_to      = date_to,
    )


# ---------------------------------------------------------------------------
# insert_profiles()
# ---------------------------------------------------------------------------

def insert_profiles(
    speaker: str,
    transcripts: list[dict],
    words: Optional[list[str]] = None,
    event_type: str = "",
) -> int:
    """
    Build brand-new speaker/word profiles from a list of transcripts.
    Skips any (speaker, word, event_type) combo that already exists in DB.

    words — list of target words; defaults to DEFAULT_TARGETS
    Returns the number of profiles created.
    """
    targets = words or DEFAULT_TARGETS
    created = 0

    all_stats = _compute_all_stats(transcripts, targets)

    for word in targets:
        existing = db.get_cached_profile(speaker, word=word, event_type=event_type)
        if existing:
            continue

        stats = all_stats[word]
        if stats["n_samples"] == 0:
            continue

        db.insert_new_profile(speaker, word, event_type)
        db.update_profile(
            speaker            = speaker,
            word               = word,
            event_type         = event_type,
            hit_rate_lifetime  = stats["hit_rate_lifetime"],
            hit_rate_recent    = stats["hit_rate_lifetime"],  # all samples are "recent" on first insert
            avg_freq           = stats["avg_freq"],
            recency            = stats["recency"],
            n_samples_lifetime = stats["n_samples"],
            n_samples_recent   = stats["n_samples"],
        )
        created += 1

    print(f"  [insert_profiles] created {created} new profile(s) for {speaker!r}")
    return created


# ---------------------------------------------------------------------------
# update_profiles()
# ---------------------------------------------------------------------------

def update_profiles(
    speaker: str,
    transcripts: list[dict],
    words: Optional[list[str]] = None,
    event_type: str = "",
) -> int:
    """
    Incrementally update existing profiles with new transcript data.
    Creates a blank profile first for any word that doesn't exist yet.

    words — list of target words; defaults to DEFAULT_TARGETS
    Returns the number of profiles updated.
    """
    targets = words or DEFAULT_TARGETS
    updated = 0

    all_stats = _compute_all_stats(transcripts, targets)

    for word in targets:
        new_stats = all_stats[word]
        if new_stats["n_samples"] == 0:
            continue

        # Ensure profile row exists
        db.insert_new_profile(speaker, word, event_type)
        rows = db.get_cached_profile(speaker, word=word, event_type=event_type)
        p = rows[0]

        old_n  = p["n_samples_lifetime"]
        new_n  = old_n + new_stats["n_samples"]

        # Weighted average of old stats + new batch
        new_hit_lifetime = (
            (p["hit_rate_lifetime"] * old_n + new_stats["hit_rate_lifetime"] * new_stats["n_samples"])
            / new_n
        )
        new_avg_freq = (
            (p["avg_freq"] * old_n + new_stats["avg_freq"] * new_stats["n_samples"])
            / new_n
        )

        # Recent = new batch only (it's literally just arrived)
        new_hit_recent = new_stats["hit_rate_lifetime"]
        new_n_recent   = new_stats["n_samples"]

        db.update_profile(
            speaker            = speaker,
            word               = word,
            event_type         = event_type,
            hit_rate_lifetime  = new_hit_lifetime,
            hit_rate_recent    = new_hit_recent,
            avg_freq           = new_avg_freq,
            recency            = new_stats["recency"],
            n_samples_lifetime = new_n,
            n_samples_recent   = new_n_recent,
        )
        updated += 1

    print(f"  [update_profiles] updated {updated} profile(s) for {speaker!r}")
    return updated


# ---------------------------------------------------------------------------
# daily_health_check()
# ---------------------------------------------------------------------------

def daily_health_check() -> dict:
    """
    Log a one-line snapshot of overall DB health for ops visibility.

    Queries the DB directly using available db functions and returns a dict
    with row counts and stale fractions. Useful for spotting data rot early.

    Returns keys (all int):
        transcripts_total, transcripts_stale,
        news_cache_total,  speaker_profiles_total, trade_log_total
    """
    snap: dict = {}
    try:
        now_str = (
            datetime.now(timezone.utc) - timedelta(days=_TRANSCRIPT_MAX_AGE_DAYS)
        ).isoformat()

        with db._connect() as con:
            snap["transcripts_total"] = con.execute(
                "SELECT COUNT(*) FROM transcripts"
            ).fetchone()[0]

            snap["transcripts_stale"] = con.execute(
                "SELECT COUNT(*) FROM transcripts WHERE fetched_at < ?", (now_str,)
            ).fetchone()[0]

            snap["news_cache_total"] = con.execute(
                "SELECT COUNT(*) FROM news_cache"
            ).fetchone()[0]

            snap["speaker_profiles_total"] = con.execute(
                "SELECT COUNT(*) FROM speaker_profiles"
            ).fetchone()[0]

            snap["trade_log_total"] = con.execute(
                "SELECT COUNT(*) FROM trade_log"
            ).fetchone()[0]

            snap["trade_log_unsettled"] = con.execute(
                "SELECT COUNT(*) FROM trade_log WHERE outcome IS NULL OR outcome = ''"
            ).fetchone()[0]

        print(
            f"  [health] transcripts: {snap['transcripts_total']} "
            f"({snap['transcripts_stale']} stale)  |  "
            f"news: {snap['news_cache_total']}  |  "
            f"profiles: {snap['speaker_profiles_total']}  |  "
            f"trades: {snap['trade_log_total']} ({snap['trade_log_unsettled']} unsettled)"
        )
    except Exception as exc:
        print(f"  [health] check failed: {exc}")

    return snap


# ---------------------------------------------------------------------------
# check_speaker_profiles()  — main entry point
# ---------------------------------------------------------------------------

def check_speaker_profiles(
    speaker: str,
    word: Optional[str] = None,
    event_type: Optional[str] = None,
    event_name: Optional[str] = None,
    event_ticker: Optional[str] = None,
    words: Optional[list[str]] = None,
    max_age_days: int = _PROFILE_MAX_AGE_DAYS,
    date_from=None,
    date_to=None,
) -> list[dict]:
    """
    Main entry point for the profile agent.

    1. Calls db.get_cached_profile() for the speaker/word.
    2. If no profiles found → obtain fresh transcripts → insert_profiles().
    3. If profiles exist but are stale (> max_age_days old) →
         obtain fresh transcripts → update_profiles().
    4. Returns a clean list of profile dicts:
         {speaker, word, hit_rate_lifetime, hit_rate_recent,
          avg_freq, recency, momentum}

    Parameters
    ----------
    speaker      : Speaker name (required)
    word         : Narrow to one word (optional)
    event_type   : Narrow to one event type (optional)
    event_name   : Passed to transcript_bot if research is needed
    event_ticker : Passed to transcript lookup / research
    words        : Full word list to build profiles for (defaults to DEFAULT_TARGETS)
    max_age_days : How old a profile can be before triggering a refresh (default 30)
    date_from / date_to : Date window passed to research_transcripts if needed
    """
    profiles = db.get_cached_profile(speaker, word=word, event_type=event_type)

    if not profiles:
        # ── Red path: no profile at all ──────────────────────────────────────
        print(f"  [check] no profiles found for {speaker!r} — fetching transcripts ...")
        transcripts = obtain_transcripts(
            speaker      = speaker,
            event_type   = event_type,
            event_name   = event_name,
            event_ticker = event_ticker,
            date_from    = date_from,
            date_to      = date_to,
        )
        if transcripts:
            insert_profiles(speaker, transcripts, words=words, event_type=event_type or "")
            profiles = db.get_cached_profile(speaker, word=word, event_type=event_type)
        else:
            print(f"  [check] could not find any transcripts for {speaker!r}")
            return []

    else:
        # ── Check staleness ───────────────────────────────────────────────────
        oldest_update = min(_days_since(p["updated_at"]) for p in profiles)
        if oldest_update > max_age_days:
            print(f"  [check] profiles for {speaker!r} are {oldest_update:.0f} days old "
                  f"(max {max_age_days}) — refreshing ...")
            transcripts = obtain_transcripts(
                speaker      = speaker,
                event_type   = event_type,
                event_name   = event_name,
                event_ticker = event_ticker,
                date_from    = date_from,
                date_to      = date_to,
            )
            if transcripts:
                update_profiles(speaker, transcripts, words=words, event_type=event_type or "")
                profiles = db.get_cached_profile(speaker, word=word, event_type=event_type)
        else:
            print(f"  [check] profiles for {speaker!r} are fresh "
                  f"(last updated {oldest_update:.1f} days ago)")

    # ── Return clean subset of fields ─────────────────────────────────────────
    return [
        {
            "speaker":           p["speaker"],
            "word":              p["word"],
            "hit_rate_lifetime": p["hit_rate_lifetime"],
            "hit_rate_recent":   p["hit_rate_recent"],
            "avg_freq":          p["avg_freq"],
            "recency":           p["recency"],
            "momentum":          p["momentum"],
        }
        for p in profiles
    ]
