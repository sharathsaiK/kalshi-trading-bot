"""
db.py
-----
"""

from __future__ import annotations

import json
import os
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Optional

# ---------------------------------------------------------------------------
# DB path — respects DB_PATH env var (same as friend's .env convention)
# ---------------------------------------------------------------------------
_DEFAULT_DB = Path(__file__).parent / "kalshi.db"
_DB_PATH = Path(os.getenv("DB_PATH", str(_DEFAULT_DB)))


# ---------------------------------------------------------------------------
# Connection
# ---------------------------------------------------------------------------

@contextmanager
def _connect():
    """
    Yield a sqlite3 connection. Auto-commits on success, rolls back on error,
    always closes. WAL mode for concurrent reads; foreign keys enforced.
    """
    conn = sqlite3.connect(str(_DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _row(r: sqlite3.Row) -> dict:
    return dict(r)


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------

_DDL = """
-- One row per (speaker, word, event_type) — richer than a JSON blob.
-- hit_rate_recent covers the last 30 days; momentum = recent - lifetime.
CREATE TABLE IF NOT EXISTS speaker_profiles (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker             TEXT    NOT NULL,
    event_type          TEXT    NOT NULL DEFAULT '',
    word                TEXT    NOT NULL,
    hit_rate_lifetime   REAL    DEFAULT 0.0,
    hit_rate_recent     REAL    DEFAULT 0.0,
    momentum            REAL    DEFAULT 0.0,
    avg_freq            REAL    DEFAULT 0.0,
    recency             REAL    DEFAULT 0.0,
    n_samples_lifetime  INTEGER DEFAULT 0,
    n_samples_recent    INTEGER DEFAULT 0,
    updated_at          TEXT    NOT NULL DEFAULT '',
    UNIQUE(speaker, event_type, word)
);

CREATE INDEX IF NOT EXISTS idx_sp_speaker ON speaker_profiles (speaker);

-- Richer trade log: tracks paper vs live mode, full bet math, and payout.
CREATE TABLE IF NOT EXISTS trade_log (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    mode             TEXT NOT NULL DEFAULT 'paper'
                          CHECK(mode IN ('paper', 'live')),
    ticker           TEXT NOT NULL,
    speaker          TEXT NOT NULL,
    event_type       TEXT NOT NULL DEFAULT '',
    word             TEXT NOT NULL,
    our_probability  REAL NOT NULL,
    kalshi_odds      REAL NOT NULL,
    ev_per_contract  REAL NOT NULL,
    bet_side         TEXT NOT NULL CHECK(bet_side IN ('yes', 'no')),
    contracts        INTEGER NOT NULL DEFAULT 0,
    outcome          TEXT    DEFAULT NULL
                          CHECK(outcome IN ('win', 'loss', 'cancelled', NULL)),
    payout_cents     REAL    DEFAULT 0.0,
    placed_at        TEXT    NOT NULL,
    resolved_at      TEXT    DEFAULT NULL
);

CREATE INDEX IF NOT EXISTS idx_tl_ticker ON trade_log (ticker);

-- Active transcripts: content_hash for dedup (our design).
CREATE TABLE IF NOT EXISTS transcripts (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker       TEXT    NOT NULL,
    event_type    TEXT    NOT NULL DEFAULT '',
    event_ticker  TEXT    NOT NULL DEFAULT '',
    content_hash  TEXT    NOT NULL UNIQUE,
    full_text     TEXT    NOT NULL,
    source        TEXT    NOT NULL DEFAULT '',
    source_url    TEXT    NOT NULL DEFAULT '',
    event_date    TEXT    DEFAULT NULL,
    fetched_at    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_tr_speaker  ON transcripts (speaker);
CREATE INDEX IF NOT EXISTS idx_tr_fetched  ON transcripts (fetched_at);

-- Long-term archive: moved here before deletion so momentum calc stays valid.
CREATE TABLE IF NOT EXISTS transcripts_archive (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker       TEXT    NOT NULL,
    event_type    TEXT    NOT NULL DEFAULT '',
    event_ticker  TEXT    NOT NULL DEFAULT '',
    content_hash  TEXT    NOT NULL,
    full_text     TEXT    NOT NULL,
    source        TEXT    NOT NULL DEFAULT '',
    source_url    TEXT    NOT NULL DEFAULT '',
    event_date    TEXT    DEFAULT NULL,
    archived_at   TEXT    NOT NULL
);

-- News cache: URL uniqueness prevents duplicate articles (our design).
CREATE TABLE IF NOT EXISTS news_cache (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    query        TEXT NOT NULL,
    title        TEXT NOT NULL,
    url          TEXT NOT NULL UNIQUE,
    published_at TEXT DEFAULT NULL,
    source       TEXT NOT NULL DEFAULT '',
    snippet      TEXT NOT NULL DEFAULT '',
    fetched_at   TEXT NOT NULL,
    relevancy    REAL DEFAULT NULL,
    event_type   TEXT NOT NULL DEFAULT '',
    article_type TEXT NOT NULL DEFAULT 'news'
);

CREATE INDEX IF NOT EXISTS idx_nc_query   ON news_cache (query);
CREATE INDEX IF NOT EXISTS idx_nc_fetched ON news_cache (fetched_at);

-- Embeddings cache for RAG / semantic similarity.
-- DDL bug fixed: missing comma before UNIQUE in friend's original.
CREATE TABLE IF NOT EXISTS embeddings_cache (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    target_id       INTEGER NOT NULL,
    table_reference TEXT    NOT NULL,
    vector          BLOB    NOT NULL,
    created_at      TEXT    NOT NULL,
    UNIQUE(target_id, table_reference)
);

-- LightGBM training data: one row per (speaker, word, settled Kalshi event).
-- Populated automatically by run_pipeline.py when a market settles, and
-- augmented synthetically from speaker_profiles during training.
CREATE TABLE IF NOT EXISTS training_data (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker             TEXT NOT NULL,
    word                TEXT NOT NULL,
    event_type          TEXT NOT NULL DEFAULT '',
    event_ticker        TEXT NOT NULL DEFAULT '',
    -- Speaker profile features captured at prediction time (pre-update)
    hit_rate_lifetime   REAL NOT NULL DEFAULT 0.5,
    hit_rate_recent     REAL NOT NULL DEFAULT 0.5,
    momentum            REAL NOT NULL DEFAULT 0.0,
    avg_freq            REAL NOT NULL DEFAULT 1.0,
    recency             REAL NOT NULL DEFAULT 0.5,
    n_samples_lifetime  INTEGER NOT NULL DEFAULT 0,
    n_samples_recent    INTEGER NOT NULL DEFAULT 0,
    -- News relevancy features at time of event
    rel_max             REAL NOT NULL DEFAULT 0.0,
    rel_mean            REAL NOT NULL DEFAULT 0.0,
    rel_top3_mean       REAL NOT NULL DEFAULT 0.0,
    rel_count_hi        INTEGER NOT NULL DEFAULT 0,
    rel_n               INTEGER NOT NULL DEFAULT 0,
    -- Market features
    kalshi_odds         REAL NOT NULL DEFAULT 0.5,
    ev_score            REAL NOT NULL DEFAULT 0.0,
    -- Ground truth: did the speaker say the word? (Kalshi official resolution)
    did_say_word        INTEGER NOT NULL CHECK(did_say_word IN (0, 1)),
    created_at          TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_td_speaker ON training_data (speaker);
CREATE INDEX IF NOT EXISTS idx_td_word    ON training_data (word);

-- Post-cutoff holdout: same schema as training_data but NEVER enters training.
-- Populated by harvest_training_data.py --holdout
CREATE TABLE IF NOT EXISTS training_data_holdout (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    speaker             TEXT NOT NULL,
    word                TEXT NOT NULL,
    event_type          TEXT NOT NULL DEFAULT '',
    event_ticker        TEXT NOT NULL DEFAULT '',
    hit_rate_lifetime   REAL NOT NULL DEFAULT 0.5,
    hit_rate_recent     REAL NOT NULL DEFAULT 0.5,
    momentum            REAL NOT NULL DEFAULT 0.0,
    avg_freq            REAL NOT NULL DEFAULT 1.0,
    recency             REAL NOT NULL DEFAULT 0.5,
    n_samples_lifetime  INTEGER NOT NULL DEFAULT 0,
    n_samples_recent    INTEGER NOT NULL DEFAULT 0,
    rel_max             REAL NOT NULL DEFAULT 0.0,
    rel_mean            REAL NOT NULL DEFAULT 0.0,
    rel_top3_mean       REAL NOT NULL DEFAULT 0.0,
    rel_count_hi        INTEGER NOT NULL DEFAULT 0,
    rel_n               INTEGER NOT NULL DEFAULT 0,
    kalshi_odds         REAL NOT NULL DEFAULT 0.5,
    ev_score            REAL NOT NULL DEFAULT 0.0,
    did_say_word        INTEGER NOT NULL CHECK(did_say_word IN (0, 1)),
    created_at          TEXT NOT NULL,
    topic_match         REAL NOT NULL DEFAULT 0.5,
    event_title         TEXT NOT NULL DEFAULT '',
    event_date          TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_tdh_speaker ON training_data_holdout (speaker);
CREATE INDEX IF NOT EXISTS idx_tdh_word    ON training_data_holdout (word);
"""


def _init_db() -> None:
    with _connect() as conn:
        conn.executescript(_DDL)


def _migrate_news_cache() -> None:
    """Idempotent: add new columns to news_cache for existing databases."""
    new_cols = [
        ("relevancy",    "REAL    DEFAULT NULL"),
        ("event_type",   "TEXT    NOT NULL DEFAULT ''"),
        ("article_type", "TEXT    NOT NULL DEFAULT 'news'"),
    ]
    with _connect() as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(news_cache)").fetchall()}
        for col_name, col_def in new_cols:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE news_cache ADD COLUMN {col_name} {col_def}")
        # Add composite index if missing
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_nc_et_at ON news_cache (event_type, article_type)"
        )


def _migrate_training_data() -> None:
    """Idempotent: add new columns to training_data for existing databases."""
    new_cols = [
        ("topic_match",  "REAL NOT NULL DEFAULT 0.5"),
        ("event_title",  "TEXT NOT NULL DEFAULT ''"),
        ("event_date",   "TEXT NOT NULL DEFAULT ''"),  # ISO date of the event e.g. 2026-03-26
    ]
    with _connect() as conn:
        existing = {row[1] for row in conn.execute("PRAGMA table_info(training_data)").fetchall()}
        for col_name, col_def in new_cols:
            if col_name not in existing:
                conn.execute(f"ALTER TABLE training_data ADD COLUMN {col_name} {col_def}")


_init_db()
_migrate_news_cache()
_migrate_training_data()


# ---------------------------------------------------------------------------
# training_data
# ---------------------------------------------------------------------------

def save_training_row(
    speaker: str,
    word: str,
    event_type: str,
    event_ticker: str,
    hit_rate_lifetime: float,
    hit_rate_recent: float,
    momentum: float,
    avg_freq: float,
    recency: float,
    n_samples_lifetime: int,
    n_samples_recent: int,
    rel_max: float,
    rel_mean: float,
    rel_top3_mean: float,
    rel_count_hi: int,
    rel_n: int,
    kalshi_odds: float,
    ev_score: float,
    did_say_word: int,
    topic_match: float = 0.5,
    event_title: str = "",
    event_date: str = "",
) -> int:
    """Persist one labeled training example. Returns the new row id."""
    sql = """
        INSERT INTO training_data
            (speaker, word, event_type, event_ticker,
             hit_rate_lifetime, hit_rate_recent, momentum, avg_freq, recency,
             n_samples_lifetime, n_samples_recent,
             rel_max, rel_mean, rel_top3_mean, rel_count_hi, rel_n,
             kalshi_odds, ev_score, did_say_word, created_at,
             topic_match, event_title, event_date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with _connect() as conn:
        cur = conn.execute(sql, (
            speaker, word, event_type, event_ticker,
            float(hit_rate_lifetime), float(hit_rate_recent),
            float(momentum), float(avg_freq), float(recency),
            int(n_samples_lifetime), int(n_samples_recent),
            float(rel_max), float(rel_mean), float(rel_top3_mean),
            int(rel_count_hi), int(rel_n),
            float(kalshi_odds), float(ev_score),
            int(did_say_word), _now(),
            float(topic_match), event_title, event_date,
        ))
        return cur.lastrowid


def get_training_data() -> list[dict]:
    """Return all rows from training_data table."""
    with _connect() as conn:
        rows = conn.execute("SELECT * FROM training_data ORDER BY created_at").fetchall()
    return [_row(r) for r in rows]


def save_holdout_row(
    speaker: str,
    word: str,
    event_type: str,
    event_ticker: str,
    hit_rate_lifetime: float,
    hit_rate_recent: float,
    momentum: float,
    avg_freq: float,
    recency: float,
    n_samples_lifetime: int,
    n_samples_recent: int,
    rel_max: float,
    rel_mean: float,
    rel_top3_mean: float,
    rel_count_hi: int,
    rel_n: int,
    kalshi_odds: float,
    ev_score: float,
    did_say_word: int,
    topic_match: float = 0.5,
    event_title: str = "",
    event_date: str = "",
) -> int:
    """Persist one post-cutoff holdout row. NEVER enters training."""
    sql = """
        INSERT INTO training_data_holdout
            (speaker, word, event_type, event_ticker,
             hit_rate_lifetime, hit_rate_recent, momentum, avg_freq, recency,
             n_samples_lifetime, n_samples_recent,
             rel_max, rel_mean, rel_top3_mean, rel_count_hi, rel_n,
             kalshi_odds, ev_score, did_say_word, created_at,
             topic_match, event_title, event_date)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """
    with _connect() as conn:
        cur = conn.execute(sql, (
            speaker, word, event_type, event_ticker,
            float(hit_rate_lifetime), float(hit_rate_recent),
            float(momentum), float(avg_freq), float(recency),
            int(n_samples_lifetime), int(n_samples_recent),
            float(rel_max), float(rel_mean), float(rel_top3_mean),
            int(rel_count_hi), int(rel_n),
            float(kalshi_odds), float(ev_score),
            int(did_say_word), _now(),
            float(topic_match), event_title, event_date,
        ))
        return cur.lastrowid


def get_holdout_data() -> list[dict]:
    """Return all rows from training_data_holdout (post-cutoff, real holdout)."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM training_data_holdout ORDER BY event_date, created_at"
        ).fetchall()
    return [_row(r) for r in rows]


def holdout_stats() -> dict:
    """Quick summary of the holdout set."""
    with _connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM training_data_holdout").fetchone()[0]
        priced = conn.execute(
            "SELECT COUNT(*) FROM training_data_holdout "
            "WHERE kalshi_odds > 0.04 AND kalshi_odds < 0.96"
        ).fetchone()[0]
        warm = conn.execute(
            "SELECT COUNT(*) FROM training_data_holdout WHERE n_samples_lifetime >= 3"
        ).fetchone()[0]
        tickers = conn.execute(
            "SELECT COUNT(DISTINCT event_ticker) FROM training_data_holdout"
        ).fetchone()[0]
    return {"total": total, "priced": priced, "warm": warm, "events": tickers}


# ---------------------------------------------------------------------------
# speaker_profiles
# ---------------------------------------------------------------------------

def get_cached_profile(
    speaker: str,
    word: Optional[str] = None,
    event_type: Optional[str] = None,
) -> list[dict]:
    """
    Return matching speaker profiles (one row per word).

    speaker    — required (always filter by speaker)
    word       — optional; if given, returns only that word's row
    event_type — optional additional filter
    """
    clauses = ["speaker = ?"]
    params: list[Any] = [speaker]

    if word is not None:
        clauses.append("word = ?")
        params.append(word)
    if event_type is not None:
        clauses.append("event_type = ?")
        params.append(event_type)

    sql = ("SELECT * FROM speaker_profiles WHERE "
           + " AND ".join(clauses)
           + " ORDER BY updated_at DESC")

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row(r) for r in rows]


def insert_new_profile(
    speaker: str,
    word: str,
    event_type: str = "",
) -> int:
    """
    Initialise a blank profile for a new (speaker, word, event_type) combo.
    Uses INSERT OR IGNORE so it's safe to call repeatedly.
    Returns the row id (new or existing).
    """
    sql = """
        INSERT OR IGNORE INTO speaker_profiles (speaker, word, event_type, updated_at)
        VALUES (?, ?, ?, ?)
    """
    with _connect() as conn:
        cur = conn.execute(sql, (speaker, word, event_type, _now()))
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM speaker_profiles WHERE speaker=? AND word=? AND event_type=?",
            (speaker, word, event_type),
        ).fetchone()
        return row["id"] if row else -1


def update_profile(
    speaker: str,
    word: str,
    event_type: str,
    hit_rate_lifetime: float,
    hit_rate_recent: float,
    avg_freq: float,
    recency: float,
    n_samples_lifetime: int,
    n_samples_recent: int,
) -> None:
    """
    Overwrite all stats for a (speaker, word, event_type) row.
    Momentum is derived automatically as recent - lifetime.

    Callers must compute hit_rate_recent using KalshiCounter — do NOT
    use LIKE queries on raw text (misses Kalshi plural/possessive rules).
    """
    momentum = hit_rate_recent - hit_rate_lifetime
    sql = """
        UPDATE speaker_profiles
           SET hit_rate_lifetime  = ?,
               hit_rate_recent    = ?,
               momentum           = ?,
               avg_freq           = ?,
               recency            = ?,
               n_samples_lifetime = ?,
               n_samples_recent   = ?,
               updated_at         = ?
         WHERE speaker = ? AND word = ? AND event_type = ?
    """
    with _connect() as conn:
        conn.execute(sql, (
            hit_rate_lifetime, hit_rate_recent, momentum,
            avg_freq, recency,
            n_samples_lifetime, n_samples_recent,
            _now(),
            speaker, word, event_type,
        ))


# ---------------------------------------------------------------------------
# trade_log
# ---------------------------------------------------------------------------

def record_trade(
    ticker: str,
    speaker: str,
    word: str,
    our_probability: float,
    kalshi_odds: float,
    ev_per_contract: float,
    bet_side: str,
    contracts: int,
    event_type: str = "",
    mode: str = "paper",
) -> int:
    """
    Log a new trade (paper or live). Returns the new row id.

    bet_side       — "yes" or "no"
    our_probability — our model's P(YES) estimate
    kalshi_odds    — current Kalshi price for bet_side (0-1)
    ev_per_contract — expected value per contract in dollars
    contracts      — number of contracts placed
    mode           — "paper" (default) or "live"
    """
    sql = """
        INSERT INTO trade_log
            (mode, ticker, speaker, event_type, word,
             our_probability, kalshi_odds, ev_per_contract,
             bet_side, contracts, placed_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _connect() as conn:
        cur = conn.execute(sql, (
            mode.lower(), ticker, speaker, event_type, word,
            float(our_probability), float(kalshi_odds), float(ev_per_contract),
            bet_side.lower(), int(contracts), _now(),
        ))
        return cur.lastrowid


def record_outcome(
    trade_id: int,
    outcome: str,
    payout_cents: float = 0.0,
) -> None:
    """
    Update a trade with its final outcome once the Kalshi market settles.

    outcome — "win", "loss", or "cancelled"
    """
    if outcome not in ("win", "loss", "cancelled"):
        raise ValueError(f"outcome must be 'win', 'loss', or 'cancelled', got {outcome!r}")
    sql = """
        UPDATE trade_log
           SET outcome     = ?,
               payout_cents = ?,
               resolved_at  = ?
         WHERE id = ?
    """
    with _connect() as conn:
        conn.execute(sql, (outcome, float(payout_cents), _now(), trade_id))


# ---------------------------------------------------------------------------
# transcripts
# ---------------------------------------------------------------------------

def get_transcripts(
    speaker: Optional[str] = None,
    event_type: Optional[str] = None,
    event_ticker: Optional[str] = None,
    content_hash: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
) -> list[dict]:
    """Return cached transcripts matching the given filters."""
    clauses: list[str] = []
    params: list[Any] = []

    if speaker is not None:
        clauses.append("speaker = ?")
        params.append(speaker)
    if event_type is not None:
        clauses.append("event_type = ?")
        params.append(event_type)
    if event_ticker is not None:
        clauses.append("event_ticker = ?")
        params.append(event_ticker)
    if content_hash is not None:
        clauses.append("content_hash = ?")
        params.append(content_hash)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM transcripts {where} ORDER BY fetched_at DESC LIMIT ? OFFSET ?"
    params.extend([limit, offset])

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row(r) for r in rows]


def insert_transcript(
    speaker: str,
    full_text: str,
    content_hash: str,
    event_type: str = "",
    event_ticker: str = "",
    source: str = "",
    source_url: str = "",
    event_date: Optional[date] = None,
) -> int:
    """
    Cache a transcript. Silently ignores duplicates (same content_hash).
    Returns the row id (new or existing).
    """
    sql = """
        INSERT OR IGNORE INTO transcripts
            (speaker, event_type, event_ticker, content_hash, full_text,
             source, source_url, event_date, fetched_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    event_date_str = event_date.isoformat() if event_date else None
    with _connect() as conn:
        cur = conn.execute(sql, (
            speaker, event_type, event_ticker, content_hash, full_text,
            source, source_url, event_date_str, _now(),
        ))
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute(
            "SELECT id FROM transcripts WHERE content_hash = ?", (content_hash,)
        ).fetchone()
        return row["id"] if row else -1


def delete_old_transcripts(max_age_weeks: int = 2) -> int:
    """
    Archive transcripts older than max_age_weeks into transcripts_archive,
    then delete them from active storage. Returns the number archived/deleted.

    Archiving (not hard-deleting) preserves data for long-term momentum calc.
    Default: 2 weeks. Pass 3 for 3 weeks.
    """
    cutoff = (datetime.utcnow() - timedelta(weeks=max_age_weeks)).isoformat(timespec="seconds") + "Z"

    archive_sql = """
        INSERT OR IGNORE INTO transcripts_archive
            (speaker, event_type, event_ticker, content_hash, full_text,
             source, source_url, event_date, archived_at)
        SELECT speaker, event_type, event_ticker, content_hash, full_text,
               source, source_url, event_date, ?
          FROM transcripts
         WHERE fetched_at < ?
    """
    delete_sql = "DELETE FROM transcripts WHERE fetched_at < ?"

    with _connect() as conn:
        conn.execute(archive_sql, (_now(), cutoff))
        cur = conn.execute(delete_sql, (cutoff,))
        return cur.rowcount


# ---------------------------------------------------------------------------
# news_cache
# ---------------------------------------------------------------------------

def insert_news_cache(
    query: str,
    title: str,
    url: str,
    published_at: Optional[str] = None,
    source: str = "",
    snippet: str = "",
    relevancy: Optional[float] = None,
    event_type: str = "",
    article_type: str = "news",
) -> int:
    """
    Insert a news article. Silently ignores duplicate URLs.
    Returns the row id (new or existing).
    """
    sql = """
        INSERT OR IGNORE INTO news_cache
            (query, title, url, published_at, source, snippet, fetched_at,
             relevancy, event_type, article_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """
    with _connect() as conn:
        cur = conn.execute(
            sql,
            (query, title, url, published_at, source, snippet, _now(),
             relevancy, event_type, article_type),
        )
        if cur.lastrowid:
            return cur.lastrowid
        row = conn.execute("SELECT id FROM news_cache WHERE url = ?", (url,)).fetchone()
        return row["id"] if row else -1


def get_news_articles(
    query: Optional[str] = None,
    speaker: Optional[str] = None,
    max_age_days: int = 14,
) -> list[dict]:
    """
    Return cached news articles fetched within max_age_days.

    query   — exact match on the RSS query string used to fetch
    speaker — case-insensitive substring match on query OR title
    """
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat(timespec="seconds") + "Z"
    clauses = ["fetched_at >= ?"]
    params: list[Any] = [cutoff]

    if query is not None:
        clauses.append("query = ?")
        params.append(query)
    if speaker is not None:
        pat = f"%{speaker.lower()}%"
        clauses.append("(LOWER(query) LIKE ? OR LOWER(title) LIKE ?)")
        params.extend([pat, pat])

    sql = ("SELECT * FROM news_cache WHERE "
           + " AND ".join(clauses)
           + " ORDER BY fetched_at DESC")

    with _connect() as conn:
        rows = conn.execute(sql, params).fetchall()
    return [_row(r) for r in rows]


def delete_news_cache(max_age_days: int = 14) -> int:
    """Delete news articles older than max_age_days. Returns count deleted."""
    cutoff = (datetime.utcnow() - timedelta(days=max_age_days)).isoformat(timespec="seconds") + "Z"
    with _connect() as conn:
        cur = conn.execute("DELETE FROM news_cache WHERE fetched_at < ?", (cutoff,))
        return cur.rowcount


# TTL (days) keyed by (event_type, article_type). "*" means wildcard.
# More specific rules take priority; ("*", "*") is the fallback.
_NEWS_TTL: dict[tuple[str, str], int] = {
    ("fomc",        "news"):          7,
    ("fomc",        "press_release"): 3,
    ("fomc",        "opinion"):       2,
    ("earnings",    "news"):          3,
    ("earnings",    "opinion"):       1,
    ("earnings",    "press_release"): 3,
    ("sotu",        "news"):         14,
    ("debate",      "news"):          7,
    ("press_conf",  "news"):          5,
    ("un_speech",   "news"):          7,
    ("*",           "opinion"):       2,
    ("*",           "press_release"): 5,
    ("*",           "analysis"):      7,
    ("*",           "*"):            14,
}


def _news_ttl_days(event_type: str, article_type: str) -> int:
    return (
        _NEWS_TTL.get((event_type, article_type))
        or _NEWS_TTL.get(("*", article_type))
        or _NEWS_TTL.get(("*", "*"), 14)
    )


def delete_news_cache_dynamic() -> int:
    """
    Delete news articles using per-(event_type, article_type) TTLs.
    More specific rules run first; the default mops up anything remaining
    that exceeds the fallback TTL.
    Returns total rows deleted.
    """
    total = 0
    # Sort so specific rules (no wildcards) run before wildcard rules.
    ordered = sorted(
        _NEWS_TTL.items(),
        key=lambda kv: (kv[0][0] == "*", kv[0][1] == "*"),
    )
    with _connect() as conn:
        for (et, at), days in ordered:
            cutoff = (
                datetime.utcnow() - timedelta(days=days)
            ).isoformat(timespec="seconds") + "Z"
            if et == "*" and at == "*":
                cur = conn.execute(
                    "DELETE FROM news_cache WHERE fetched_at < ?", (cutoff,)
                )
            elif et == "*":
                cur = conn.execute(
                    "DELETE FROM news_cache WHERE article_type=? AND fetched_at < ?",
                    (at, cutoff),
                )
            elif at == "*":
                cur = conn.execute(
                    "DELETE FROM news_cache WHERE event_type=? AND fetched_at < ?",
                    (et, cutoff),
                )
            else:
                cur = conn.execute(
                    "DELETE FROM news_cache WHERE event_type=? AND article_type=? AND fetched_at < ?",
                    (et, at, cutoff),
                )
            total += cur.rowcount
    return total


# ---------------------------------------------------------------------------
# embeddings_cache
# ---------------------------------------------------------------------------

def save_embedding(target_id: int, table_reference: str, vector: bytes) -> int:
    """
    Upsert a semantic vector blob. table_reference is the source table name
    ('speaker_profiles' or 'transcripts'). Returns the row id.
    """
    sql = """
        INSERT INTO embeddings_cache (target_id, table_reference, vector, created_at)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(target_id, table_reference) DO UPDATE SET
            vector     = excluded.vector,
            created_at = excluded.created_at
    """
    with _connect() as conn:
        cur = conn.execute(sql, (target_id, table_reference, vector, _now()))
        return cur.lastrowid


def get_cached_embedding(
    target_id: int,
    table_reference: str,
) -> Optional[dict]:
    """Return the stored embedding for (target_id, table_reference), or None."""
    sql = """
        SELECT * FROM embeddings_cache
         WHERE target_id = ? AND table_reference = ?
    """
    with _connect() as conn:
        row = conn.execute(sql, (target_id, table_reference)).fetchone()
    return _row(row) if row else None
