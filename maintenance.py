"""
maintenance.py
--------------
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import db

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s  [maintenance]  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("maintenance")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DAILY_INTERVAL_SEC  = 24 * 60 * 60   # 24 hours
_WEEKLY_INTERVAL_SEC =  7 * 24 * 60 * 60   # 7 days
_TRANSCRIPT_MAX_AGE_WEEKS = 2
_NEWS_MAX_AGE_DAYS        = 14


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------

def daily_health_check() -> None:
    """
    Log a snapshot of the database — how many records, how many are stale.
    Does NOT delete anything.
    """
    log.info("── Daily health check ──────────────────────────────────")

    try:
        conn = sqlite3.connect(str(db._DB_PATH))

        # Transcript counts
        total_tr = conn.execute("SELECT COUNT(*) FROM transcripts").fetchone()[0]
        stale_tr = conn.execute(
            "SELECT COUNT(*) FROM transcripts WHERE fetched_at < datetime('now', ?)",
            (f"-{_TRANSCRIPT_MAX_AGE_WEEKS * 7} days",)
        ).fetchone()[0]

        # News counts
        total_nc = conn.execute("SELECT COUNT(*) FROM news_cache").fetchone()[0]
        stale_nc = conn.execute(
            "SELECT COUNT(*) FROM news_cache WHERE fetched_at < datetime('now', ?)",
            (f"-{_NEWS_MAX_AGE_DAYS} days",)
        ).fetchone()[0]

        # Speaker profile count
        total_sp = conn.execute("SELECT COUNT(*) FROM speaker_profiles").fetchone()[0]
        speakers = conn.execute(
            "SELECT COUNT(DISTINCT speaker) FROM speaker_profiles"
        ).fetchone()[0]

        # Trade log counts
        total_tl   = conn.execute("SELECT COUNT(*) FROM trade_log").fetchone()[0]
        unsettled  = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE outcome IS NULL"
        ).fetchone()[0]

        # Archive size
        archived = conn.execute(
            "SELECT COUNT(*) FROM transcripts_archive"
        ).fetchone()[0]

        conn.close()

        log.info(f"  transcripts    : {total_tr} total  |  {stale_tr} stale (>{_TRANSCRIPT_MAX_AGE_WEEKS}w)")
        log.info(f"  news_cache     : {total_nc} total  |  {stale_nc} stale (>{_NEWS_MAX_AGE_DAYS}d)")
        log.info(f"  speaker profiles: {total_sp} rows across {speakers} speaker(s)")
        log.info(f"  trade_log      : {total_tl} total  |  {unsettled} unsettled")
        log.info(f"  archive        : {archived} transcript(s)")

        if stale_tr > 0 or stale_nc > 0:
            log.info(f"  ⚠  {stale_tr} stale transcript(s) and {stale_nc} stale news "
                     f"article(s) pending — will be cleaned on next weekly run")

    except Exception as exc:
        log.error(f"Health check failed: {exc}")

    log.info("────────────────────────────────────────────────────────")


def weekly_cleanup() -> None:
    """
    Archive + delete old transcripts and purge old news cache entries.
    """
    log.info("── Weekly cleanup ──────────────────────────────────────")

    try:
        deleted_tr = db.delete_old_transcripts(max_age_weeks=_TRANSCRIPT_MAX_AGE_WEEKS)
        log.info(f"  transcripts : archived + deleted {deleted_tr} row(s) "
                 f"older than {_TRANSCRIPT_MAX_AGE_WEEKS} week(s)")
    except Exception as exc:
        log.error(f"  transcript cleanup failed: {exc}")

    try:
        deleted_nc = db.delete_news_cache_dynamic()
        log.info(f"  news_cache  : deleted {deleted_nc} article(s) (dynamic TTL by event/article type)")
    except Exception as exc:
        log.error(f"  news cache cleanup failed: {exc}")

    log.info("────────────────────────────────────────────────────────")


# ---------------------------------------------------------------------------
# Scheduler
# ---------------------------------------------------------------------------

class _RepeatingTimer:
    """
    Calls `func` every `interval` seconds in a background daemon thread.
    Runs `func` immediately on start, then on the interval.
    """

    def __init__(self, interval: float, func, name: str):
        self.interval = interval
        self.func     = func
        self.name     = name
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()
        log.info(f"Scheduled '{self.name}' every {self.interval:.0f}s")

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        # Fire immediately on first start
        self._safe_call()
        while not self._stop.wait(timeout=self.interval):
            self._safe_call()

    def _safe_call(self) -> None:
        try:
            self.func()
        except Exception as exc:
            log.error(f"'{self.name}' raised an exception: {exc}")


def run_scheduler(
    daily_interval:  float = _DAILY_INTERVAL_SEC,
    weekly_interval: float = _WEEKLY_INTERVAL_SEC,
) -> None:
    """
    Start both jobs and block the main thread until KeyboardInterrupt.
    Both timers are daemon threads — they die if the main process exits.
    """
    log.info("Maintenance scheduler starting ...")
    log.info(f"  daily  health check : every {daily_interval  / 3600:.1f} h")
    log.info(f"  weekly cleanup      : every {weekly_interval / 3600:.1f} h")

    daily_timer  = _RepeatingTimer(daily_interval,  daily_health_check, "daily-health")
    weekly_timer = _RepeatingTimer(weekly_interval, weekly_cleanup,     "weekly-cleanup")

    daily_timer.start()
    weekly_timer.start()

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Shutting down maintenance scheduler ...")
        daily_timer.stop()
        weekly_timer.stop()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--now",    action="store_true",
                   help="Run health check + cleanup once right now and exit")
    p.add_argument("--daily",  type=float, default=_DAILY_INTERVAL_SEC,
                   help="Override daily interval in seconds (for testing)")
    p.add_argument("--weekly", type=float, default=_WEEKLY_INTERVAL_SEC,
                   help="Override weekly interval in seconds (for testing)")
    args = p.parse_args()

    if args.now:
        log.info("Running one-shot maintenance ...")
        daily_health_check()
        weekly_cleanup()
        log.info("Done.")
        return

    run_scheduler(daily_interval=args.daily, weekly_interval=args.weekly)


if __name__ == "__main__":
    main()
