"""
rebuild_profiles_from_training.py
----------------------------------
Computes speaker profiles directly from training_data outcomes and refreshes
cold training rows (n_samples_lifetime < 3) with the updated stats.

No web scraping needed — uses the did_say_word outcomes already in the DB.

Steps:
  1. Aggregate training_data by (speaker, word) to compute hit rates, n_samples.
  2. Write those aggregates into speaker_profiles.
  3. Refresh cold training_data rows so the model sees real n_samples_lifetime.

Run this after any new harvest to keep profiles in sync:
    python3 rebuild_profiles_from_training.py
"""

from __future__ import annotations

import math
from datetime import date, datetime, timezone

import db


_RECENT_DAYS = 90   # same window as the model's "recent" definition


def _today() -> str:
    return date.today().isoformat()


def _days_since(date_str: str) -> float:
    if not date_str:
        return 180.0
    try:
        d = datetime.fromisoformat(date_str).replace(tzinfo=timezone.utc)
    except ValueError:
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        except ValueError:
            return 180.0
    return (datetime.now(tz=timezone.utc) - d).total_seconds() / 86_400


def rebuild() -> None:
    rows = db.get_training_data()
    if not rows:
        print("No training data found.")
        return

    print(f"Loaded {len(rows)} training rows.")

    # ── 1. Aggregate by (speaker, word, event_type) ─────────────────────────
    # Use event_type="" (global) as the profile key so profiles are usable
    # across event types (same as the harvest script does).

    from collections import defaultdict

    # Group: (speaker, word) → list of (event_date, did_say_word)
    groups: dict[tuple, list] = defaultdict(list)
    for r in rows:
        key = (r["speaker"], r["word"])
        groups[key].append({
            "event_date":   r.get("event_date", ""),
            "did_say_word": int(r["did_say_word"]),
        })

    today_str = _today()
    cutoff_date = _subtract_days(today_str, _RECENT_DAYS)

    # Only skip (speaker, word) pairs that already have n_samples_lifetime >= 3
    # in the profiles — those are high-quality transcript-based entries we
    # don't want to overwrite.  Words with no profile or n_samples < 3 get
    # rebuilt from training_data outcomes.
    with db._connect() as conn:
        strong_profiles = {
            (r[0], r[1])
            for r in conn.execute(
                "SELECT speaker, word FROM speaker_profiles "
                "WHERE n_samples_lifetime >= 3"
            ).fetchall()
        }

    print(f"  Keeping {len(strong_profiles)} strong (n≥3) existing profiles.")
    print(f"\nRebuild profiles for weak/missing (speaker, word) pairs ...")

    updated = 0
    skipped = 0

    for (speaker, word), events in sorted(groups.items()):
        if (speaker, word) in strong_profiles:
            skipped += 1
            continue
        n_total   = len(events)
        n_yes     = sum(e["did_say_word"] for e in events)
        hit_rate  = n_yes / n_total

        # Recent: events in last 90 days
        recent_events = [e for e in events
                         if e["event_date"] and e["event_date"] >= cutoff_date]
        if recent_events:
            n_rec_yes   = sum(e["did_say_word"] for e in recent_events)
            hit_recent  = n_rec_yes / len(recent_events)
            n_recent    = len(recent_events)
        else:
            hit_recent  = hit_rate
            n_recent    = 0

        # Recency: days since most recent event (0 = just happened, 1 = very old)
        dates = [e["event_date"] for e in events if e["event_date"]]
        if dates:
            latest = max(dates)
            days   = _days_since(latest)
            # Exponential decay with 180-day half-life → [0, 1]
            recency = math.exp(-days * math.log(2) / 180.0)
        else:
            recency = 0.5

        # avg_freq: average mentions per event (we only know yes/no, not count)
        # Use hit_rate as a proxy (1.0 when always said, 0.0 when never)
        avg_freq = float(n_yes) / max(n_total, 1)

        # Write profile
        db.insert_new_profile(speaker=speaker, word=word, event_type="")
        db.update_profile(
            speaker            = speaker,
            word               = word,
            event_type         = "",
            hit_rate_lifetime  = round(hit_rate,  4),
            hit_rate_recent    = round(hit_recent, 4),
            avg_freq           = round(avg_freq,   4),
            recency            = round(recency,    4),
            n_samples_lifetime = n_total,
            n_samples_recent   = n_recent,
        )
        updated += 1

    print(f"  Updated {updated} profiles.")

    # ── 2. Refresh cold training rows from updated profiles ──────────────────
    print(f"\nRefreshing cold training_data rows (n_samples_lifetime < 3) ...")

    with db._connect() as conn:
        cold_rows = conn.execute(
            "SELECT id, speaker, word FROM training_data WHERE n_samples_lifetime < 3"
        ).fetchall()

    print(f"  Found {len(cold_rows)} cold rows to refresh.")
    refreshed = 0
    still_cold = 0

    with db._connect() as conn:
        for row in cold_rows:
            row_id  = row["id"]
            speaker = row["speaker"]
            word    = row["word"]

            profs = db.get_cached_profile(speaker, word=word)
            if not profs:
                still_cold += 1
                continue

            p = profs[0]
            n = int(p.get("n_samples_lifetime") or 0)
            if n < 3:
                still_cold += 1
                continue

            conn.execute("""
                UPDATE training_data
                   SET hit_rate_lifetime  = ?,
                       hit_rate_recent    = ?,
                       momentum           = ?,
                       avg_freq           = ?,
                       recency            = ?,
                       n_samples_lifetime = ?,
                       n_samples_recent   = ?,
                       ev_score           = hit_rate_lifetime - kalshi_odds
                 WHERE id = ?
            """, (
                float(p["hit_rate_lifetime"]),
                float(p["hit_rate_recent"]),
                float(p["hit_rate_recent"]) - float(p["hit_rate_lifetime"]),
                float(p["avg_freq"]),
                float(p["recency"]),
                n,
                int(p.get("n_samples_recent") or 0),
                row_id,
            ))
            refreshed += 1

    print(f"  Refreshed: {refreshed} rows  |  Still cold: {still_cold}")

    # ── 3. Summary ───────────────────────────────────────────────────────────
    with db._connect() as conn:
        total = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        warm  = conn.execute(
            "SELECT COUNT(*) FROM training_data WHERE n_samples_lifetime >= 3"
        ).fetchone()[0]
        cold  = total - warm

    print(f"\n{'=' * 50}")
    print("REBUILD COMPLETE")
    print(f"{'=' * 50}")
    print(f"  Total rows  : {total}")
    print(f"  Warm (≥3)   : {warm}  ({100*warm/total:.1f}%)")
    print(f"  Cold (<3)   : {cold}")
    print()
    print("Next: run `python3 kalshi_model.py` to retrain with updated warm rows.")


def _subtract_days(date_str: str, days: int) -> str:
    """Return the date string `days` before date_str."""
    try:
        d = datetime.strptime(date_str, "%Y-%m-%d")
        result = d.replace(year=d.year, month=d.month, day=d.day)
        import datetime as dt_mod
        result = (dt_mod.date.fromisoformat(date_str) -
                  dt_mod.timedelta(days=days)).isoformat()
        return result
    except Exception:
        return ""


if __name__ == "__main__":
    rebuild()
