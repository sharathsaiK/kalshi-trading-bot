"""
Backfill news relevancy features for training rows that have rel_n = 0.

Strategy: fetch articles once per event (not per word), then re-score each
word against those articles. 110 API calls instead of 2,346.
"""
from __future__ import annotations

import time
import datetime
from typing import Optional

import db
import news_scraper


def _relevancy_with_event_date(
    article: dict,
    speaker: str,
    word: str,
    event_name: str,
    event_date: str,
) -> float:
    """Compute relevancy using event_date as reference (not today)."""
    try:
        ref = datetime.datetime.fromisoformat(event_date).replace(
            tzinfo=datetime.timezone.utc
        )
    except Exception:
        ref = None

    title = article.get("title", "")
    source = article.get("source", "")
    pub = article.get("published_at")

    import math
    recency = 0.5
    if pub and ref:
        try:
            pub_dt = datetime.datetime.fromisoformat(pub.rstrip("Z")).replace(
                tzinfo=datetime.timezone.utc
            )
            age_days = max(0.0, (ref - pub_dt).total_seconds() / 86400)
            recency = math.exp(-age_days / 7.0)
        except ValueError:
            pass

    authority = news_scraper._source_authority(source)
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


def run(dry_run: bool = False, delay_s: float = 1.0) -> None:
    with db._connect() as conn:
        events = conn.execute("""
            SELECT DISTINCT event_ticker, event_title, event_date, speaker
            FROM training_data
            WHERE rel_n = 0 AND event_date IS NOT NULL AND event_title IS NOT NULL
            ORDER BY event_date
        """).fetchall()

    print(f"Events to scrape: {len(events)}")
    updated_total = 0

    for i, (ticker, title, event_date, speaker) in enumerate(events):
        print(f"\n[{i+1}/{len(events)}] {event_date}  {speaker}  {ticker}")

        try:
            d = datetime.date.fromisoformat(event_date[:10])
        except ValueError:
            print("  skip: bad date")
            continue

        date_from = datetime.datetime(d.year, d.month, d.day,
                                      tzinfo=datetime.timezone.utc) - datetime.timedelta(days=14)
        date_to   = datetime.datetime(d.year, d.month, d.day,
                                      tzinfo=datetime.timezone.utc) + datetime.timedelta(days=1)

        try:
            articles = news_scraper.fetch_news(
                speaker=speaker,
                event_name=title,
                word=None,
                date_from=date_from,
                date_to=date_to,
                max_results=20,
                persist=True,
            )
        except Exception as exc:
            print(f"  fetch error: {exc}")
            time.sleep(delay_s * 2)
            continue

        print(f"  fetched {len(articles)} articles")
        if not articles:
            time.sleep(delay_s)
            continue

        # Get all words for this event
        with db._connect() as conn:
            words = conn.execute(
                "SELECT word FROM training_data WHERE event_ticker = ? AND rel_n = 0",
                (ticker,),
            ).fetchall()

        updated_event = 0
        for (word,) in words:
            scores = [
                _relevancy_with_event_date(a, speaker, word, title, event_date)
                for a in articles
            ]
            scores = [s for s in scores if s is not None]
            if not scores:
                continue

            top3 = sorted(scores, reverse=True)[:3]
            feats = {
                "rel_max":      round(max(scores), 4),
                "rel_mean":     round(sum(scores) / len(scores), 4),
                "rel_count_hi": sum(1 for s in scores if s >= 0.5),
                "rel_n":        len(scores),
            }

            if not dry_run:
                with db._connect() as conn:
                    conn.execute("""
                        UPDATE training_data
                        SET rel_max=?, rel_mean=?, rel_count_hi=?, rel_n=?
                        WHERE event_ticker=? AND word=?
                    """, (
                        feats["rel_max"], feats["rel_mean"],
                        feats["rel_count_hi"], feats["rel_n"],
                        ticker, word,
                    ))
            updated_event += 1

        updated_total += updated_event
        print(f"  updated {updated_event} words  (running total: {updated_total})")
        time.sleep(delay_s)

    print(f"\nDone. Total rows updated: {updated_total}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delay", type=float, default=1.0,
                   help="seconds between event fetches (default 1.0)")
    args = p.parse_args()
    run(dry_run=args.dry_run, delay_s=args.delay)
