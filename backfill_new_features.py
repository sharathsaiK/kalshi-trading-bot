"""
Backfill ported-feature columns (news decay/cooccur/velocity/polarity/tone +
price velocity) for existing training_data and training_data_holdout rows.

Strategy: fetch news once per event (not per word) and re-score each word
against those articles — same pattern as backfill_news.py. Also fetches each
event's market list once to map word -> per-word market ticker, needed for
ko_velocity (candlestick data is per-market, not per-event).

Historical backfill uses only GDELT/Guardian/NYT — NewsAPI's free tier only
covers a 30-day window and can't reach back into older training data.

word_semantic_proximity needs no backfill: it's computed live from
(speaker, word) pairs already in training_data.
"""
from __future__ import annotations

import time
import datetime

import db
import news_scraper
import kalshi_api
import kalshi_model as km

# Restrict to sources with real historical depth for this backfill.
news_scraper._ADAPTERS = [
    (name, fn) for name, fn in news_scraper._ADAPTERS
    if name in ("gdelt", "guardian", "nyt")
]


def _backfill_table(table: str, dry_run: bool, delay_s: float, limit: int | None = None) -> int:
    with db._connect() as conn:
        events = conn.execute(f"""
            SELECT DISTINCT event_ticker, event_title, event_date, speaker
            FROM {table}
            WHERE news_decay_score IS NULL
              AND event_date IS NOT NULL AND event_date != ''
              AND event_ticker IS NOT NULL AND event_ticker != ''
            ORDER BY event_date
        """).fetchall()

    if limit:
        events = events[:limit]

    print(f"[{table}] events to backfill: {len(events)}")
    updated_total = 0

    for i, (ticker, title, event_date, speaker) in enumerate(events):
        print(f"[{table}] [{i+1}/{len(events)}] {event_date}  {speaker}  {ticker}")

        try:
            d = datetime.date.fromisoformat(event_date[:10])
        except ValueError:
            print("  skip: bad date")
            continue

        ref_dt = datetime.datetime(d.year, d.month, d.day, tzinfo=datetime.timezone.utc)
        date_from = ref_dt - datetime.timedelta(days=14)
        date_to   = ref_dt + datetime.timedelta(days=1)
        ref_ts    = int(ref_dt.timestamp())

        # ── News: fetch once per event ──────────────────────────────────
        try:
            articles = news_scraper.fetch_news(
                speaker=speaker, event_name=title, word=None,
                date_from=date_from, date_to=date_to,
                max_results=20, persist=True,
            )
        except Exception as exc:
            print(f"  news fetch error: {exc}")
            articles = []
            time.sleep(delay_s * 2)

        # ── Markets: fetch once per event, map word -> per-word ticker ──
        word_ticker_map: dict[str, str] = {}
        try:
            markets = kalshi_api.get_event_markets(ticker, historical=True)
            word_ticker_map = {m.word: m.ticker for m in markets if m.word}
        except Exception as exc:
            print(f"  market fetch error: {exc}")

        with db._connect() as conn:
            rows = conn.execute(
                f"SELECT DISTINCT word, kalshi_odds FROM {table} "
                f"WHERE event_ticker=? AND news_decay_score IS NULL",
                (ticker,),
            ).fetchall()

        updated_event = 0
        for word, kalshi_odds in rows:
            news_feats = km._compute_raw_news_features(articles, speaker, word, date_to=ref_dt)

            v24 = v48 = float("nan")
            word_ticker = word_ticker_map.get(word)
            if word_ticker is not None and kalshi_odds is not None:
                try:
                    v24, v48 = km._compute_ko_velocity(word_ticker, float(kalshi_odds), ref_ts=ref_ts)
                except Exception as exc:
                    print(f"    velocity error for {word!r}: {exc}")

            def _n(v):
                return None if v is None or v != v else float(v)  # v!=v catches NaN

            if not dry_run:
                with db._connect() as conn:
                    conn.execute(f"""
                        UPDATE {table}
                        SET news_decay_score=?, news_cooccur_rate=?, news_velocity=?,
                            news_title_polarity=?, news_tone_mean=?,
                            ko_velocity_24h=?, ko_velocity_48h=?
                        WHERE event_ticker=? AND word=?
                    """, (
                        _n(news_feats["news_decay_score"]), _n(news_feats["news_cooccur_rate"]),
                        _n(news_feats["news_velocity"]), _n(news_feats["news_title_polarity"]),
                        _n(news_feats["news_tone_mean"]), _n(v24), _n(v48),
                        ticker, word,
                    ))
            updated_event += 1

        updated_total += updated_event
        print(f"  updated {updated_event} words  (running total: {updated_total})")
        time.sleep(delay_s)

    print(f"[{table}] done. Total rows updated: {updated_total}")
    return updated_total


def run(dry_run: bool = False, delay_s: float = 1.0, limit: int | None = None) -> None:
    total = 0
    total += _backfill_table("training_data", dry_run, delay_s, limit)
    total += _backfill_table("training_data_holdout", dry_run, delay_s, limit)
    print(f"\nGRAND TOTAL rows updated: {total}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--delay", type=float, default=1.0,
                    help="seconds between event fetches (default 1.0)")
    p.add_argument("--limit", type=int, default=None,
                    help="only backfill the first N events per table (for testing)")
    args = p.parse_args()
    run(dry_run=args.dry_run, delay_s=args.delay, limit=args.limit)
