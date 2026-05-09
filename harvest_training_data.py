"""
harvest_training_data.py
------------------------
Backfill real labeled training rows from settled Kalshi markets.

Uses historical candlestick API for pre-event mid-prices (trade history is
purged for finalized markets). Loops multiple speakers in one run.

  - Pre-event price: median of pre-event-day hourly candle means / mid-prices
  - event_date: derived from ticker, used for HOLDOUT_CUTOFF check
  - event_type: derived from event title via topic_match.classify_event
  - Skips events on or after HOLDOUT_CUTOFF so holdout set stays clean
"""

import time

import db
import kalshi_api
import topic_match
from kalshi_model import HOLDOUT_CUTOFF


# Speakers we have profiles for and want to harvest events for.
# Order roughly by expected event volume (Trump first → most rows).
DEFAULT_SPEAKERS = [
    # High-volume speakers first
    "Donald Trump",
    "J.D. Vance",
    "Jerome Powell",
    "Marco Rubio",
    "Michael Barr",
    # Friend's additional speakers (Fed governors + Treasury)
    "Scott Bessent",
    "Michelle Bowman",
    "John Williams",
    "Lisa Cook",
    "Philip Jefferson",
]


def _harvest_one(
    speaker_name: str,
    max_events: int,
    holdout_mode: bool,
    verbose: bool,
    allow_no_price: bool = False,
) -> dict:
    """
    Core per-speaker harvest loop used by both training and holdout modes.

    holdout_mode=False → collects pre-cutoff events into training_data.
    holdout_mode=True  → collects post-cutoff events into training_data_holdout.

    allow_no_price=True → store rows even when no real pre-event price is
    available, using the settlement value (0.99/0.01) which the model will
    mask to NaN. Labels (did_say_word) are still real. Useful for boosting
    word priors and n_samples_lifetime without synthetic prices.
    """
    if verbose:
        mode_tag = "[HOLDOUT]" if holdout_mode else "[TRAIN]"
        print(f"\n{'=' * 60}")
        print(f"{mode_tag} Harvesting events for: {speaker_name}")
        print(f"{'=' * 60}")

    try:
        events = kalshi_api.find_speaker_events(speaker_name, max_events=max_events)
    except Exception as ex:
        print(f"  [error] find_speaker_events failed: {ex}")
        return {"new_rows": 0, "new_events": 0, "skipped_cutoff": 0,
                "skipped_no_price": 0, "skipped_no_settled": 0}

    if verbose:
        print(f"Found {len(events)} candidate events")

    # Build set of already-collected tickers for the target table
    table = "training_data_holdout" if holdout_mode else "training_data"
    with db._connect() as conn:
        existing = {r[0] for r in conn.execute(
            f"SELECT DISTINCT event_ticker FROM {table}"
        ).fetchall()}

    new_rows = 0
    new_events = 0
    skipped_cutoff = 0
    skipped_no_price = 0
    skipped_no_settled = 0

    for e in events:
        ticker = e.get("event_ticker", "")
        if not ticker or ticker in existing:
            continue

        event_date = kalshi_api._ticker_to_date(ticker) or ""

        if holdout_mode:
            # Only want post-cutoff events
            if not event_date or event_date < HOLDOUT_CUTOFF:
                skipped_cutoff += 1
                continue
        else:
            # Only want pre-cutoff events
            if event_date and event_date >= HOLDOUT_CUTOFF:
                skipped_cutoff += 1
                continue

        title = e.get("title") or e.get("sub_title") or ""
        event_type = topic_match.classify_event(title)

        try:
            markets = kalshi_api.get_event_markets(ticker, historical=True)
            if not markets:
                markets = kalshi_api.get_event_markets(ticker, historical=False)
            time.sleep(0.4)
        except Exception as ex:
            if verbose:
                print(f"  [skip] {ticker}: {ex}")
            time.sleep(2)
            continue

        settled = [m for m in markets if m.word and m.result in ("yes", "no")]
        if not settled:
            skipped_no_settled += 1
            continue

        rows_added = 0
        for m in settled:
            pre_price = kalshi_api.get_pre_event_mid_price(m.ticker, event_date)
            if pre_price <= 0.04 or pre_price >= 0.96:
                if 0.04 < m.previous_yes_ask < 0.96:
                    pre_price = m.previous_yes_ask
                elif 0.04 < m.previous_yes_bid < 0.96:
                    pre_price = m.previous_yes_bid
                elif 0.04 < m.previous_price < 0.96:
                    pre_price = m.previous_price
                elif allow_no_price:
                    # Store settlement value — model masks extremes to NaN.
                    # Label is still real; row contributes to word priors and
                    # n_samples_lifetime without fabricating any price.
                    pre_price = 0.99 if m.result == "yes" else 0.01
                    skipped_no_price += 1
                else:
                    skipped_no_price += 1
                    continue

            profs = db.get_cached_profile(speaker_name, word=m.word)
            prof = profs[0] if profs else {}
            hl = float(prof.get("hit_rate_lifetime") or 0.5)
            tm_score = topic_match.compute_match_safe(title, m.word)

            row_kwargs = dict(
                speaker            = speaker_name,
                word               = m.word,
                event_type         = event_type,
                event_ticker       = ticker,
                hit_rate_lifetime  = hl,
                hit_rate_recent    = float(prof.get("hit_rate_recent") or hl),
                momentum           = float(prof.get("momentum")        or 0.0),
                avg_freq           = float(prof.get("avg_freq")        or 1.0),
                recency            = float(prof.get("recency")         or 0.5),
                n_samples_lifetime = int(prof.get("n_samples_lifetime") or 0),
                n_samples_recent   = int(prof.get("n_samples_recent")   or 0),
                rel_max=0.0, rel_mean=0.0, rel_top3_mean=0.0,
                rel_count_hi=0, rel_n=0,
                kalshi_odds        = pre_price,
                ev_score           = float(hl - pre_price),
                did_say_word       = int(m.result == "yes"),
                topic_match        = tm_score,
                event_title        = title,
                event_date         = event_date,
            )

            if holdout_mode:
                db.save_holdout_row(**row_kwargs)
            else:
                db.save_training_row(**row_kwargs)
            rows_added += 1

            time.sleep(0.5)

        if rows_added > 0:
            new_rows += rows_added
            new_events += 1
            if verbose:
                print(f"  [+] {ticker:<40} {rows_added:>3} rows  ({title[:42]})")

    if verbose:
        cutoff_label = "pre-cutoff skipped" if holdout_mode else "holdout skipped"
        print(f"\n  {speaker_name}: +{new_rows} rows from {new_events} events "
              f"({skipped_cutoff} {cutoff_label}, "
              f"{skipped_no_settled} unsettled, "
              f"{skipped_no_price} no-price)")

    return {
        "new_rows": new_rows,
        "new_events": new_events,
        "skipped_cutoff": skipped_cutoff,
        "skipped_no_price": skipped_no_price,
        "skipped_no_settled": skipped_no_settled,
    }


def harvest(speaker_name: str, max_events: int = 200, verbose: bool = True,
            allow_no_price: bool = False) -> dict:
    """Harvest pre-cutoff training rows for one speaker."""
    return _harvest_one(speaker_name, max_events=max_events,
                        holdout_mode=False, verbose=verbose,
                        allow_no_price=allow_no_price)


def harvest_holdout(speaker_name: str, max_events: int = 200, verbose: bool = True,
                    allow_no_price: bool = False) -> dict:
    """Harvest post-cutoff holdout rows for one speaker."""
    return _harvest_one(speaker_name, max_events=max_events,
                        holdout_mode=True, verbose=verbose,
                        allow_no_price=allow_no_price)


def harvest_all(speakers: list[str] = None, max_events: int = 500,
                holdout_mode: bool = False, allow_no_price: bool = False) -> None:
    speakers = speakers or DEFAULT_SPEAKERS
    table = "training_data_holdout" if holdout_mode else "training_data"

    with db._connect() as conn:
        before = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]

    mode_label = "HOLDOUT (post-cutoff)" if holdout_mode else "TRAINING (pre-cutoff)"
    print(f"\nStarting {mode_label} harvest. Current {table} rows: {before}")
    print(f"Speakers: {', '.join(speakers)}")
    print(f"Holdout cutoff: {HOLDOUT_CUTOFF}\n")

    for sp in speakers:
        try:
            _harvest_one(sp, max_events=max_events,
                         holdout_mode=holdout_mode, verbose=True,
                         allow_no_price=allow_no_price)
        except Exception as ex:
            print(f"  [error] harvest({sp}) failed: {ex}")

    with db._connect() as conn:
        after = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        real_price = conn.execute(
            f"SELECT COUNT(*) FROM {table} "
            "WHERE kalshi_odds > 0.04 AND kalshi_odds < 0.96"
        ).fetchone()[0]

    print(f"\n{'=' * 60}")
    print(f"HARVEST COMPLETE — {mode_label}")
    print(f"{'=' * 60}")
    print(f"  Rows before:        {before}")
    print(f"  Rows after:         {after}")
    print(f"  Net new rows:       {after - before}")
    print(f"  Rows w/ real price: {real_price}")
    print(f"  Speakers covered:   {len(speakers)}")

    if holdout_mode:
        stats = db.holdout_stats()
        print(f"\n  Holdout set stats:")
        print(f"    Total rows:   {stats['total']}")
        print(f"    Events:       {stats['events']}")
        print(f"    Priced rows:  {stats['priced']}")
        print(f"    Warm rows:    {stats['warm']}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    holdout_mode    = "--holdout"        in args
    allow_no_price  = "--allow-no-price" in args
    args = [a for a in args if a not in ("--holdout", "--allow-no-price")]

    if args and args[0] == "--all":
        harvest_all(holdout_mode=holdout_mode, allow_no_price=allow_no_price)
    elif args:
        harvest_all(speakers=args, holdout_mode=holdout_mode,
                    allow_no_price=allow_no_price)
    else:
        harvest_all(holdout_mode=holdout_mode, allow_no_price=allow_no_price)
