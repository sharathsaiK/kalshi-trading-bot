"""
backfill_topic_match.py
-----------------------
One-time backfill: compute topic_match for all existing training_data rows.

For each unique event_ticker:
  1. Fetch event title from Kalshi API
  2. For each row with that ticker, compute topic_match(title, word)
  3. Update the row in-place

Run once after adding the topic_match column.
"""

import db
import kalshi_api
import topic_match


def backfill():
    with db._connect() as conn:
        # Re-backfill all rows to pick up topic_match improvements
        rows = conn.execute("""
            SELECT id, event_ticker, word, event_title
            FROM training_data
        """).fetchall()

    if not rows:
        print("No rows to backfill.")
        return

    print(f"Backfilling {len(rows)} rows ...")

    # Cache event titles per ticker (avoid duplicate API calls)
    title_cache: dict[str, str] = {}

    n_updated = 0
    for row in rows:
        row_id, event_ticker, word = row[0], row[1], row[2]
        existing_title = row[3] if len(row) > 3 else ""

        # If we already have the title cached on the row, use it
        if existing_title and event_ticker not in title_cache:
            title_cache[event_ticker] = existing_title

        # Otherwise fetch from API
        if event_ticker not in title_cache:
            try:
                meta = kalshi_api.get_event_meta(event_ticker)
                title = (
                    meta.get("title") or
                    meta.get("sub_title") or
                    meta.get("subtitle") or
                    ""
                )
                title_cache[event_ticker] = title
                print(f"  fetched: {event_ticker:<35} → {title[:60]}")
            except Exception as e:
                print(f"  failed:  {event_ticker:<35} → {e}")
                title_cache[event_ticker] = ""

        title = title_cache[event_ticker]
        if not title:
            continue

        match_score = topic_match.compute_match_safe(title, word)

        with db._connect() as conn:
            conn.execute(
                "UPDATE training_data SET topic_match = ?, event_title = ? WHERE id = ?",
                (float(match_score), title, row_id),
            )
        n_updated += 1

    print(f"\nUpdated {n_updated}/{len(rows)} rows.")
    print(f"Unique events: {len(title_cache)}")


if __name__ == "__main__":
    backfill()
