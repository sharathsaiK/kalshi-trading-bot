"""
restore_kalshi_odds.py
----------------------
Revert kalshi_odds in training_data back to the deterministic settlement
values (0.99 if did_say=1, 0.01 if did_say=0). Used to undo the synthetic
randomization in fix_training_prices.py.

Recomputes ev_score = hit_rate_lifetime - kalshi_odds.
"""

import db


def restore():
    with db._connect() as conn:
        rows = conn.execute("""
            SELECT id, hit_rate_lifetime, did_say_word
            FROM training_data
        """).fetchall()

    print(f"Restoring kalshi_odds on {len(rows)} rows ...")

    n_done = 0
    for row in rows:
        rid, hl, did_say = row[0], float(row[1]), int(row[2])
        new_odds = 0.99 if did_say else 0.01
        new_ev   = hl - new_odds
        with db._connect() as conn:
            conn.execute(
                "UPDATE training_data SET kalshi_odds = ?, ev_score = ? WHERE id = ?",
                (new_odds, float(new_ev), rid),
            )
        n_done += 1

    print(f"Restored {n_done} rows.")


if __name__ == "__main__":
    restore()
