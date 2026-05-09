"""
fix_training_prices.py
----------------------
Replace settlement-value kalshi_odds (0.99/0.01) with realistic
mid-range entry prices generated from speaker hit rate + noise.

Why: training rows were saved with kalshi_odds = m.last_price = settlement
value (0.99 if YES, 0.01 if NO). The model learned "predict the odds back"
which doesn't generalize to live mid-range markets.

Fix: kalshi_odds = hit_rate_lifetime + N(0, 0.18), with 15% randomly
mispriced. This matches the synthetic data generator and gives the model
realistic odds to learn from.
"""

import db
import numpy as np


def fix_prices():
    rng = np.random.default_rng(42)

    with db._connect() as conn:
        rows = conn.execute("""
            SELECT id, hit_rate_lifetime, did_say_word
            FROM training_data
            WHERE kalshi_odds >= 0.95 OR kalshi_odds <= 0.05
        """).fetchall()

    print(f"Fixing {len(rows)} rows with extreme kalshi_odds ...")

    n_fixed = 0
    for row in rows:
        rid, hl, did_say = row[0], float(row[1]), int(row[2])

        # 15% of rows: deliberately mispriced (matches synthetic data style).
        # 85%: market is roughly correct, odds correlate with hit rate.
        if rng.random() < 0.15:
            new_odds = float(np.clip(rng.uniform(0.05, 0.95), 0.04, 0.96))
        else:
            new_odds = float(np.clip(hl + rng.normal(0, 0.18), 0.04, 0.96))

        new_ev = hl - new_odds
        with db._connect() as conn:
            conn.execute(
                "UPDATE training_data SET kalshi_odds = ?, ev_score = ? WHERE id = ?",
                (new_odds, float(new_ev), rid),
            )
        n_fixed += 1

    print(f"Fixed {n_fixed} rows.")

    # Verify the new distribution
    with db._connect() as conn:
        rows = conn.execute("SELECT kalshi_odds FROM training_data").fetchall()
    extreme = sum(1 for r in rows if r[0] >= 0.95 or r[0] <= 0.05)
    mid = sum(1 for r in rows if 0.05 < r[0] < 0.95)
    print(f"\nNew distribution:")
    print(f"  Extreme (>=0.95 or <=0.05): {extreme} ({100*extreme//len(rows)}%)")
    print(f"  Mid-range (0.05–0.95): {mid} ({100*mid//len(rows)}%)")


if __name__ == "__main__":
    fix_prices()
