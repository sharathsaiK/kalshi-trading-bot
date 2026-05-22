"""
backtest_trades.py
------------------
Simulate what the bot WOULD have done on a past Kalshi event.

For each settled word market:
  1. Use the historical entry price (previous_yes_ask / previous_no_ask)
     — this is what the market was at before converging to 0.99 / 0.01
  2. Run the LightGBM model to get our predicted probability
  3. Compute Kelly-sized contracts (with fractional Kelly + position cap)
  4. Compare against the actual settlement
  5. Compute hypothetical P&L

Usage:
  python3 backtest_trades.py KXTRUMPMENTION-26MAY06
  python3 backtest_trades.py KXTRUMPMENTION-26MAY06 --bankroll 5000 --kelly 0.5
"""

from __future__ import annotations

import argparse
import statistics
import sys
import requests

import kalshi_api
import kalshi_model
from run_pipeline import (
    _check_liquidity,
    _kelly_contracts,
    _DEFAULT_BANKROLL,
    _DEFAULT_KELLY_FRACTION,
)


def _get_historical_entry_price(market_ticker: str) -> tuple[float, float, int]:
    """
    Fetch historical trades for a market and return a realistic entry price.

    Returns (yes_entry_price, no_entry_price, n_trades).
    Uses the FIRST 25% of trades by time — the early-window price before
    the market converged toward settlement. This is what a bot entering
    early would actually have paid.
    """
    try:
        url = "https://api.elections.kalshi.com/trade-api/v2/markets/trades"
        all_trades: list[dict] = []
        cursor = None
        # Paginate through up to 5 pages (5000 trades max)
        for _ in range(5):
            params = {"ticker": market_ticker, "limit": 1000}
            if cursor:
                params["cursor"] = cursor
            r = requests.get(url, params=params, timeout=15)
            if r.status_code != 200:
                break
            data = r.json()
            trades = data.get("trades", [])
            if not trades:
                break
            all_trades.extend(trades)
            cursor = data.get("cursor")
            if not cursor:
                break

        if not all_trades:
            return 0.0, 0.0, 0

        # Sort oldest → newest by created_time
        all_trades.sort(key=lambda t: t.get("created_time", ""))
        # Take the first 25% — early-window prices
        early = all_trades[: max(1, len(all_trades) // 4)]
        yes_prices = [float(t.get("yes_price_dollars") or 0) for t in early]
        no_prices  = [float(t.get("no_price_dollars")  or 0) for t in early]
        yes_prices = [p for p in yes_prices if 0.01 < p < 0.99]
        no_prices  = [p for p in no_prices  if 0.01 < p < 0.99]
        if not yes_prices:
            return 0.0, 0.0, len(all_trades)
        return (
            statistics.median(yes_prices),
            statistics.median(no_prices) if no_prices else (1.0 - statistics.median(yes_prices)),
            len(all_trades),
        )
    except Exception:
        return 0.0, 0.0, 0


def backtest_event(
    event_ticker: str,
    speaker: str = "Donald Trump",
    event_type: str = "speech",
    bankroll: float = _DEFAULT_BANKROLL,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
    min_edge: float = 0.10,
    yes_min_edge: float = 0.18,
) -> None:
    """
    Run a full hypothetical backtest on a single Kalshi event.
    Prints per-market trade decisions and overall P&L summary.
    """
    markets = kalshi_api.get_event_markets(event_ticker, historical=False)
    settled = [m for m in markets if m.word and m.result in ("yes", "no")]

    if not settled:
        print(f"No settled markets found for {event_ticker}")
        print("(Backtest only works on past/finalized events.)")
        return

    # Fetch event title for context-aware predictions
    try:
        meta = kalshi_api.get_event_meta(event_ticker)
        event_title = (meta.get("title") or meta.get("sub_title")
                       or meta.get("subtitle") or "")
    except Exception:
        event_title = ""

    print(f"\n{'=' * 90}")
    print(f"BACKTEST: {event_ticker}  —  {len(settled)} settled markets")
    print(f"Title: {event_title}")
    print(f"Bankroll: ${bankroll:.0f}  Kelly: {kelly_fraction}  Min edge: {min_edge}")
    print(f"{'=' * 90}")

    print(f"\n{'Word':<22} {'EntryPx':<8} {'Model':<7} {'Side':<5} "
          f"{'Contr':<6} {'Cost':<8} {'Result':<6} {'P&L':<10} Outcome")
    print("-" * 90)

    total_cost     = 0.0
    total_pnl      = 0.0
    n_trades       = 0
    n_wins         = 0
    n_losses       = 0
    n_skipped      = 0

    for m in settled:
        word = m.word
        # Get a realistic entry price from the historical trades feed.
        # This is the median of the first 25% of trades — what an early
        # entrant would actually have paid before the market converged.
        yes_ask, no_ask, n_hist = _get_historical_entry_price(m.ticker)

        # Fallback to previous_yes_ask if no trade history available
        if yes_ask <= 0:
            yes_ask = m.previous_yes_ask if 0.03 < m.previous_yes_ask < 0.97 else 0
            no_ask  = (1.0 - m.previous_yes_bid) if m.previous_yes_bid > 0 else 0

        # Skip if we can't find a realistic entry price.
        # Boundaries match model's settlement mask: [0.04, 0.96] is "live market".
        if yes_ask <= 0.04 or yes_ask >= 0.96:
            print(f"{word:<22} {yes_ask:<8.2f} {'-':<7} {'-':<5} "
                  f"{'-':<6} {'-':<8} {m.result:<6} {'-':<10} (no historical price)")
            n_skipped += 1
            continue

        # Run model with event context for topic-match awareness
        our_prob = kalshi_model.predict_proba(
            speaker     = speaker,
            word        = word,
            event_type  = event_type,
            kalshi_odds = yes_ask,
            event_title = event_title,
        )

        # Pick side
        ev_yes = our_prob - yes_ask
        ev_no  = (1.0 - our_prob) - no_ask

        if ev_yes >= yes_min_edge and ev_yes >= ev_no:
            side, ask_price, side_prob = "YES", yes_ask, our_prob
        elif ev_no >= min_edge:
            if yes_ask > kalshi_model._MAX_NO_BET_ODDS:
                print(f"{word:<22} {yes_ask:<8.2f} {our_prob:<7.3f} {'NO':<5} "
                      f"{'-':<6} {'-':<8} {m.result:<6} {'-':<10} NO blocked (odds>{kalshi_model._MAX_NO_BET_ODDS:.0%})")
                n_skipped += 1
                continue
            side, ask_price, side_prob = "NO",  no_ask,  (1.0 - our_prob)
        else:
            print(f"{word:<22} {yes_ask:<8.2f} {our_prob:<7.3f} {'-':<5} "
                  f"{'-':<6} {'-':<8} {m.result:<6} {'-':<10} no edge")
            n_skipped += 1
            continue

        # Kelly sizing
        contracts = _kelly_contracts(
            prob           = side_prob,
            ask_price      = ask_price,
            bankroll       = bankroll,
            kelly_fraction = kelly_fraction,
        )
        if contracts <= 0:
            print(f"{word:<22} {yes_ask:<8.2f} {our_prob:<7.3f} {side:<5} "
                  f"{'0':<6} {'-':<8} {m.result:<6} {'-':<10} kelly=0")
            n_skipped += 1
            continue

        cost = contracts * ask_price
        total_cost += cost
        n_trades   += 1

        # Did we win?
        won = (side == "YES" and m.result == "yes") or \
              (side == "NO"  and m.result == "no")

        if won:
            payout = contracts * 1.0   # each winning contract pays $1
            pnl    = payout - cost
            n_wins += 1
            outcome = "✓ WIN"
        else:
            pnl    = -cost
            n_losses += 1
            outcome = "✗ LOSS"

        total_pnl += pnl

        print(f"{word:<22} {yes_ask:<8.2f} {our_prob:<7.3f} {side:<5} "
              f"{contracts:<6} ${cost:<7.2f} {m.result:<6} ${pnl:<+9.2f} {outcome}")

    print("-" * 90)
    print(f"\nSUMMARY")
    print(f"  Markets analyzed   : {len(settled)}")
    print(f"  Trades executed    : {n_trades}")
    print(f"  Skipped            : {n_skipped}  (no edge, kelly=0, or extreme price)")
    if n_trades > 0:
        win_rate = 100 * n_wins / n_trades
        roi      = 100 * total_pnl / total_cost if total_cost else 0.0
        print(f"  Wins / Losses      : {n_wins} / {n_losses}  "
              f"({win_rate:.0f}% win rate)")
        print(f"  Total invested     : ${total_cost:.2f}")
        print(f"  Total P&L          : ${total_pnl:+.2f}  "
              f"({roi:+.1f}% ROI)")
        print(f"  Bankroll: ${bankroll:.0f} → ${bankroll + total_pnl:.0f}")
    else:
        print(f"  No trades — model found no edge on this event.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("ticker", help="Past event ticker (e.g. KXTRUMPMENTION-26MAY06)")
    p.add_argument("--speaker",   default="Donald Trump")
    p.add_argument("--event-type", default="speech")
    p.add_argument("--bankroll",  type=float, default=_DEFAULT_BANKROLL)
    p.add_argument("--kelly",     type=float, default=_DEFAULT_KELLY_FRACTION)
    p.add_argument("--min-edge",  type=float, default=0.10)
    args = p.parse_args()

    backtest_event(
        event_ticker   = args.ticker,
        speaker        = args.speaker,
        event_type     = args.event_type,
        bankroll       = args.bankroll,
        kelly_fraction = args.kelly,
        min_edge       = args.min_edge,
    )


if __name__ == "__main__":
    main()
    sys.stdout.flush()
