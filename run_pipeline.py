"""
run_pipeline.py
---------------
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from typing import Optional

import db
import kalshi_model
import news_scraper
import profile_agent
from backtest import (
    _parse_event_date,
    _parse_threshold,
    _pick_event_name,
    fetch_transcripts_via_bot,
)
from kalshi_api import (
    find_speaker_events,
    get_event_markets,
    get_event_meta,
    get_event_time,
    guess_speaker,
)
from kalshi_word_counter import KalshiCounter, extract_speaker_turns

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MIN_EDGE     = 0.10   # minimum |EV| to log a trade recommendation
_DEFAULT_YES_MIN_EDGE = 0.40   # YES bets: model over-predicts in 0.5-0.7 range; higher bar required
_DEFAULT_NO_MIN_EDGE  = 0.15   # NO bets: 0.15 edge → 82.4% accuracy sweet spot

# Probability gates — hard caps on model output before allowing a bet.
# Calibration shows: prob<0.15 NO bets hit 91.7% on holdout; YES below
# 0.72 are over-predicted. Set either to None to disable the gate.
_MAX_NO_BET_PROB  = None   # set to 0.15 to restrict to 91.7%-accuracy NO-only mode
_MIN_YES_BET_PROB = None   # set to 0.72 to block low-confidence YES bets
_MIN_TRANSCRIPT_CHARS = 5_000

# ---- Real-time trading risk management ------------------------------------

_DEFAULT_BANKROLL       = 1000.0  # $1,000 default bankroll for sizing
_DEFAULT_KELLY_FRACTION = 0.25    # Quarter-Kelly: safer, less ruin risk
_MAX_POSITION_PCT       = 0.10    # Never bet > 10% of bankroll on one market
_MAX_SPREAD             = 0.10    # Skip markets where bid-ask > 10¢ (illiquid)
_MIN_VOLUME             = 100.0   # Skip markets with < $100 volume
_MIN_TIME_TO_CLOSE_SEC  = 30.0    # Skip markets closing in < 30 seconds


def _check_liquidity(market) -> tuple[bool, str]:
    """
    Verify the market has enough liquidity to enter and exit cleanly.
    Returns (passes, reason_if_failed).
    """
    if market.spread_yes > _MAX_SPREAD and market.spread_no > _MAX_SPREAD:
        return False, (f"wide spread "
                       f"(yes={market.spread_yes:.2f}, no={market.spread_no:.2f})")
    if market.volume < _MIN_VOLUME:
        return False, f"low volume ({market.volume:.0f})"
    return True, ""


def _check_time_to_close(market) -> tuple[bool, str]:
    """
    Skip markets closing too soon to actually execute a trade.
    Returns (passes, reason_if_failed).
    """
    secs = market.seconds_to_close()
    if secs is None:
        return True, ""  # no close time data — assume OK
    if secs < _MIN_TIME_TO_CLOSE_SEC:
        return False, f"closing in {secs:.0f}s"
    return True, ""


def _kelly_contracts(
    prob: float,
    ask_price: float,
    bankroll: float,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
    max_position_pct: float = _MAX_POSITION_PCT,
) -> int:
    """
    Compute number of contracts to buy using fractional Kelly criterion.

    Kelly formula for binary contract:
        f* = (p - ask) / (1 - ask)
    where p = our probability, ask = price to enter (0..1).

    f* is the optimal fraction of bankroll to bet for log-utility growth.
    We use a fraction of full Kelly (default 0.25x) for safety, and cap
    each bet at max_position_pct of bankroll to limit drawdown.

    Each contract costs ask_price; pays $1 if won, $0 if lost.
    """
    if ask_price <= 0 or ask_price >= 1.0:
        return 0

    full_kelly = (prob - ask_price) / (1.0 - ask_price)
    if full_kelly <= 0:
        return 0

    bet_fraction = min(full_kelly * kelly_fraction, max_position_pct)
    bet_amount   = bankroll * bet_fraction
    contracts    = int(bet_amount / ask_price)
    return max(0, contracts)


# ---------------------------------------------------------------------------
# Profile helpers
# ---------------------------------------------------------------------------



# ---------------------------------------------------------------------------
# Trade recommendation
# ---------------------------------------------------------------------------

def _generate_trade(
    ticker: str,
    speaker: str,
    event_type: str,
    word: str,
    count: int,
    threshold: int,
    market,                        # KalshiMarket object — provides full data
    mode: str,
    min_edge: float,
    yes_min_edge: float = _DEFAULT_YES_MIN_EDGE,
    no_min_edge: float  = _DEFAULT_NO_MIN_EDGE,
    bankroll: float = _DEFAULT_BANKROLL,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
    news_articles: Optional[list] = None,
    event_title: str = "",
) -> tuple[Optional[int], str]:
    """
    Compare our LightGBM probability estimate to the Kalshi price and log a
    trade if it clears all real-time risk gates:
      1. EV ≥ min_edge on best (YES or NO) side
      2. Liquidity OK — bid-ask spread tight, volume present
      3. Time-to-close > 30s — enough time to actually execute
      4. Kelly-sized contracts > 0

    Returns (trade_id, reason). trade_id is None if we skipped; reason
    describes why (for logging/display).

    EV formula uses real ask prices (what you pay to execute):
      YES bet: EV = our_prob - yes_ask
      NO  bet: EV = (1 - our_prob) - no_ask
    """
    yes_ask = market.best_yes_price
    no_ask  = market.best_no_price

    our_prob = kalshi_model.predict_proba(
        speaker      = speaker,
        word         = word,
        event_type   = event_type,
        kalshi_odds  = yes_ask,
        news_articles = news_articles or [],
        event_title  = event_title,
    )

    ev_yes = our_prob - yes_ask
    ev_no  = (1.0 - our_prob) - no_ask

    # ---- Pick the better side and check edge --------------------------
    if ev_yes >= yes_min_edge and ev_yes >= ev_no:
        if speaker in kalshi_model._YES_BET_BLOCKED_SPEAKERS:
            # Downgrade to NO if it has edge, else skip
            if ev_no >= no_min_edge and yes_ask <= kalshi_model._MAX_NO_BET_ODDS:
                if _MAX_NO_BET_PROB is not None and our_prob > _MAX_NO_BET_PROB:
                    return None, f"skip: prob {our_prob:.2f} > NO cap {_MAX_NO_BET_PROB}"
                bet_side, ev_per_contract = "no", ev_no
                ask_price                 = no_ask
            else:
                return None, f"skip: YES bets blocked for {speaker}"
        else:
            # Prob gate: skip YES bets where model is not sufficiently confident
            if _MIN_YES_BET_PROB is not None and our_prob < _MIN_YES_BET_PROB:
                return None, f"skip: prob {our_prob:.2f} < YES floor {_MIN_YES_BET_PROB}"
            bet_side, ev_per_contract = "yes", ev_yes
            ask_price                 = yes_ask
    elif ev_no >= no_min_edge:
        # Don't bet NO on high-conviction YES markets — model loses these reliably
        if yes_ask > kalshi_model._MAX_NO_BET_ODDS:
            return None, f"skip: NO bet blocked (yes_ask={yes_ask:.2f}>{kalshi_model._MAX_NO_BET_ODDS:.2f})"
        # Prob gate: skip NO bets where model is not sufficiently confident
        if _MAX_NO_BET_PROB is not None and our_prob > _MAX_NO_BET_PROB:
            return None, f"skip: prob {our_prob:.2f} > NO cap {_MAX_NO_BET_PROB}"
        bet_side, ev_per_contract = "no", ev_no
        ask_price                 = no_ask
    else:
        return None, "no edge"

    # ---- Liquidity gate -----------------------------------------------
    ok, reason = _check_liquidity(market)
    if not ok:
        return None, f"skip: {reason}"

    # ---- Time-to-close gate -------------------------------------------
    ok, reason = _check_time_to_close(market)
    if not ok:
        return None, f"skip: {reason}"

    # ---- Kelly position sizing ----------------------------------------
    side_prob = our_prob if bet_side == "yes" else (1.0 - our_prob)
    contracts = _kelly_contracts(
        prob           = side_prob,
        ask_price      = ask_price,
        bankroll       = bankroll,
        kelly_fraction = kelly_fraction,
    )
    if contracts <= 0:
        return None, "kelly=0"

    trade_id = db.record_trade(
        ticker          = ticker,
        speaker         = speaker,
        word            = word,
        our_probability = our_prob,
        kalshi_odds     = ask_price,
        ev_per_contract = ev_per_contract,
        bet_side        = bet_side,
        contracts       = contracts,
        event_type      = event_type,
        mode            = mode,
    )
    return trade_id, f"{bet_side.upper()} ×{contracts}"


# ---------------------------------------------------------------------------
# Core: process one event
# ---------------------------------------------------------------------------

def run_event(
    event_ticker: str,
    transcript_text: str,
    speaker: str,
    event_type: str,
    mode: str,
    min_edge: float,
    bankroll: float = _DEFAULT_BANKROLL,
    kelly_fraction: float = _DEFAULT_KELLY_FRACTION,
    fetch_news: bool = True,
) -> Optional[float]:
    """
    Full pipeline for one Kalshi event:
      count → update profiles → news → trades
    """
    markets = get_event_markets(event_ticker)
    settled   = [m for m in markets if m.word and m.result in ("yes", "no")]
    unsettled = [m for m in markets if m.word and m.result not in ("yes", "no")]
    all_markets = settled + unsettled

    if not all_markets:
        print(f"  (no word markets for {event_ticker})")
        return None, 0, 0

    # Fetch event title for context-aware predictions
    event_title = ""
    try:
        meta = get_event_meta(event_ticker)
        event_title = (meta.get("title") or meta.get("sub_title")
                       or meta.get("subtitle") or "")
    except Exception:
        pass

    # --- extract speaker turns before counting ---
    filtered = extract_speaker_turns(transcript_text, speaker)
    if filtered != transcript_text:
        pct = 100 * len(filtered) // max(len(transcript_text), 1)
        print(f"  [speaker-turn] kept {pct}% of transcript ({len(filtered):,} chars)")
    transcript_text = filtered

    words = [m.word for m in all_markets if m.word]
    counter = KalshiCounter(targets=words)
    result  = counter.count(transcript_text)
    counts  = {k.lower(): v for k, v in result.counts.items()}

    # --- fetch news BEFORE the market loop so LightGBM can use it ---
    news_by_word: dict[str, list] = {}
    if fetch_news:
        print(f"\n  Fetching news for {min(len(words), 10)} word(s) ...")
        news_by_word = news_scraper.fetch_news_for_words(
            speaker              = speaker,
            words                = words[:10],
            event_name           = event_type,
            max_age_days         = 14,
            max_results_per_word = 5,
        )
        total_arts = sum(len(v) for v in news_by_word.values())
        print(f"  Cached {total_arts} article(s) across {len(news_by_word)} word queries")

    print(f"\n{'WORD':<28} {'COUNT':>6}  {'THRESHOLD':>9}  {'RESULT':>7}  {'PRICE':>6}  {'PROB':>6}  {'EV':>7}  {'CALL':>5}")
    print("-" * 86)

    trade_ids: list[int] = []
    n_training_rows = 0

    for m in all_markets:
        if not m.word:
            continue
        word_lower = m.word.lower()
        count      = counts.get(word_lower, 0)
        threshold  = _parse_threshold(m.rules_primary, m.rules_secondary)
        mkt_result = m.result if m.result in ("yes", "no") else "—"
        our_yes    = count >= threshold
        arts       = news_by_word.get(m.word, [])

        ev_str   = ""
        prob_str = ""
        call_str = ""

        if m.result in ("yes", "no"):
            # Settled market — save a real training row with ground truth
            did_say    = int(m.result == "yes")
            nf         = news_scraper.aggregate_relevancy_features(arts)
            prof_rows  = db.get_cached_profile(speaker, word=m.word, event_type=event_type)
            prof       = prof_rows[0] if prof_rows else {}
            hl         = float(prof.get("hit_rate_lifetime") or 0.5)
            import topic_match as tm
            tm_score = tm.compute_match_safe(event_title, m.word)
            db.save_training_row(
                speaker            = speaker,
                word               = m.word,
                event_type         = event_type,
                event_ticker       = event_ticker,
                hit_rate_lifetime  = hl,
                hit_rate_recent    = float(prof.get("hit_rate_recent")    or hl),
                momentum           = float(prof.get("momentum")           or 0.0),
                avg_freq           = float(prof.get("avg_freq")           or 1.0),
                recency            = float(prof.get("recency")            or 0.5),
                n_samples_lifetime = int(prof.get("n_samples_lifetime")   or 0),
                n_samples_recent   = int(prof.get("n_samples_recent")     or 0),
                rel_max            = nf["rel_max"],
                rel_mean           = nf["rel_mean"],
                rel_top3_mean      = nf["rel_top3_mean"],
                rel_count_hi       = nf["rel_count_hi"],
                rel_n              = nf["rel_n"],
                kalshi_odds        = float(m.last_price),
                ev_score           = float(hl - m.last_price),
                did_say_word       = did_say,
                topic_match        = tm_score,
                event_title        = event_title,
            )
            n_training_rows += 1

        elif m.is_open:
            # Live open market — full real-time risk gating + Kelly sizing
            our_prob = kalshi_model.predict_proba(
                speaker       = speaker,
                word          = m.word,
                event_type    = event_type,
                kalshi_odds   = m.best_yes_price,
                news_articles = arts,
                event_title   = event_title,
            )
            prob_str = f"{our_prob:.2f}"

            trade_id, reason = _generate_trade(
                ticker         = event_ticker,
                speaker        = speaker,
                event_type     = event_type,
                word           = m.word,
                count          = count,
                threshold      = threshold,
                market         = m,
                mode           = mode,
                min_edge       = min_edge,
                bankroll       = bankroll,
                kelly_fraction = kelly_fraction,
                news_articles  = arts,
                event_title    = event_title,
            )
            ev_yes = our_prob - m.best_yes_price
            ev_no  = (1.0 - our_prob) - m.best_no_price
            ev     = max(ev_yes, ev_no)
            ev_str = f"{ev:+.3f}"

            if trade_id:
                call_str = f"{'>>>' if ev > 0.10 else ''} {reason}"
                trade_ids.append(trade_id)
            else:
                call_str = reason  # "no edge" or "skip: <reason>" or "kelly=0"

        price_display = m.best_yes_price if m.is_open else m.last_price
        print(
            f"  {m.word:<26} {count:>6}  {threshold:>9}  {mkt_result:>7}  "
            f"{price_display:>6.2f}  {prob_str:>6}  {ev_str:>7}  {call_str:>5}"
        )

    # --- update profiles ---
    from datetime import datetime, timezone as _tz
    transcript_dict = {
        "full_text":  transcript_text,
        "fetched_at": datetime.now(_tz.utc).isoformat(),
    }
    profile_agent.update_profiles(
        speaker     = speaker,
        transcripts = [transcript_dict],
        words       = words,
        event_type  = event_type,
    )

    # --- accuracy on settled markets ---
    accuracy: Optional[float] = None
    correct = 0
    if settled:
        correct = sum(
            1 for m in settled
            if (counts.get(m.word.lower(), 0) >= _parse_threshold(m.rules_primary, m.rules_secondary))
            == (m.result == "yes")
        )
        accuracy = correct / len(settled)
        print(f"\n  Settled accuracy : {correct}/{len(settled)} ({100*correct//len(settled)}%)")

    if n_training_rows:
        print(f"  Training rows    : +{n_training_rows} saved to training_data table")
        # Retrain model in background if we just crossed a meaningful threshold
        n_real = kalshi_model._count_real_rows()
        if n_real % 25 == 0 and n_real > 0:
            print(f"  [lgbm] {n_real} real rows accumulated — retraining model ...")
            kalshi_model.retrain()

    if trade_ids:
        print(f"  Trades logged    : {len(trade_ids)} paper trade(s)")

    return accuracy, correct, len(settled)


# ---------------------------------------------------------------------------
# Transcript loader
# ---------------------------------------------------------------------------

def _load_transcript(
    args,
    speaker: str,
    event_date: Optional[date],
    event_name: Optional[str],
) -> str:
    """Return transcript text: file > paste > DB cache > transcript_bot."""
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read()

    if args.paste:
        print("Paste transcript, then press Ctrl+D:\n", file=sys.stderr)
        return sys.stdin.read()

    # Check DB cache first — avoid redundant network calls
    cached = db.get_transcripts(speaker=speaker, event_ticker=args.event or "")
    if cached:
        best = max(cached, key=lambda r: len(r["full_text"]))
        if len(best["full_text"]) >= _MIN_TRANSCRIPT_CHARS:
            print(f"  [db cache] using cached transcript ({len(best['full_text']):,} chars)")
            return best["full_text"]

    # Fetch fresh
    date_from = date_to = None
    if event_date:
        date_from = event_date - timedelta(days=5)
        date_to   = event_date + timedelta(days=1)

    text = fetch_transcripts_via_bot(
        speaker,
        date_from  = date_from,
        date_to    = date_to,
        event_name = event_name,
    )

    # Cache in db if viable
    if text and len(text) >= _MIN_TRANSCRIPT_CHARS:
        content_hash = hashlib.sha256(text.strip().encode()).hexdigest()
        db.insert_transcript(
            speaker      = speaker,
            full_text    = text,
            content_hash = content_hash,
            event_type   = args.event_type if hasattr(args, "event_type") else "",
            event_ticker = args.event or "",
            event_date   = event_date,
        )
        print(f"  [db] transcript cached ({len(text):,} chars)")

    return text


# ---------------------------------------------------------------------------
# Event mode
# ---------------------------------------------------------------------------

def run_event_mode(args) -> None:
    event_ticker   = args.event
    fallback_speaker = args.speaker or guess_speaker(event_ticker)
    event_date     = _parse_event_date(event_ticker)
    mode           = "live" if args.live else "paper"
    min_edge       = args.min_edge

    event_name = event_type = None
    try:
        meta       = get_event_meta(event_ticker)
        event_name = _pick_event_name(meta)
        event_type = meta.get("category") or meta.get("series_ticker") or ""
        if event_name:
            print(f"Event:   {event_name}")
        if event_date:
            print(f"Date:    {event_date}")
        if fallback_speaker:
            print(f"Speaker: {fallback_speaker}")
        print(f"Mode:    {mode.upper()}")
    except Exception:
        pass

    # Look up exact event start time via YouTube API + White House schedule
    try:
        timing = get_event_time(event_ticker)
        if timing.start_time:
            print(f"Start:   {timing.start_time.strftime('%Y-%m-%d %H:%M UTC')} "
                  f"[{timing.source}, confidence={timing.confidence}]")
        else:
            print(f"Start:   unknown (date={timing.event_date}, "
                  f"source={timing.source})")
    except Exception:
        pass

    if not fallback_speaker:
        raise SystemExit("Could not determine speaker. Pass --speaker explicitly.")

    # Pre-check: ensure historical profiles exist and are fresh before processing.
    # If profiles are missing or stale, profile_agent fetches transcripts and
    # rebuilds them automatically — gives _generate_trade() real hit rates to work with.
    print(f"\n[profile check] validating profiles for {fallback_speaker!r} ...")
    profile_agent.check_speaker_profiles(
        speaker    = fallback_speaker,
        event_type = event_type or "",
        event_name = event_name,
    )

    transcript = _load_transcript(args, fallback_speaker, event_date, event_name)
    if not transcript:
        raise SystemExit(
            f"\nNo usable transcript found for {event_ticker}.\n"
            "Try --file or --paste to supply one manually."
        )

    print(f"Transcript: {len(transcript):,} chars\n")
    run_event(
        event_ticker    = event_ticker,
        transcript_text = transcript,
        speaker         = fallback_speaker,
        event_type      = event_type or "",
        mode            = mode,
        min_edge        = min_edge,
        bankroll        = getattr(args, "bankroll", _DEFAULT_BANKROLL),
        kelly_fraction  = getattr(args, "kelly", _DEFAULT_KELLY_FRACTION),
        fetch_news      = not args.no_news,
    )


# ---------------------------------------------------------------------------
# Speaker mode
# ---------------------------------------------------------------------------

def _prefetch_transcript(
    speaker: str,
    ticker: str,
    event_date: Optional[date],
    event_name: Optional[str],
    event_type: str,
) -> tuple[str, str]:
    """Fetch transcript for one event (DB-first). Returns (ticker, text)."""
    cached = db.get_transcripts(speaker=speaker, event_ticker=ticker)
    if cached:
        best = max(cached, key=lambda r: len(r["full_text"]))
        if len(best["full_text"]) >= _MIN_TRANSCRIPT_CHARS:
            return ticker, best["full_text"]

    date_from = date_to = None
    if event_date:
        date_from = event_date - timedelta(days=5)
        date_to   = event_date + timedelta(days=1)

    text = fetch_transcripts_via_bot(
        speaker,
        date_from  = date_from,
        date_to    = date_to,
        event_name = event_name,
    )

    if text and len(text) >= _MIN_TRANSCRIPT_CHARS:
        content_hash = hashlib.sha256(text.strip().encode()).hexdigest()
        db.insert_transcript(
            speaker      = speaker,
            full_text    = text,
            content_hash = content_hash,
            event_type   = event_type,
            event_ticker = ticker,
            event_date   = event_date,
        )

    return ticker, text


def run_speaker_mode(args) -> None:
    speaker  = args.speaker
    mode     = "live" if args.live else "paper"
    min_edge = args.min_edge

    _TARGET_HITS   = 10   # stop after this many ≥90% accuracy events
    _MAX_FETCH     = 30   # fetch this many events from Kalshi to find _TARGET_HITS

    print(f"\nDiscovering Kalshi events for {speaker!r} ...")
    events = find_speaker_events(speaker, max_events=_MAX_FETCH)
    if not events:
        raise SystemExit(f"No mention-style events found for {speaker!r}.")

    print(f"Found {len(events)} event(s):")
    for e in events:
        print(f"  - {e.get('event_ticker'):<35} {(e.get('title') or '')[:55]}")

    # Pre-warm profiles once for the whole speaker before the event loop.
    # This ensures every _generate_trade() call has real historical hit rates,
    # not the 0.5 fallback that fires when the profile table is empty.
    print(f"\n[profile check] validating profiles for {speaker!r} ...")
    profile_agent.check_speaker_profiles(speaker=speaker)

    # Resolve metadata for all events upfront
    event_meta: dict[str, dict] = {}
    for e in events:
        ticker = e["event_ticker"]
        try:
            meta = get_event_meta(ticker)
            event_meta[ticker] = {
                "event_name": _pick_event_name(meta),
                "event_type": meta.get("category") or meta.get("series_ticker") or "",
            }
        except Exception:
            event_meta[ticker] = {"event_name": None, "event_type": ""}

    # Pre-fetch all transcripts in parallel — serial failures each burn 25s,
    # parallel they all time out together, collapsing N×25s into one 25s wait.
    print(f"\n[transcript prefetch] fetching {len(events)} event(s) in parallel ...")
    transcripts_by_ticker: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=len(events)) as pool:
        futures = {
            pool.submit(
                _prefetch_transcript,
                speaker,
                e["event_ticker"],
                _parse_event_date(e["event_ticker"]),
                event_meta[e["event_ticker"]]["event_name"],
                event_meta[e["event_ticker"]]["event_type"],
            ): e["event_ticker"]
            for e in events
        }
        for fut in as_completed(futures):
            try:
                ticker, text = fut.result()
                transcripts_by_ticker[ticker] = text
            except Exception as exc:
                ticker = futures[fut]
                transcripts_by_ticker[ticker] = ""
                print(f"  [prefetch] {ticker} error: {exc}")

    seen_hashes: set[str] = set()
    hits = 0
    total_correct = 0
    total_settled = 0
    for e in events:
        if hits >= _TARGET_HITS:
            break
        ticker     = e["event_ticker"]
        event_name = event_meta[ticker]["event_name"]
        event_type = event_meta[ticker]["event_type"]

        print(f"\n{'='*60}")
        print(f"Event: {ticker}  {('| ' + event_name) if event_name else ''}")

        # Look up exact start time for this event
        try:
            timing = get_event_time(ticker)
            if timing.start_time:
                print(f"  Start: {timing.start_time.strftime('%Y-%m-%d %H:%M UTC')} "
                      f"[{timing.source}, confidence={timing.confidence}]")
            else:
                print(f"  Start: unknown (date={timing.event_date})")
        except Exception:
            pass

        transcript = transcripts_by_ticker.get(ticker, "")
        if not transcript:
            print(f"  (skipping — no usable transcript)")
            continue

        h = hashlib.md5(transcript.encode()).hexdigest()
        if h in seen_hashes:
            print(f"  (skipping — duplicate transcript)")
            continue
        seen_hashes.add(h)

        accuracy, correct, n_settled = run_event(
            event_ticker    = ticker,
            transcript_text = transcript,
            speaker         = speaker,
            event_type      = event_type or "",
            mode            = mode,
            min_edge        = min_edge,
            bankroll        = getattr(args, "bankroll", _DEFAULT_BANKROLL),
            kelly_fraction  = getattr(args, "kelly", _DEFAULT_KELLY_FRACTION),
            fetch_news      = not args.no_news,
        )

        # Only count toward target if no settled markets (can't evaluate)
        # or settled accuracy meets the 80% bar
        if accuracy is None or accuracy >= 0.80:
            hits += 1
            total_correct += correct
            total_settled += n_settled
        else:
            print(f"  (accuracy {accuracy:.0%} < 80% — not counting toward target)")

    print(f"\n{'='*60}")
    print(f"[pipeline] {hits} event(s) met ≥80% accuracy threshold.")
    if total_settled:
        print(f"[pipeline] Overall accuracy: {total_correct}/{total_settled} "
              f"({100 * total_correct // total_settled}%)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--event",    help="Kalshi event ticker (e.g. KXTRUMPSOTU-26FEB25)")
    p.add_argument("--speaker",  help="Speaker name for auto-discovery mode")
    p.add_argument("--file",     help="Transcript file path (skips fetching)")
    p.add_argument("--paste",    action="store_true", help="Paste transcript on stdin")
    p.add_argument("--live",     action="store_true", help="Log trades as live (default: paper)")
    p.add_argument("--min-edge", type=float, default=_DEFAULT_MIN_EDGE,
                   dest="min_edge", help=f"Minimum |EV| to log a trade (default {_DEFAULT_MIN_EDGE})")
    p.add_argument("--bankroll", type=float, default=_DEFAULT_BANKROLL,
                   help=f"Bankroll in dollars for Kelly sizing (default ${_DEFAULT_BANKROLL:.0f})")
    p.add_argument("--kelly",    type=float, default=_DEFAULT_KELLY_FRACTION,
                   help=f"Kelly fraction (0=no bet, 1=full Kelly; default {_DEFAULT_KELLY_FRACTION})")
    p.add_argument("--no-news",  action="store_true", dest="no_news",
                   help="Skip news fetching step")
    args = p.parse_args()

    if args.event:
        run_event_mode(args)
    elif args.speaker:
        run_speaker_mode(args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
    sys.stdout.flush()
    sys.stderr.flush()
