"""
backtest.py
-----------
Score the Kalshi word counter + transcript bot against real Kalshi
mention markets, scoped to *individual events* (one speech / briefing /
SOTU per evaluation).

Earlier versions of this script aggregated weekly markets, which mixed
many appearances together and dragged accuracy down. The unit of
evaluation is now a single Kalshi event, with the transcript pulled from
a tight ±1-day window around the event date so we score the speech the
market is actually watching.

Modes:

  A) Event mode — score a specific event:
        python backtest.py --event KXVANCEINGRAHAM-25MAR14
        python backtest.py --event KXFOMCMENTION-26MAY07 --speaker "Jerome Powell"
        python backtest.py --event KXTRUMPSOTU-26FEB25 --file sotu.txt
        python backtest.py --event KXTRUMPSOTU-26FEB25 --paste

  B) Speaker mode — auto-discover past per-event markets and score each:
        python backtest.py --speaker "Donald Trump"
        python backtest.py --speaker "Jerome Powell"
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from datetime import date, timedelta

from kalshi_api import find_speaker_events, get_event_markets, get_event_meta, guess_speaker
from kalshi_word_counter import KalshiCounter, extract_speaker_turns


# ---------------------------------------------------------------------------
# Threshold parser
# ---------------------------------------------------------------------------

def _parse_threshold(rules_primary: str, rules_secondary: str = "") -> int:
    blob = (rules_primary + " " + rules_secondary).lower()
    # "at least N time(s)" / "mentioned at least N time(s)"
    m = re.search(r'(?:at least|mentioned at least|said at least|used at least)\s+(\d+)\s+time', blob)
    if m:
        return int(m.group(1))
    # "at least N mention(s)" / "at least N occurrence(s)"
    m = re.search(r'at least\s+(\d+)\s+(?:mention|occurrence|instance|use)', blob)
    if m:
        return int(m.group(1))
    # "N or more time(s)" / "N or more mention(s)"
    m = re.search(r'(\d+)\s+or\s+more\s+(?:time|mention|occurrence|instance|use)', blob)
    if m:
        return int(m.group(1))
    # "more than N time(s)"
    m = re.search(r'more than\s+(\d+)\s+time', blob)
    if m:
        return int(m.group(1)) + 1
    # "no fewer than N" / "minimum of N" / "a minimum of N"
    m = re.search(r'(?:no fewer than|minimum of|a minimum of)\s+(\d+)', blob)
    if m:
        return int(m.group(1))
    # "≥ N" / ">= N" / "> N" (bare numeric comparator)
    m = re.search(r'[≥>]=?\s*(\d+)', blob)
    if m:
        n = int(m.group(1))
        return n if '>=' in blob or '≥' in blob else n + 1
    # "N+ times" (e.g. "5+ times")
    m = re.search(r'(\d+)\+\s*times', blob)
    if m:
        return int(m.group(1))
    # Kalshi single-mention format: "If X says Y ... resolves to Yes/No" — no
    # explicit count means exactly 1 mention is the threshold.  Return silently.
    if re.search(r'\bif\b.+\bsays\b.+\bresolves to\b', blob):
        return 1
    # "qualifying event does not occur / does not qualify" — binary, threshold 1.
    if re.search(r'does not (?:occur|qualify|take place)', blob):
        return 1
    import sys as _sys
    print(f"  [threshold] no pattern matched rules — defaulting to 1. "
          f"Rules: {(rules_primary or '')[:120]!r}", file=_sys.stderr)
    return 1


# ---------------------------------------------------------------------------
# Date parsing from event ticker
# ---------------------------------------------------------------------------

_MONTH_MAP = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}

def _parse_event_date(event_ticker: str) -> date | None:
    # Optional trailing letter (e.g. "26MAR26B" for same-day disambiguation)
    m = re.search(r'-(\d{2})([A-Z]{3})(\d{2})[A-Z]?(?:-|$)', event_ticker)
    if not m:
        return None
    yy, mon, dd = m.group(1), m.group(2), m.group(3)
    month = _MONTH_MAP.get(mon)
    if not month:
        return None
    return date(2000 + int(yy), month, int(dd))


# Subtitles like "On Mar 6, 2026" or "Before Jan 12, 2026" are just dates and
# carry no event-specific info. In those cases we want the descriptive title
# instead so the transcript bot can search for the actual speech.
_DATE_ONLY_RE = re.compile(
    r'^\s*(on|before|after|by)?\s*'
    r'(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*'
    r'\s+\d{1,2},?\s+\d{4}\s*$',
    re.IGNORECASE,
)


def _pick_event_name(meta: dict) -> str | None:
    """Pick the most descriptive event name from Kalshi metadata."""
    subtitle = (meta.get("sub_title") or meta.get("subtitle") or "").strip()
    title    = (meta.get("title") or "").strip()
    # If subtitle is just a date, prefer the title (which describes the event).
    if subtitle and not _DATE_ONLY_RE.match(subtitle):
        return subtitle
    return title or subtitle or None


# ---------------------------------------------------------------------------
# Transcript loading
# ---------------------------------------------------------------------------

def fetch_transcripts_via_bot(
    speaker: str,
    n: int = 3,
    date_from: date | None = None,
    date_to: date | None = None,
    event_name: str | None = None,
    timeout: float = 25.0,
) -> str:
    """Fetch the transcript for a single Kalshi event.

    Event-scoped: we want exactly the speech the market is watching, not a
    week's worth of appearances. Caller passes a tight date window around the
    event date and we return only the best-matching transcript.

    timeout=25s per event keeps 10-event speaker runs under 5 minutes total.
    """
    from transcript_bot import fetch_transcripts, TranscriptQuery
    date_info = f" ({date_from} → {date_to})" if date_from and date_to else ""
    event_info = f" [{event_name}]" if event_name else ""
    print(f"  → fetching transcript for {speaker!r}{event_info}{date_info} ...", file=sys.stderr)
    # strict_date=True when a date window is given: undated WH/factbase pages
    # leak through the bot's "no dated results — using N undated" fallback and
    # produce 0.27-confidence wrong-event matches that the backtest then has to
    # reject downstream. Skipping at the bot level is cleaner.
    results = fetch_transcripts(TranscriptQuery(
        speaker_name=speaker,
        event_name=event_name,
        date_from=date_from,
        date_to=date_to,
        max_results=n,
        strict_date=bool(date_from and date_to),
        timeout=timeout,
    ))
    if not results:
        return ""
    # Filter by length first, then pick best confidence from viable candidates.
    # Without this, a 1,977-char WH stub (confidence=0.95) beats an 80k
    # singju_post transcript (confidence=0.60) and the event gets skipped.
    MIN_CHARS = 5_000
    MIN_CONF = 0.35
    viable = [r for r in results if len(r.full_text) >= MIN_CHARS]
    if not viable:
        longest = max(results, key=lambda r: len(r.full_text))
        print(f"  → best match too short ({len(longest.full_text):,} chars < "
              f"{MIN_CHARS:,}); skipping event", file=sys.stderr)
        return ""
    best = max(viable, key=lambda r: r.match_confidence)
    # Reject transcripts whose relevance-adjusted confidence is too low —
    # this catches "right speaker, wrong event" matches where the bot
    # returned a generic Trump speech for, say, the Memphis Roundtable.
    if best.match_confidence < MIN_CONF:
        print(f"  → best match confidence too low "
              f"({best.match_confidence:.2f} < {MIN_CONF}); skipping event",
              file=sys.stderr)
        return ""
    print(f"  → got {len(results)} transcript(s); using best match "
          f"(confidence={best.match_confidence:.2f}, {len(best.full_text):,} chars)",
          file=sys.stderr)
    return best.full_text


def load_transcript(
    args,
    fallback_speaker: str | None,
    event_date: date | None = None,
    event_name: str | None = None,
) -> str:
    if args.file:
        with open(args.file, encoding="utf-8", errors="replace") as f:
            return f.read()
    if args.paste:
        print("Paste transcript, then press Ctrl+D when done:\n", file=sys.stderr)
        return sys.stdin.read()
    speaker = args.speaker or fallback_speaker
    if not speaker:
        raise SystemExit("Need a speaker — pass --speaker, --file, or --paste.")

    date_from = date_to = None
    if event_date:
        # Look back 5 days: Kalshi settlement dates often lag the actual speech
        # by 1-4 days (e.g. SOTU ticker is Feb 28 but speech was Feb 25).
        date_from = event_date - timedelta(days=5)
        date_to = event_date + timedelta(days=1)

    text = fetch_transcripts_via_bot(
        speaker,
        date_from=date_from,
        date_to=date_to,
        event_name=event_name,
    )
    return text  # may be "" — caller decides whether to skip or abort


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score_event(event_ticker: str, transcript: str, show_bets: bool = True,
                speaker: str | None = None) -> tuple[int, int]:
    """Run counter against one event. Returns (correct, total)."""
    markets = get_event_markets(event_ticker)
    settled  = [m for m in markets if m.word and m.result in ("yes", "no")]
    unsettled = [m for m in markets if m.word and m.result not in ("yes", "no")]
    all_markets = settled + unsettled

    # If "Event does not qualify" settled YES, the event was cancelled/invalid.
    # Scoring it would just measure how wrong the wrong transcript is.
    dnq = next(
        (m for m in settled
         if "does not qualify" in (m.word or "").lower() and m.result == "yes"),
        None,
    )
    if dnq:
        print(f"  (skipping {event_ticker} — event did not qualify/was cancelled)")
        return (0, 0)

    words = [m.word for m in all_markets if m.word]
    if not words:
        print(f"  (no word markets for {event_ticker})")
        return (0, 0)

    # Strip other speakers' lines before counting — reporters asking about
    # "nuclear" or "Iran" shouldn't inflate our count of Trump's word usage.
    if speaker:
        filtered = extract_speaker_turns(transcript, speaker)
        if filtered != transcript:
            print(f"  [speaker-turn] extracted {len(filtered):,} / {len(transcript):,} chars "
                  f"({100*len(filtered)//len(transcript)}% of transcript kept)",
                  file=sys.stderr)
        else:
            print(f"  [speaker-turn] no turns detected — using full transcript", file=sys.stderr)
        transcript = filtered

    counter = KalshiCounter(targets=words)
    result = counter.count(transcript)
    counts = {k.lower(): v for k, v in result.counts.items()}

    rows, correct = [], 0
    bet_rows = []

    for m in all_markets:
        if not m.word:
            continue
        our = counts.get(m.word.lower(), 0)
        threshold = _parse_threshold(m.rules_primary, m.rules_secondary)
        we_yes = our >= threshold
        mkt_yes = m.result == "yes" if m.result in ("yes", "no") else None
        settled_flag = mkt_yes is not None

        if settled_flag:
            ok = we_yes == mkt_yes
            if ok:
                correct += 1
            rows.append((
                "OK " if ok else "MISS",
                m.word, our, threshold, m.result, m.last_price, m.volume
            ))
        else:
            # Unsettled — generate bet recommendation
            our_call = "YES" if we_yes else "NO"
            # Edge: if we say YES and price < 0.5, or we say NO and price > 0.5
            edge = (m.last_price if not we_yes else 1 - m.last_price)
            bet_rows.append((m.word, our, threshold, our_call, m.last_price, m.volume, edge))

    # Print settled results
    if rows:
        rows.sort(key=lambda r: (r[0] == "OK ", -r[6]))
        print(f"\n=== {event_ticker} ===")
        print(f"Accuracy: {correct}/{len(rows)} ({100*correct/len(rows):.0f}%)\n")
        print(f"{'':<5} {'WORD':<28} {'OURS':>5} {'MIN':>5} {'MKT':>5} {'PRICE':>6} {'VOL':>10}")
        print("-" * 70)
        for match, word, our, threshold, mkt_result, price, vol in rows:
            print(f"{match:<5} {word:<28} {our:>5} {threshold:>5} {mkt_result:>5} {price:>6.2f} {vol:>10,.0f}")

    # Print bet recommendations for unsettled markets
    if bet_rows and show_bets:
        bet_rows.sort(key=lambda r: -r[6])  # sort by edge
        print(f"\n  {'BET RECOMMENDATIONS (unsettled)'}")
        print(f"  {'WORD':<28} {'OURS':>5} {'MIN':>5} {'CALL':>5} {'PRICE':>6} {'EDGE':>6} {'VOL':>10}")
        print("  " + "-" * 70)
        for word, our, threshold, call, price, vol, edge in bet_rows:
            marker = ">>>" if edge > 0.3 else "  "
            print(f"  {marker} {word:<26} {our:>5} {threshold:>5} {call:>5} {price:>6.2f} {edge:>6.2f} {vol:>10,.0f}")

    return correct, len(rows)


# ---------------------------------------------------------------------------
# Modes
# ---------------------------------------------------------------------------

def run_event_mode(event_ticker: str, args) -> None:
    fallback_speaker = args.speaker or guess_speaker(event_ticker)
    event_date = _parse_event_date(event_ticker)

    # Pull Kalshi metadata to get the specific event name
    event_name = None
    try:
        meta = get_event_meta(event_ticker)
        event_name = _pick_event_name(meta)
        if event_name:
            print(f"Kalshi watching: {event_name}")
        if event_date:
            print(f"Event date:      {event_date}")
        if fallback_speaker:
            print(f"Speaker:         {fallback_speaker}")
    except Exception:
        pass

    transcript = load_transcript(
        args,
        fallback_speaker=fallback_speaker,
        event_date=event_date,
        event_name=event_name,
    )
    if not transcript:
        raise SystemExit(
            f"\ntranscript_bot found no usable transcript for {event_ticker}.\n"
            f"Try --file or --paste to supply the transcript manually."
        )
    print(f"Transcript: {len(transcript):,} chars\n")
    score_event(event_ticker, transcript, show_bets=True, speaker=fallback_speaker)


def run_speaker_mode(speaker: str, args) -> None:
    print(f"\nDiscovering Kalshi events that mention {speaker!r} ...")
    events = find_speaker_events(speaker)
    if not events:
        raise SystemExit(
            f"No mention-style Kalshi events found for {speaker!r}.\n"
            f"Try --event with a specific ticker instead."
        )
    print(f"Found {len(events)} event(s):")
    for e in events:
        print(f"  - {e.get('event_ticker'):<35} {(e.get('title') or '')[:60]}")

    total_correct = total_n = 0
    scored_events = skipped_events = 0
    seen_transcript_hashes: set[str] = set()
    for e in events:
        ticker = e["event_ticker"]
        event_date = _parse_event_date(ticker)

        event_name = None
        try:
            meta = get_event_meta(ticker)
            event_name = _pick_event_name(meta)
            if event_name:
                print(f"\n  [{ticker}] Kalshi watching: {event_name}")
        except Exception:
            pass

        transcript = load_transcript(
            args,
            fallback_speaker=speaker,
            event_date=event_date,
            event_name=event_name,
        )
        if not transcript:
            print(f"  (skipping {ticker} — no usable transcript)")
            skipped_events += 1
            continue

        import hashlib
        h = hashlib.md5(transcript.encode()).hexdigest()
        if h in seen_transcript_hashes:
            print(f"  (skipping {ticker} — duplicate transcript, same text as a prior event)")
            skipped_events += 1
            continue
        seen_transcript_hashes.add(h)

        print(f"  Transcript: {len(transcript):,} chars", file=sys.stderr)
        c, n = score_event(ticker, transcript, show_bets=False, speaker=speaker)
        total_correct += c
        total_n += n
        scored_events += 1

    print("\n" + "=" * 40)
    if skipped_events:
        print(f"Skipped {skipped_events}/{len(events)} event(s) — no dated transcript")
    if total_n:
        print(f"OVERALL: {total_correct}/{total_n} "
              f"({100*total_correct/total_n:.0f}%) across {scored_events} scored event(s)")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--event",   help="Specific Kalshi event ticker")
    p.add_argument("--speaker", help="Speaker name (drives discovery + transcript fetch)")
    p.add_argument("--file",    help="Use a transcript file instead of fetching")
    p.add_argument("--paste",   action="store_true",
                   help="Paste transcript on stdin (Ctrl+D to end)")
    args = p.parse_args()

    if args.event:
        run_event_mode(args.event, args)
    elif args.speaker:
        run_speaker_mode(args.speaker, args)
    else:
        p.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
    # Background adapter threads (cancelled stragglers) are non-daemon and
    # would otherwise hold the interpreter open for minutes. Flush then force exit.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
