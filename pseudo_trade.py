"""
pseudo_trade.py
---------------
Fixed-holdout evaluation — the same iteration framework the friend uses.

Trains on all pre-cutoff warm rows, evaluates on training_data_holdout
(post-2026-03-01 settled markets). Every model change gets its own run
and the scorecard is directly comparable to the friend's iteration table.

Usage:
    python3 pseudo_trade.py            # train + evaluate
    python3 pseudo_trade.py --no-train # load saved model, evaluate only
"""

from __future__ import annotations

import sys
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, accuracy_score

import db
import kalshi_model


NO_MIN_EDGE  = 0.10   # matches run_pipeline._DEFAULT_NO_MIN_EDGE
YES_MIN_EDGE = 0.18   # matches run_pipeline._DEFAULT_YES_MIN_EDGE (model over-predicts YES)
MIN_WARM     = 0          # allow cold rows for NO bets; YES still requires n>=3 (enforced inside _simulate)
HOLDOUT_CUTOFF = kalshi_model.HOLDOUT_CUTOFF


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _calibration_error(y_true: np.ndarray, probs: np.ndarray,
                        n_bins: int = 5) -> float:
    """Mean absolute calibration error across equal-frequency bins."""
    order    = np.argsort(probs)
    y_sorted = y_true[order]
    p_sorted = probs[order]
    bins     = np.array_split(np.arange(len(probs)), n_bins)
    errors   = []
    for b in bins:
        if len(b) == 0:
            continue
        errors.append(abs(y_sorted[b].mean() - p_sorted[b].mean()))
    return float(np.mean(errors)) if errors else 0.0


def _simulate(y: np.ndarray, probs: np.ndarray,
              ko: np.ndarray, warm: np.ndarray,
              nsamp: Optional[np.ndarray] = None) -> dict:
    """Flat $1/bet P&L simulation — warm-gated, separate YES/NO thresholds."""
    max_no = kalshi_model._MAX_NO_BET_ODDS

    total = yes_pnl = no_pnl = warm_pnl = cold_pnl = 0.0
    n_bets = n_yes = n_no = n_correct = n_warm_bets = n_skipped_warm = 0
    evs = []

    for i in range(len(probs)):
        bettable = 0.04 < ko[i] < 0.96
        if not bettable:
            continue
        is_warm = bool(warm[i])
        if not is_warm:
            n_skipped_warm += 1
            continue

        p    = float(probs[i])
        odds = float(ko[i])
        yi   = int(y[i])
        n_s  = int(nsamp[i]) if nsamp is not None else 99

        ev_yes = p - odds
        ev_no  = (1.0 - p) - (1.0 - odds)

        # YES bets require ≥3 events of history — n_samples=2 predictions are
        # dominated by hit_rate_lifetime which is too noisy for YES confidence
        if ev_yes >= YES_MIN_EDGE and ev_yes >= ev_no and n_s >= 3:
            side, entry = "YES", odds
            won = (yi == 1)
            evs.append(ev_yes)
        elif ev_no >= NO_MIN_EDGE and odds <= max_no:
            side, entry = "NO", (1.0 - odds)
            won = (yi == 0)
            evs.append(ev_no)
        else:
            continue

        pnl = (1.0 - entry) if won else -entry
        total += pnl
        n_bets += 1
        if won:
            n_correct += 1
        if side == "YES":
            yes_pnl += pnl; n_yes += 1
        else:
            no_pnl += pnl; n_no += 1
        if is_warm:
            warm_pnl += pnl; n_warm_bets += 1
        else:
            cold_pnl += pnl

    return {
        "pnl":           round(total * 100),   # cents
        "yes_pnl":       round(yes_pnl * 100),
        "no_pnl":        round(no_pnl * 100),
        "warm_pnl":      round(warm_pnl * 100),
        "cold_pnl":      round(cold_pnl * 100),
        "n_bets":        n_bets,
        "n_yes":         n_yes,
        "n_no":          n_no,
        "n_correct":     n_correct,
        "n_skipped_warm": n_skipped_warm,
        "bet_accuracy":  n_correct / max(n_bets, 1),
        "mean_ev":       float(np.mean(evs)) if evs else 0.0,
    }


def _print_scorecard(
    train_rows: int,
    holdout_rows: int,
    brier: float,
    baseline_brier: float,
    cal_error: float,
    auc: float,
    accuracy: float,
    sim: dict,
    bettable: int,
    warm_holdout: int,
) -> None:
    sep = "=" * 60
    print(f"\n{sep}")
    print("PSEUDO-TRADE SCORECARD  (fixed holdout, post-2026-03-01)")
    print(sep)

    print(f"\n  {'Metric':<30} {'Ours':>10}  {'Friend Iter #6':>15}")
    print(f"  {'-'*58}")

    def row(name, val, ref):
        print(f"  {name:<30} {val:>10}  {ref:>15}")

    row("Train rows",         f"{train_rows:,}",        "1,318")
    row("Holdout rows",       f"{holdout_rows:,}",      "853")
    row("Brier score",        f"{brier:.4f}",           "0.2698")
    row("Baseline Brier",     f"{baseline_brier:.4f}",  "0.2978")
    row("Improvement",        f"{baseline_brier-brier:+.4f}", "+0.0280")
    row("Calibration error",  f"{cal_error:.4f}",       "0.2251")
    row("AUC-ROC",            f"{auc:.3f}",             "—")
    row("Accuracy (≥0.50)",   f"{accuracy:.1%}",        "60.5%")
    row("Bettable rows",      f"{bettable}",            "365")
    row("Warm bettable",      f"{warm_holdout}",        "—")
    row("Bets placed",        f"{sim['n_bets']}",       "301")
    row("Bets skipped (cold)",f"{sim['n_skipped_warm']}","20")
    row("Bet accuracy",       f"{sim['bet_accuracy']:.1%}", "49.5%")
    row("Mean EV per bet",    f"{sim['mean_ev']:.4f}",  "0.3012")
    row("Simulated P&L",      f"{sim['pnl']:+d}¢",     "+2,036¢")
    row("YES bet P&L",        f"{sim['yes_pnl']:+d}¢",  "+623¢")
    row("NO  bet P&L",        f"{sim['no_pnl']:+d}¢",   "+1,413¢")
    row("Warm row P&L",       f"{sim['warm_pnl']:+d}¢", "+2,036¢")

    if sim["n_bets"] and holdout_rows:
        roi = sim["pnl"] / sim["n_bets"]
        projected = roi * 301
        print(f"\n  ROI per bet      : {roi:+.1f}¢")
        print(f"  Projected @301   : {projected:+.0f}¢  (friend's bet count)")

    print(f"\n{sep}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(retrain: bool = True) -> None:

    # ── 1. Train ────────────────────────────────────────────────────────────
    if retrain:
        print("Training on pre-cutoff warm rows ...")
        kalshi_model.train(save=True, verbose=True)
    else:
        print("Skipping retrain — loading saved model ...")

    # ── 2. Load holdout ─────────────────────────────────────────────────────
    holdout_rows = db.get_holdout_data()
    if not holdout_rows:
        print("\n[!] No holdout data found.")
        print("    Run:  python3 harvest_training_data.py --holdout")
        return

    df = pd.DataFrame(holdout_rows)
    df["_word"]       = df["word"]
    df["_event_type"] = df["event_type"]
    df["_speaker"]    = df["speaker"]

    print(f"\nHoldout loaded: {len(df)} rows, "
          f"{df['event_ticker'].nunique()} events, "
          f"{df['did_say_word'].mean():.1%} hit rate")

    # ── 3. Build features using saved priors ─────────────────────────────────
    priors = kalshi_model._get_word_priors()
    ens    = kalshi_model._get_ensemble()
    cal    = kalshi_model._get_calibrator()

    X        = kalshi_model._build_features_with_priors(df, priors)
    raw_prob = np.mean([b.predict(X) for b in ens], axis=0)

    # Blend in LR ensemble member — must match predict_proba() behavior in run_pipeline.
    lr_bundle = kalshi_model._load_lr_model()
    if lr_bundle is not None:
        X_lr = X.copy()
        for col, med in lr_bundle["col_medians"].items():
            if col in X_lr.columns:
                X_lr[col] = X_lr[col].fillna(med)
        X_lr_s  = lr_bundle["scaler"].transform(X_lr.values)
        lr_prob = lr_bundle["model"].predict_proba(X_lr_s)[:, 1]
        raw_prob = (raw_prob * 10 + lr_prob) / 11

    probs    = (kalshi_model._apply_calibrator(cal, raw_prob)
                if cal is not None else raw_prob)
    probs    = kalshi_model._post_process_probs(probs, df["kalshi_odds"].values.astype(float))

    y     = df["did_say_word"].astype(int).values
    ko    = df["kalshi_odds"].values.astype(float)
    warm  = (df["n_samples_lifetime"] >= MIN_WARM).values
    nsamp = df["n_samples_lifetime"].values.astype(int)

    # ── 4. Metrics ───────────────────────────────────────────────────────────
    brier          = brier_score_loss(y, probs)
    baseline_brier = float(np.mean(y) * (1 - np.mean(y)))   # hit_rate * (1-hit_rate)
    cal_error      = _calibration_error(y, probs)
    try:
        auc = roc_auc_score(y, probs)
    except Exception:
        auc = float("nan")
    preds    = (probs >= 0.50).astype(int)
    accuracy = accuracy_score(y, preds)

    bettable     = int(((ko > 0.04) & (ko < 0.96)).sum())
    warm_bettable = int(((ko > 0.04) & (ko < 0.96) & warm).sum())

    sim = _simulate(y, probs, ko, warm, nsamp=nsamp)

    # ── 5. Calibration buckets ───────────────────────────────────────────────
    print("\nCALIBRATION BUCKETS (holdout)")
    print(f"  {'Bucket':<10} {'N':>5} {'Predicted':>10} {'Actual':>8} {'Error':>8}")
    print(f"  {'-'*45}")
    edges = [0.0, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (probs >= lo) & (probs < hi)
        if mask.sum() < 3:
            continue
        pred_mean = probs[mask].mean()
        act_mean  = y[mask].mean()
        err       = abs(pred_mean - act_mean)
        print(f"  {lo:.1f}–{hi:.1f}    {mask.sum():>5}  {pred_mean:>10.3f}  {act_mean:>8.3f}  {err:>8.3f}")

    # ── 6. Per-speaker breakdown ─────────────────────────────────────────────
    print("\nPER-SPEAKER (holdout)")
    print(f"  {'Speaker':<20} {'Rows':>5} {'Brier':>7} {'Bets':>5} {'P&L':>7}")
    print(f"  {'-'*47}")
    for sp in sorted(df["speaker"].unique()):
        mask = df["speaker"] == sp
        if mask.sum() == 0:
            continue
        sp_brier = brier_score_loss(y[mask], probs[mask])
        sp_sim   = _simulate(y[mask], probs[mask], ko[mask], warm[mask], nsamp=nsamp[mask])
        print(f"  {sp:<20} {mask.sum():>5}  {sp_brier:.4f}  {sp_sim['n_bets']:>5}  "
              f"{sp_sim['pnl']:>+6}¢")

    # ── 7. Full scorecard ────────────────────────────────────────────────────
    _print_scorecard(
        train_rows     = kalshi_model._count_real_rows(),
        holdout_rows   = len(df),
        brier          = brier,
        baseline_brier = baseline_brier,
        cal_error      = cal_error,
        auc            = auc,
        accuracy       = accuracy,
        sim            = sim,
        bettable       = bettable,
        warm_holdout   = warm_bettable,
    )


if __name__ == "__main__":
    retrain = "--no-train" not in sys.argv
    run(retrain=retrain)
