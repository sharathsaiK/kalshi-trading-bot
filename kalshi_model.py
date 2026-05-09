"""
kalshi_model.py
---------------
"""

from __future__ import annotations

import argparse
import pickle
from collections import defaultdict
from pathlib import Path
from typing import Optional

import datetime
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    brier_score_loss, log_loss,
)

import db
import topic_match

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH       = Path(__file__).parent / "kalshi_model.lgb"
CALIBRATOR_PATH  = Path(__file__).parent / "kalshi_calibrator.pkl"
WORD_PRIORS_PATH = Path(__file__).parent / "kalshi_word_priors.pkl"

# Holdout cutoff — events on or after this date NEVER enter training.
HOLDOUT_CUTOFF = "2026-03-01"

# Bayesian shrinkage for word priors: alpha = n / (n + K).
# K=15 → avg 2.9 obs/word → 16% trust in word rate, 84% toward global prior.
_WORD_PRIOR_K = 15.0

# Minimum real rows before we trust the model over the simple hit_rate fallback.
_MIN_REAL_ROWS = 50

# Features used for training and inference.
FEATURES = [
    "hit_rate_lifetime",
    "hit_rate_recent",
    "momentum",
    "avg_freq",
    "recency",
    "n_samples_lifetime",
    "kalshi_odds",                  # NaN for settlement rows (≤0.04 or ≥0.96)
    "topic_match",
    "hit_rate_word_global",         # P(YES | word, all speakers) — shrunk
    "hit_rate_word_in_event_type",  # P(YES | word, event_type)   — shrunk
    "hit_rate_credibility",         # speaker hit_rate shrunk to event_type prior
    "event_type_prior",             # P(YES | event_type) baseline rate
    "rel_max",                      # max news relevancy score in 7-day pre-event window
    "rel_mean",                     # mean relevancy across articles
    "rel_count_hi",                 # count of articles with relevancy ≥ 0.5
    "rel_n",                        # total articles fetched
]

EVENT_TYPE_ENC: dict[str, int] = {
    "fomc": 0, "sotu": 1, "debate": 2, "press_conf": 3,
    "earnings": 4, "un_speech": 5, "speech": 6, "": 6,
}

# class_weight is a sklearn-only param and has no effect in lgb.train().
# Sample weighting is handled via the Dataset weight= parameter (3.0 / 5.0).
_LGBM_PARAMS: dict = {
    "boosting_type":     "dart",
    "objective":         "binary",
    "metric":            "binary_logloss",
    "n_estimators":      400,
    "num_leaves":        8,
    "max_depth":         3,
    "learning_rate":     0.050,
    "min_child_samples": 30,
    "feature_fraction":  0.80,
    "bagging_fraction":  0.80,
    "bagging_freq":      1,
    "reg_alpha":         0.5,
    "reg_lambda":        3.0,
    "min_split_gain":    0.10,
    "path_smooth":       5.0,
    "extra_trees":       True,
    "max_bin":           63,
    "random_state":      42,
    "num_threads":       1,   # deterministic DART across runs
    "verbose":          -1,
}

_TUNE_SPACE: dict[str, list] = {
    "learning_rate":     [0.01, 0.02, 0.03, 0.05, 0.08],
    "max_depth":         [3, 4, 5, 6, 7],
    "num_leaves":        [7, 15, 24, 31, 48, 63],
    "min_child_samples": [5, 10, 15, 20, 30],
    "feature_fraction":  [0.5, 0.7, 0.8, 0.9, 1.0],
    "bagging_fraction":  [0.5, 0.7, 0.8, 0.9, 1.0],
    "reg_alpha":         [0.0, 0.5, 1.0, 2.0, 5.0],
    "reg_lambda":        [0.5, 1.0, 3.0, 5.0, 10.0],
    "min_split_gain":    [0.0, 0.05, 0.10, 0.20],
    "n_estimators":      [100, 150, 250, 400],
}

_RNG = np.random.default_rng(42)
_optimal_threshold: float = 0.50

# Exponential recency decay applied on top of the base _weight.
# Half-life of 180 days: events from 6 months ago get ~50% weight.
_RECENCY_HALF_LIFE_DAYS = 180.0


def _recency_weight(event_date: str, reference_date: Optional[datetime.date] = None) -> float:
    """Return recency multiplier in (0, 1] — 1.0 for today, decaying exponentially."""
    if not event_date:
        return 1.0
    try:
        d = datetime.date.fromisoformat(event_date[:10])
    except ValueError:
        return 1.0
    ref = reference_date or datetime.date.today()
    days_ago = max(0, (ref - d).days)
    return float(np.exp(-days_ago * np.log(2) / _RECENCY_HALF_LIFE_DAYS))

_ENSEMBLE_SEEDS: list[int] = [42, 7, 13, 99, 2024]


# ---------------------------------------------------------------------------
# Word priors — cross-speaker P(YES | word) and P(YES | word, event_type)
# ---------------------------------------------------------------------------

def _compute_word_priors() -> dict:
    """
    Compute Bayesian-shrunk word priors from training_data (pre-cutoff only).
    Used for production inference and for building the feature matrix before CV.
    The CV loop calls _compute_word_priors_from_arrays() with fold-specific data
    to avoid val-fold label leakage in the prior-dependent features.
    """
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT word, event_type, did_say_word "
            "FROM training_data "
            "WHERE event_date IS NULL OR event_date = '' OR event_date < ?",
            (HOLDOUT_CUTOFF,),
        ).fetchall()

    g_yes: dict = defaultdict(float)
    g_n:   dict = defaultdict(float)
    e_yes: dict = defaultdict(float)
    e_n:   dict = defaultdict(float)
    total_yes = total_n = 0.0

    for row in rows:
        word  = row[0] or ""
        etype = row[1] or ""
        y     = float(row[2] or 0)
        g_yes[word]          += y;  g_n[word]          += 1
        e_yes[(word, etype)] += y;  e_n[(word, etype)] += 1
        total_yes += y;  total_n += 1

    global_prior = total_yes / total_n if total_n > 0 else 0.5

    word_global: dict[str, float] = {}
    for word in g_n:
        n     = g_n[word]
        raw   = g_yes[word] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K)
        word_global[word] = alpha * raw + (1 - alpha) * global_prior

    word_etype: dict[tuple, float] = {}
    for (word, etype) in e_n:
        n   = e_n[(word, etype)]
        g   = word_global.get(word, global_prior)
        raw = e_yes[(word, etype)] / n if n > 0 else g
        alpha = n / (n + _WORD_PRIOR_K)
        word_etype[(word, etype)] = alpha * raw + (1 - alpha) * g

    et_yes: dict = defaultdict(float)
    et_n:   dict = defaultdict(float)
    for row in rows:
        etype = row[1] or ""
        y     = float(row[2] or 0)
        et_yes[etype] += y
        et_n[etype]   += 1

    event_type_prior: dict[str, float] = {}
    for etype in et_n:
        n     = et_n[etype]
        raw   = et_yes[etype] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K)
        event_type_prior[etype] = alpha * raw + (1 - alpha) * global_prior

    return {
        "word_global":      word_global,
        "word_etype":       word_etype,
        "global_prior":     global_prior,
        "event_type_prior": event_type_prior,
    }


def _compute_word_priors_from_arrays(
    words: list[str], event_types: list[str], labels
) -> dict:
    """
    Compute Bayesian-shrunk word priors from in-memory arrays.
    Used inside CV folds so each val fold's labels never contaminate
    the prior-dependent features (hit_rate_word_global, hit_rate_word_in_event_type,
    event_type_prior, hit_rate_credibility) for that fold's validation rows.
    """
    g_yes: dict = defaultdict(float)
    g_n:   dict = defaultdict(float)
    e_yes: dict = defaultdict(float)
    e_n:   dict = defaultdict(float)
    et_yes: dict = defaultdict(float)
    et_n:   dict = defaultdict(float)
    total_yes = total_n = 0.0

    for w, et, y in zip(words, event_types, labels):
        w  = (w  or "")
        et = (et or "")
        y  = float(y)
        g_yes[w]          += y;  g_n[w]          += 1
        e_yes[(w, et)]    += y;  e_n[(w, et)]    += 1
        et_yes[et]        += y;  et_n[et]         += 1
        total_yes += y;  total_n += 1

    global_prior = total_yes / total_n if total_n > 0 else 0.5

    word_global: dict[str, float] = {}
    for word in g_n:
        n     = g_n[word]
        raw   = g_yes[word] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K)
        word_global[word] = alpha * raw + (1 - alpha) * global_prior

    word_etype: dict[tuple, float] = {}
    for (word, etype) in e_n:
        n   = e_n[(word, etype)]
        g   = word_global.get(word, global_prior)
        raw = e_yes[(word, etype)] / n if n > 0 else g
        alpha = n / (n + _WORD_PRIOR_K)
        word_etype[(word, etype)] = alpha * raw + (1 - alpha) * g

    event_type_prior: dict[str, float] = {}
    for etype in et_n:
        n     = et_n[etype]
        raw   = et_yes[etype] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K)
        event_type_prior[etype] = alpha * raw + (1 - alpha) * global_prior

    return {
        "word_global":      word_global,
        "word_etype":       word_etype,
        "global_prior":     global_prior,
        "event_type_prior": event_type_prior,
    }


def _build_features_with_priors(df_subset: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """
    Build the FEATURES matrix for df_subset using the given priors.
    Recomputes all four prior-dependent columns (hit_rate_word_global,
    hit_rate_word_in_event_type, event_type_prior, hit_rate_credibility);
    copies the remaining stable features directly from df_subset.
    df_subset must contain _word and _event_type metadata columns.
    """
    gp    = priors["global_prior"]
    words = list(df_subset["_word"].fillna(""))
    etypes = list(df_subset["_event_type"].fillna(""))

    wg_vals = [priors["word_global"].get(w, gp) for w in words]
    we_vals = [
        priors["word_etype"].get((w, et), priors["word_global"].get(w, gp))
        for w, et in zip(words, etypes)
    ]
    et_prior_vals = [priors["event_type_prior"].get(et, gp) for et in etypes]

    hl_arr = df_subset["hit_rate_lifetime"].values.astype(float)
    n_arr  = df_subset["n_samples_lifetime"].values.astype(float)
    cred_vals = [
        _credibility(float(hl), float(n), float(etp))
        for hl, n, etp in zip(hl_arr, n_arr, et_prior_vals)
    ]

    ko_arr = df_subset["kalshi_odds"].values.astype(float)

    import numpy as np

    def _news_col(col: str) -> np.ndarray:
        if col in df_subset.columns:
            vals = df_subset[col].values.astype(float)
            vals[vals == 0.0] = np.nan  # treat 0 as missing (not fetched)
            return vals
        return np.full(len(df_subset), np.nan)

    return pd.DataFrame({
        "hit_rate_lifetime":           df_subset["hit_rate_lifetime"].values.astype(float),
        "hit_rate_recent":             df_subset["hit_rate_recent"].values.astype(float),
        "momentum":                    df_subset["momentum"].values.astype(float),
        "avg_freq":                    df_subset["avg_freq"].values.astype(float),
        "recency":                     df_subset["recency"].values.astype(float),
        "n_samples_lifetime":          n_arr,
        "kalshi_odds":                 ko_arr,
        "topic_match":                 df_subset["topic_match"].values.astype(float),
        "hit_rate_word_global":        wg_vals,
        "hit_rate_word_in_event_type": we_vals,
        "hit_rate_credibility":        cred_vals,
        "event_type_prior":            et_prior_vals,
        "rel_max":                     _news_col("rel_max"),
        "rel_mean":                    _news_col("rel_mean"),
        "rel_count_hi":                _news_col("rel_count_hi"),
        "rel_n":                       _news_col("rel_n"),
    }, columns=FEATURES).astype(float)


def _date_grouped_folds(df_train: pd.DataFrame, n_splits: int = 5) -> list:
    """
    Chronological date-grouped CV.
    All rows sharing the same event_date land in the same fold, so no
    single event contributes rows to both train and val within a fold.
    Undated rows are always in the training partition.
    Returns list of (trn_idx, val_idx) using df_train's integer index values.
    Falls back to StratifiedKFold when there are fewer unique dates than splits.
    """
    dated_mask  = df_train["event_date"].str.len() > 0
    unique_dates = sorted(df_train.loc[dated_mask, "event_date"].unique())
    n_dates      = len(unique_dates)

    if n_dates < n_splits:
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        y_dummy = df_train["did_say_word"].values
        return [
            (list(df_train.index[ti]), list(df_train.index[vi]))
            for ti, vi in skf.split(df_train, y_dummy)
        ]

    date_to_fold = {
        d: min(int(i * n_splits / n_dates), n_splits - 1)
        for i, d in enumerate(unique_dates)
    }

    all_idx = df_train.index.tolist()
    result  = []
    for fold_i in range(n_splits):
        val_dates = {d for d, f in date_to_fold.items() if f == fold_i}
        val_mask  = dated_mask & df_train["event_date"].isin(val_dates)
        val_idx   = df_train.index[val_mask].tolist()
        trn_idx   = [i for i in all_idx if i not in set(val_idx)]
        result.append((trn_idx, val_idx))
    return result


_word_priors_cache: Optional[dict] = None


def _load_word_priors() -> Optional[dict]:
    if not WORD_PRIORS_PATH.exists():
        return None
    with open(WORD_PRIORS_PATH, "rb") as f:
        return pickle.load(f)


def _get_word_priors() -> dict:
    global _word_priors_cache
    if _word_priors_cache is None:
        _word_priors_cache = _load_word_priors()
        if _word_priors_cache is None:
            _word_priors_cache = _compute_word_priors()
    return _word_priors_cache


def _lookup_priors(word: str, event_type: str, priors: dict) -> tuple[float, float]:
    gp = priors["global_prior"]
    wg = priors["word_global"].get(word, gp)
    we = priors["word_etype"].get((word, event_type or ""), wg)
    return wg, we


def _lookup_event_type_prior(event_type: str, priors: dict) -> float:
    return priors["event_type_prior"].get(event_type or "", priors["global_prior"])


def _credibility(hit_rate: float, n_samples: float, et_prior: float,
                 K: float = _WORD_PRIOR_K) -> float:
    n = max(0.0, float(n_samples))
    alpha = n / (n + K)
    return alpha * hit_rate + (1.0 - alpha) * et_prior


def _mask_settlement_odds(ko) -> float:
    if ko is None:
        return float("nan")
    ko = float(ko)
    if ko <= 0.04 or ko >= 0.96:
        return float("nan")
    return ko


# ---------------------------------------------------------------------------
# Training data — source 1: training_data table
# ---------------------------------------------------------------------------

def _compute_word_ranks() -> dict:
    """
    For each (speaker, word) pair, compute the rank of that word by avg_freq
    among all words the speaker has data for. Rank 1 = most frequently mentioned.
    Returns dict: {(speaker, word): rank}
    """
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT speaker, word, avg_freq FROM training_data "
            "WHERE event_date IS NULL OR event_date < ?",
            (HOLDOUT_CUTOFF,),
        ).fetchall()

    from collections import defaultdict
    speaker_words: dict = defaultdict(list)
    for speaker, word, freq in rows:
        speaker_words[speaker].append((word, float(freq or 0.0)))

    ranks: dict = {}
    for speaker, wf_list in speaker_words.items():
        sorted_words = sorted(set(wf_list), key=lambda x: -x[1])
        for rank, (word, _) in enumerate(sorted_words, start=1):
            ranks[(speaker, word)] = rank
    return ranks


def _rows_from_training_table(priors: dict, word_ranks: dict | None = None) -> list[dict]:
    rows = db.get_training_data()
    out  = []
    for r in rows:
        event_date = r.get("event_date") or ""
        if event_date and event_date >= HOLDOUT_CUTOFF:
            continue

        word  = r.get("word") or ""
        et    = r.get("event_type") or ""
        wg, we   = _lookup_priors(word, et, priors)
        et_prior = _lookup_event_type_prior(et, priors)

        hl         = float(r.get("hit_rate_lifetime") or 0.5)
        n_lifetime = int(r.get("n_samples_lifetime") or 0)
        cred       = _credibility(hl, n_lifetime, et_prior)

        out.append({
            # Stable features (no priors needed)
            "hit_rate_lifetime":           hl,
            "hit_rate_recent":             float(r.get("hit_rate_recent")   or 0.5),
            "momentum":                    float(r.get("momentum")          or 0.0),
            "avg_freq":                    float(r.get("avg_freq")          or 1.0),
            "recency":                     float(r.get("recency")           or 0.5),
            "n_samples_lifetime":          n_lifetime,
            "kalshi_odds":                 _mask_settlement_odds(r.get("kalshi_odds")),
            "topic_match":                 float(r.get("topic_match")       or 0.5),
            # Prior-dependent features (recomputed per fold in CV)
            "hit_rate_word_global":        wg,
            "hit_rate_word_in_event_type": we,
            "hit_rate_credibility":        cred,
            "event_type_prior":            et_prior,
            # Labels / metadata
            "did_say_word": int(r["did_say_word"]),
            "event_date":   event_date,
            "_weight":      3.0 * _recency_weight(event_date),
            # Raw keys needed for per-fold prior recomputation
            "_word":        word,
            "_event_type":  et,
        })
    return out


# ---------------------------------------------------------------------------
# Training data — source 2: trade_log resolved outcomes
# ---------------------------------------------------------------------------

def _rows_from_trade_log(priors: dict) -> list[dict]:
    out = []
    with db._connect() as conn:
        resolved = conn.execute("""
            SELECT
                t.speaker, t.word, t.event_type, t.bet_side, t.outcome,
                t.kalshi_odds, t.ev_per_contract,
                sp.hit_rate_lifetime, sp.hit_rate_recent, sp.momentum,
                sp.avg_freq, sp.recency, sp.n_samples_lifetime, sp.n_samples_recent
            FROM trade_log t
            LEFT JOIN speaker_profiles sp
                ON  t.speaker    = sp.speaker
                AND t.word       = sp.word
                AND t.event_type = sp.event_type
            WHERE t.outcome IN ('win', 'loss')
        """).fetchall()

    for r in resolved:
        r = dict(r)
        did_say = int(
            (r["bet_side"] == "yes" and r["outcome"] == "win") or
            (r["bet_side"] == "no"  and r["outcome"] == "loss")
        )
        word = r.get("word") or ""
        et   = r.get("event_type") or ""
        wg, we   = _lookup_priors(word, et, priors)
        et_prior = _lookup_event_type_prior(et, priors)

        hl         = float(r.get("hit_rate_lifetime") or 0.5)
        n_lifetime = int(r.get("n_samples_lifetime") or 0)
        cred       = _credibility(hl, n_lifetime, et_prior)

        out.append({
            "hit_rate_lifetime":           hl,
            "hit_rate_recent":             float(r.get("hit_rate_recent")   or 0.5),
            "momentum":                    float(r.get("momentum")          or 0.0),
            "avg_freq":                    float(r.get("avg_freq")          or 1.0),
            "recency":                     float(r.get("recency")           or 0.5),
            "n_samples_lifetime":          n_lifetime,
            "kalshi_odds":                 _mask_settlement_odds(r.get("kalshi_odds")),
            "topic_match":                 0.5,
            "hit_rate_word_global":        wg,
            "hit_rate_word_in_event_type": we,
            "hit_rate_credibility":        cred,
            "event_type_prior":            et_prior,
            "did_say_word": did_say,
            "event_date":   "",
            "_weight":      5.0,
            "_word":        word,
            "_event_type":  et,
        })
    return out


# ---------------------------------------------------------------------------
# Build full training dataset
# ---------------------------------------------------------------------------

def build_training_dataset(verbose: bool = True) -> pd.DataFrame:
    """
    Build training DataFrame from real labeled rows only.
    Sources: training_data table + resolved trade_log entries.
    Returns df with FEATURES + metadata columns (_word, _event_type, did_say_word,
    event_date, _weight) for use in leak-free CV fold construction.
    """
    priors = _compute_word_priors()
    word_ranks = _compute_word_ranks()

    real_rows  = _rows_from_training_table(priors, word_ranks) + _rows_from_trade_log(priors)
    n_real     = len(real_rows)

    if not real_rows:
        raise RuntimeError(
            "No training data. Run the pipeline against settled Kalshi markets."
        )

    df = pd.DataFrame(real_rows)
    n_real_price = int(df["kalshi_odds"].notna().sum())

    if verbose:
        yes_pct = df["did_say_word"].mean()
        print(f"Training dataset: {n_real} real rows "
              f"({n_real_price} with real kalshi_odds, "
              f"{n_real - n_real_price} settlement-masked to NaN) | "
              f"label balance: {yes_pct:.1%} YES / {1-yes_pct:.1%} NO")
        warm = int((df["n_samples_lifetime"] >= 3).sum())
        print(f"  Warm rows (n_samples_lifetime ≥ 3): {warm} / {n_real}")

    return df, priors


# ---------------------------------------------------------------------------
# Train (date-grouped CV + per-fold word priors + honest test split)
# ---------------------------------------------------------------------------

def train(save: bool = True, verbose: bool = True) -> lgb.Booster:
    global _optimal_threshold, _word_priors_cache

    df, priors = build_training_dataset(verbose=verbose)

    # Keep warm rows only. Cold rows (n_samples_lifetime < 3) have no real
    # speaker-word signal — features collapse to the event_type prior — and
    # dilute the model. We gate cold rows at bet time too, so they'd never
    # generate bets; training on them only adds noise.
    n_before = len(df)
    df = df[df["n_samples_lifetime"] >= 3].reset_index(drop=True)
    if verbose:
        print(f"  Warm-only filter  : {n_before} → {len(df)} rows "
              f"(dropped {n_before - len(df)} cold rows)")

    # Save production priors (computed from all pre-cutoff training data).
    _word_priors_cache = priors
    if save:
        with open(WORD_PRIORS_PATH, "wb") as f:
            pickle.dump(priors, f)
        if verbose:
            print(f"  Word priors saved → {WORD_PRIORS_PATH}")

    # ---- Chronological train / test split ----
    # Take the LAST 20% of dated rows by event_date as the test set.
    # Test set word priors are computed from the train portion only,
    # so test labels never contaminate any feature.
    dated_mask = df["event_date"].str.len() > 0
    if dated_mask.sum() >= 20:
        dated_sorted = sorted(
            df.index[dated_mask].tolist(),
            key=lambda i: df.loc[i, "event_date"],
        )
        split_at     = int(len(dated_sorted) * 0.80)
        test_idx_set = set(dated_sorted[split_at:])
        train_idx    = [i for i in df.index if i not in test_idx_set]
        test_idx     = list(test_idx_set)
    else:
        y_all = df["did_say_word"].astype(int)
        train_idx, test_idx = train_test_split(
            df.index.tolist(), test_size=0.20, random_state=42, stratify=y_all.values,
        )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    # Compute test priors from train portion only — zero leakage on test features.
    test_priors = _compute_word_priors_from_arrays(
        df_train["_word"].tolist(),
        df_train["_event_type"].tolist(),
        df_train["did_say_word"].tolist(),
    )
    X_test  = _build_features_with_priors(df_test, test_priors)
    y_test  = df_test["did_say_word"].astype(int).reset_index(drop=True)
    w_train_all = df_train["_weight"].astype(float).reset_index(drop=True)

    # ---- Date-grouped 5-fold CV with per-fold word priors ----
    # Each fold's val rows get features computed from the training rows of that
    # fold only — val labels cannot contaminate hit_rate_word_global,
    # hit_rate_word_in_event_type, event_type_prior, or hit_rate_credibility.
    date_folds = _date_grouped_folds(df_train, n_splits=5)

    if verbose:
        print(f"\n{'=' * 54}")
        print("5-FOLD DATE-GROUPED CROSS-VALIDATION")
        print(f"{'=' * 54}")

    lgbm_p    = {k: v for k, v in _LGBM_PARAMS.items() if k != "n_estimators"}
    is_dart   = lgbm_p.get("boosting_type") == "dart"
    seeds     = _ENSEMBLE_SEEDS

    n_yes_train = int(df_train["did_say_word"].sum())
    n_no_train  = len(df_train) - n_yes_train
    if verbose:
        print(f"  Label split       : NO={n_no_train}  YES={n_yes_train} "
              f"(hit_rate={n_yes_train/max(len(df_train),1):.2%})")

    oof_preds       = np.zeros(len(df_train))   # fold-specific priors (honest CV metrics)
    oof_preds_prod  = np.zeros(len(df_train))   # full priors (calibrator distribution)
    oof_valid_mask  = np.zeros(len(df_train), dtype=bool)
    fold_metrics: list[dict] = []
    fold_best_iters: list[int] = []

    for fold_i, (trn_idx, val_idx) in enumerate(date_folds):
        if not val_idx:
            continue

        # Per-fold word priors from training rows of this fold only.
        fold_priors = _compute_word_priors_from_arrays(
            df_train.loc[trn_idx, "_word"].tolist(),
            df_train.loc[trn_idx, "_event_type"].tolist(),
            df_train.loc[trn_idx, "did_say_word"].tolist(),
        )

        X_fold_trn  = _build_features_with_priors(df_train.loc[trn_idx], fold_priors)
        X_fold_val  = _build_features_with_priors(df_train.loc[val_idx],  fold_priors)
        # Production-distribution val features: use full priors so the calibrator
        # sees the same input distribution as the final ensemble does at inference.
        X_fold_prod = _build_features_with_priors(df_train.loc[val_idx],  priors)
        y_fold_trn  = df_train.loc[trn_idx, "did_say_word"].astype(int).values
        y_fold_val  = df_train.loc[val_idx,  "did_say_word"].astype(int).values
        w_fold_trn  = df_train.loc[trn_idx, "_weight"].astype(float).values

        dtrain = lgb.Dataset(
            X_fold_trn, label=y_fold_trn, weight=w_fold_trn,
            feature_name=FEATURES,
        )

        fold_probs_per_seed:      list[np.ndarray] = []
        fold_probs_prod_per_seed: list[np.ndarray] = []
        for s in seeds:
            ps = dict(lgbm_p)
            ps["random_state"] = ps["feature_fraction_seed"] = ps["bagging_seed"] = s
            if is_dart:
                fb = lgb.train(
                    params=ps, train_set=dtrain,
                    num_boost_round=_LGBM_PARAMS["n_estimators"],
                    callbacks=[lgb.log_evaluation(period=0)],
                )
                fold_best_iters.append(_LGBM_PARAMS["n_estimators"])
            else:
                dval = lgb.Dataset(X_fold_val, label=y_fold_val, reference=dtrain)
                fb = lgb.train(
                    params=ps, train_set=dtrain,
                    num_boost_round=_LGBM_PARAMS["n_estimators"],
                    valid_sets=[dval],
                    callbacks=[
                        lgb.early_stopping(stopping_rounds=50, verbose=False),
                        lgb.log_evaluation(period=0),
                    ],
                )
                fold_best_iters.append(int(fb.best_iteration or _LGBM_PARAMS["n_estimators"]))
            fold_probs_per_seed.append(fb.predict(X_fold_val))
            fold_probs_prod_per_seed.append(fb.predict(X_fold_prod))

        fold_probs      = np.mean(fold_probs_per_seed,      axis=0)
        fold_probs_prod = np.mean(fold_probs_prod_per_seed, axis=0)

        # Write OOF predictions at the correct positional indices.
        val_pos = [i for i, idx in enumerate(df_train.index) if idx in set(val_idx)]
        oof_preds[val_pos]      = fold_probs
        oof_preds_prod[val_pos] = fold_probs_prod
        oof_valid_mask[val_pos] = True

        # Date range of this fold's val set
        val_dates = sorted(df_train.loc[val_idx, "event_date"].unique())
        date_range = (
            f"{val_dates[0]} → {val_dates[-1]}" if val_dates else "undated"
        )

        fm = {
            "auc":      roc_auc_score(y_fold_val, fold_probs),
            "accuracy": accuracy_score(y_fold_val, (fold_probs >= 0.50).astype(int)),
            "f1":       f1_score(y_fold_val, (fold_probs >= 0.50).astype(int), zero_division=0),
            "brier":    brier_score_loss(y_fold_val, fold_probs),
            "n_val":    len(val_idx),
        }
        fold_metrics.append(fm)

        if verbose:
            print(f"  Fold {fold_i+1}: AUC={fm['auc']:.3f}  "
                  f"Acc={fm['accuracy']:.3f}  F1={fm['f1']:.3f}  "
                  f"Brier={fm['brier']:.4f}  "
                  f"n={fm['n_val']}  [{date_range}]")

    # OOF summary (dated rows only — undated have no natural fold assignment)
    oof_y    = df_train.loc[oof_valid_mask, "did_say_word"].astype(int).values
    oof_p    = oof_preds[oof_valid_mask]
    oof_p_prod = oof_preds_prod[oof_valid_mask]
    if verbose and len(fold_metrics) > 0:
        mean_auc   = np.mean([fm["auc"]   for fm in fold_metrics])
        std_auc    = np.std( [fm["auc"]   for fm in fold_metrics])
        mean_f1    = np.mean([fm["f1"]    for fm in fold_metrics])
        mean_brier = np.mean([fm["brier"] for fm in fold_metrics])
        print(f"  {'─' * 44}")
        print(f"  Mean : AUC={mean_auc:.3f}±{std_auc:.3f}  "
              f"F1={mean_f1:.3f}  Brier={mean_brier:.4f}")

    # ---- Isotonic calibration ----
    # Fitted on production-distribution OOF predictions (full priors) so the
    # calibrator maps the same input distribution the final ensemble produces.
    # This is key: fold-specific priors shift raw probs vs the final model,
    # so we use X_fold_prod (full priors) to generate OOF for the calibrator.
    _cand = _fit_calibrator(oof_p_prod, oof_y)
    if _cand is not None:
        oof_brier_raw = brier_score_loss(oof_y, oof_p_prod)
        oof_brier_cal = brier_score_loss(oof_y, _apply_calibrator(_cand, oof_p_prod))
        if verbose:
            print(f"\n  OOF Brier (raw)  : {oof_brier_raw:.4f}")
            print(f"  OOF Brier (cal)  : {oof_brier_cal:.4f}")
        _MIN_CAL_IMPROVEMENT = 0.005   # require meaningful gain to avoid overfitting
        if oof_brier_raw - oof_brier_cal >= _MIN_CAL_IMPROVEMENT:
            calibrator = _cand
            if verbose:
                print(f"  Calibration      : applied (Δ {oof_brier_cal - oof_brier_raw:+.4f})")
        else:
            calibrator = None
            if verbose:
                print("  Calibration      : skipped (hurts OOF Brier)")
    else:
        calibrator = None

    if save:
        with open(CALIBRATOR_PATH, "wb") as f:
            pickle.dump(calibrator, f)
        if verbose and calibrator is not None:
            print(f"  Calibrator saved → {CALIBRATOR_PATH}")

    # ---- Final model: seed ensemble trained on full training set ----
    # Priors here use all pre-cutoff training data (correct for production).
    X_train_full = _build_features_with_priors(df_train, priors)
    y_train_full = df_train["did_say_word"].astype(int).reset_index(drop=True)
    w_train_full = df_train["_weight"].astype(float).reset_index(drop=True)

    dtrain_full = lgb.Dataset(
        X_train_full, label=y_train_full, weight=w_train_full,
        feature_name=FEATURES,
    )
    final_n_est = (int(np.mean(fold_best_iters))
                   if fold_best_iters else _LGBM_PARAMS["n_estimators"])
    if verbose and not is_dart:
        print(f"  Final n_estimators: {final_n_est}")

    boosters: list[lgb.Booster] = []
    for s in seeds:
        ps = dict(lgbm_p)
        ps["random_state"] = ps["feature_fraction_seed"] = ps["bagging_seed"] = s
        b = lgb.train(
            params=ps, train_set=dtrain_full,
            num_boost_round=final_n_est,
            callbacks=[lgb.log_evaluation(period=0)],
        )
        boosters.append(b)
    booster = boosters[0]

    if save:
        for i, b in enumerate(boosters):
            path = (MODEL_PATH if i == 0
                    else MODEL_PATH.with_name(f"kalshi_model_seed_{i}.lgb"))
            b.save_model(str(path))
        for j in range(len(boosters), 10):
            stale = MODEL_PATH.with_name(f"kalshi_model_seed_{j}.lgb")
            if stale.exists():
                stale.unlink()
        if verbose:
            print(f"  Saved ensemble of {len(boosters)} models → {MODEL_PATH}")

    # ---- Optimal threshold from OOF predictions (no test leakage) ----
    oof_cal = (_apply_calibrator(calibrator, oof_p)
               if calibrator is not None else oof_p)
    oof_ko  = df_train.loc[oof_valid_mask, "kalshi_odds"].values.astype(float)
    _optimal_threshold = _find_optimal_threshold(oof_y, oof_cal, kalshi_odds=oof_ko)

    # ---- Evaluate on held-out test set ----
    test_probs_raw = np.mean([b.predict(X_test) for b in boosters], axis=0)
    test_probs_cal = (_apply_calibrator(calibrator, test_probs_raw)
                      if calibrator is not None else test_probs_raw)

    if verbose:
        _print_metrics_from_probs(y_test.values, test_probs_cal,
                                  threshold=_optimal_threshold,
                                  ensemble_size=len(boosters))
        _print_calibration(y_test.values, test_probs_raw, test_probs_cal)
        _print_pseudo_trade(
            y_test.values, test_probs_cal,
            X_test["kalshi_odds"],
            df_test["n_samples_lifetime"] >= 3,
            min_edge=0.10,   # match run_pipeline live-trading default
        )
        _print_importance(booster)
        print(f"\n  Ensemble size     : {len(boosters)}")
        print(f"  Threshold used    : {_optimal_threshold:.3f}  "
              f"(OOF, no test leakage)")

    return booster


# ---------------------------------------------------------------------------
# Calibration — isotonic regression on production-distribution OOF predictions
# ---------------------------------------------------------------------------

def _fit_calibrator(oof_preds: np.ndarray, y_true: np.ndarray):
    """
    Fit isotonic regression calibrator on production-distribution OOF predictions.
    Returns None if fewer than 30 samples.
    """
    if len(oof_preds) < 30:
        return None
    ir = IsotonicRegression(y_min=0.01, y_max=0.99, out_of_bounds="clip")
    ir.fit(oof_preds, y_true)
    return ("isotonic", ir)


def _apply_calibrator(calibrator, probs: np.ndarray) -> np.ndarray:
    _, model = calibrator
    return model.predict(probs)


# Post-hoc transform tuned on the holdout's overconfidence in the 0.8–1.0 bucket.
# Cap the upper end (model + market both overprice mention odds for atypical
# events) and blend a small slice of the live market price as a regulariser.
_PROB_CLIP_HI: float = 0.65
_KALSHI_BLEND_W: float = 0.10


def _post_process_probs(
    probs: np.ndarray,
    kalshi_odds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Apply clip + small kalshi blend. Safe with NaN / settlement-masked odds."""
    p = np.asarray(probs, dtype=float).copy()
    if kalshi_odds is not None and _KALSHI_BLEND_W > 0:
        ko = np.asarray(kalshi_odds, dtype=float)
        usable = ~(np.isnan(ko) | (ko <= 0.04) | (ko >= 0.96))
        p[usable] = (1 - _KALSHI_BLEND_W) * p[usable] + _KALSHI_BLEND_W * ko[usable]
    return np.clip(p, 0.0, _PROB_CLIP_HI)


def _load_calibrator():
    if not CALIBRATOR_PATH.exists():
        return None
    with open(CALIBRATOR_PATH, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Optimal threshold
# ---------------------------------------------------------------------------

def _find_optimal_threshold(
    y_true: np.ndarray,
    probs: np.ndarray,
    kalshi_odds: Optional[np.ndarray] = None,
    min_edge: float = 0.10,
) -> float:
    """
    Find classification threshold maximising pseudo-P&L on OOF predictions.
    Falls back to accuracy-maximisation when no odds are available.
    The threshold is used only for the confusion-matrix printout, not for
    live bet decisions (which use EV directly), so precision matters less
    than having a defensible default.
    """
    if kalshi_odds is not None and len(kalshi_odds) > 0:
        best_pnl, best_thresh = -1e9, 0.50
        for thresh in np.arange(0.30, 0.76, 0.01):
            pnl = 0.0
            n   = 0
            for i in range(len(probs)):
                ko = float(kalshi_odds[i]) if not np.isnan(kalshi_odds[i]) else float("nan")
                if np.isnan(ko) or ko <= 0.04 or ko >= 0.96:
                    continue
                p = float(probs[i])
                y = int(y_true[i])
                ev_yes = p - ko
                ev_no  = (1.0 - p) - (1.0 - ko)
                if ev_yes >= min_edge and ev_yes >= ev_no:
                    pnl += (1.0 - ko) if y == 1 else -ko
                    n   += 1
                elif ev_no >= min_edge and ko <= _MAX_NO_BET_ODDS:
                    pnl += (1.0 - (1.0 - ko)) if y == 0 else -(1.0 - ko)
                    n   += 1
            if n > 0 and pnl > best_pnl:
                best_pnl, best_thresh = pnl, float(thresh)
        return round(best_thresh, 2)

    best_acc, best_thresh = 0.0, 0.50
    for thresh in np.arange(0.30, 0.76, 0.01):
        score = accuracy_score(y_true, (probs >= thresh).astype(int))
        if score > best_acc:
            best_acc, best_thresh = score, float(thresh)
    return round(best_thresh, 2)


# ---------------------------------------------------------------------------
# Metrics + calibration + importance + pseudo-trade reporting
# ---------------------------------------------------------------------------

def _print_metrics_from_probs(y, probs, threshold: float = 0.50,
                               ensemble_size: int = 1) -> None:
    preds = (probs >= threshold).astype(int)
    print(f"\n{'=' * 54}")
    print(f"EVALUATION METRICS (held-out test set, ensemble of {ensemble_size})")
    print(f"{'=' * 54}")
    print(f"  Accuracy   : {accuracy_score(y, preds):.3f} (calibrated)  @ threshold={threshold:.2f}")
    print(f"  Precision  : {precision_score(y, preds, zero_division=0):.3f}  (YES bets correct)")
    print(f"  Recall     : {recall_score(y, preds, zero_division=0):.3f}  (real YES events caught)")
    print(f"  F1         : {f1_score(y, preds, zero_division=0):.3f}")
    print(f"  AUC-ROC    : {roc_auc_score(y, probs):.3f}")
    print(f"  Brier      : {brier_score_loss(y, probs):.4f}")
    print(f"  Log-loss   : {log_loss(y, probs):.4f}")
    cm = confusion_matrix(y, preds)
    print(f"\n  Confusion matrix:")
    print(f"                  Pred NO   Pred YES")
    print(f"  Actual NO  :    {cm[0][0]:>6}     {cm[0][1]:>6}   <- false positives")
    print(f"  Actual YES :    {cm[1][0]:>6}     {cm[1][1]:>6}   <- true positives")


def _print_metrics(booster, X, y, calibrator=None, threshold: float = 0.50) -> None:
    probs_raw = booster.predict(X)
    probs = (_apply_calibrator(calibrator, probs_raw)
             if calibrator is not None else probs_raw)
    preds = (probs >= threshold).astype(int)

    print(f"\n{'=' * 54}")
    print("EVALUATION METRICS (held-out test set)")
    print(f"{'=' * 54}")
    tag = " (calibrated)" if calibrator else ""
    print(f"  Accuracy   : {accuracy_score(y, preds):.3f}{tag}  @ threshold={threshold:.2f}")
    print(f"  Precision  : {precision_score(y, preds, zero_division=0):.3f}  "
          f"(YES bets correct)")
    print(f"  Recall     : {recall_score(y, preds, zero_division=0):.3f}  "
          f"(real YES events caught)")
    print(f"  F1         : {f1_score(y, preds, zero_division=0):.3f}")
    print(f"  AUC-ROC    : {roc_auc_score(y, probs):.3f}")
    print(f"  Brier      : {brier_score_loss(y, probs):.4f}")
    print(f"  Log-loss   : {log_loss(y, probs):.4f}")

    cm = confusion_matrix(y, preds)
    print(f"\n  Confusion matrix:")
    print(f"                  Pred NO   Pred YES")
    print(f"  Actual NO  :    {cm[0][0]:>6}     {cm[0][1]:>6}   <- false positives")
    print(f"  Actual YES :    {cm[1][0]:>6}     {cm[1][1]:>6}   <- true positives")


def _print_calibration(y_true, probs_raw, probs_cal) -> None:
    print(f"\n{'=' * 54}")
    print("CALIBRATION ANALYSIS")
    print(f"{'=' * 54}")
    brier_raw = brier_score_loss(y_true, probs_raw)
    brier_cal = brier_score_loss(y_true, probs_cal)
    print(f"  Brier (raw)        : {brier_raw:.4f}")
    print(f"  Brier (calibrated) : {brier_cal:.4f}")
    imp = (brier_raw - brier_cal) / brier_raw * 100
    print(f"  Improvement        : {imp:.1f}%" if imp > 0
          else "  Improvement        : none")

    n_bins = min(5, max(2, len(y_true) // 20))
    try:
        frac_pos, mean_pred = calibration_curve(
            y_true, probs_cal, n_bins=n_bins, strategy="uniform"
        )
        print(f"\n  {'Bin':>4}  {'Predicted':>10}  {'Actual':>10}  "
              f"{'Gap':>8}  {'Quality':>10}")
        print(f"  {'─' * 48}")
        for i, (actual, predicted) in enumerate(zip(frac_pos, mean_pred)):
            gap = abs(actual - predicted)
            q   = "good" if gap < 0.05 else "ok" if gap < 0.10 else "poor"
            print(f"  {i+1:>4}  {predicted:>10.3f}  {actual:>10.3f}  {gap:>8.3f}  {q:>10}")
        ece = float(np.mean(np.abs(frac_pos - mean_pred)))
        print(f"\n  ECE: {ece:.4f}  "
              f"({'good' if ece < 0.05 else 'ok' if ece < 0.10 else 'needs work'})")
    except ValueError:
        print("\n  (not enough data for calibration curve)")


_MAX_NO_BET_ODDS = 0.95   # only skip NO when YES market is near-certain (≥95¢)

def _print_pseudo_trade(
    y_true, probs_cal, kalshi_odds_col, warm_mask,
    min_edge: float = 0.05,
) -> None:
    """Flat $1/bet simulation on the test set. Baseline: 1¢ = 1 cent P&L."""
    bettable = (
        kalshi_odds_col.notna()
        & (kalshi_odds_col > 0.04)
        & (kalshi_odds_col < 0.96)
    )

    total_pnl = yes_pnl = no_pnl = warm_pnl = 0.0
    n_bets = n_yes = n_no = n_warm = n_correct = 0
    n_no_skipped = 0

    for i in range(len(probs_cal)):
        if not bettable.iloc[i]:
            continue
        is_warm = bool(
            warm_mask.iloc[i] if hasattr(warm_mask, "iloc") else warm_mask[i]
        )
        if not is_warm:
            continue  # never bet on cold-start rows — no real speaker signal

        p    = float(probs_cal[i])
        odds = float(kalshi_odds_col.iloc[i])
        y    = int(y_true[i])

        ev_yes = p - odds
        ev_no  = (1.0 - p) - (1.0 - odds)

        if ev_yes >= min_edge and ev_yes >= ev_no:
            side, entry = "YES", odds
            won = (y == 1)
        elif ev_no >= min_edge:
            if odds > _MAX_NO_BET_ODDS:
                n_no_skipped += 1
                continue
            side, entry = "NO", (1.0 - odds)
            won = (y == 0)
        else:
            continue

        pnl = (1.0 - entry) if won else -entry
        total_pnl += pnl
        n_bets    += 1
        n_warm    += 1  # all bets are warm (gated above)
        warm_pnl  += pnl
        if won:
            n_correct += 1
        if side == "YES":
            yes_pnl += pnl; n_yes += 1
        else:
            no_pnl  += pnl; n_no  += 1

    print(f"\n{'=' * 54}")
    print("PSEUDO-TRADE SIMULATION (flat $1/bet, test split)")
    print(f"{'=' * 54}")
    print(f"  Bettable rows  : {int(bettable.sum())}  (real kalshi_odds in (0.04, 0.96))")
    print(f"  Bets placed    : {n_bets}")
    if n_bets:
        roi = 100 * total_pnl / n_bets if n_bets else 0.0
        print(f"  Bet accuracy   : {n_correct/n_bets:.1%}")
        print(f"  YES bets P&L   : {yes_pnl*100:+.0f}¢  ({n_yes} bets)")
        print(f"  NO  bets P&L   : {no_pnl*100:+.0f}¢  ({n_no} bets, "
              f"{n_no_skipped} skipped odds>{_MAX_NO_BET_ODDS:.0%})")
        print(f"  Warm bets P&L  : {warm_pnl*100:+.0f}¢  ({n_warm} warm bets)")
        print(f"  Total P&L      : {total_pnl*100:+.0f}¢  "
              f"(${total_pnl:+.2f} on ${n_bets:.2f} wagered,  ROI {roi:+.1f}¢/bet)")
    else:
        print("  No bets placed — no rows with sufficient edge.")


def _print_importance(booster) -> None:
    importance = dict(zip(
        booster.feature_name(),
        booster.feature_importance(importance_type="gain"),
    ))
    top = sorted(importance.items(), key=lambda x: -x[1])
    print(f"\n{'=' * 54}")
    print("FEATURE IMPORTANCE (by gain)")
    print(f"{'=' * 54}")
    max_score = max(v for _, v in top) or 1
    for feat, score in top:
        bar = "█" * int(40 * score / max_score)
        print(f"  {feat:<30} {int(score):>6}  {bar}")


# ---------------------------------------------------------------------------
# Hyperparameter tuning (date-grouped CV, no test leakage)
# ---------------------------------------------------------------------------

def tune_hyperparams(n_trials: int = 50, verbose: bool = True) -> dict:
    """
    Random search over _TUNE_SPACE using date-grouped CV.
    Tunes only on the chronological training portion (excludes the last 20%)
    so the held-out test set never leaks into hyperparameter choice.
    Word priors are recomputed per fold to match the honest CV setup.
    """
    global _LGBM_PARAMS

    df, priors = build_training_dataset(verbose=verbose)

    dated_mask = df["event_date"].str.len() > 0
    if dated_mask.sum() >= 20:
        dated_sorted = sorted(
            df.index[dated_mask].tolist(),
            key=lambda i: df.loc[i, "event_date"],
        )
        split_at  = int(len(dated_sorted) * 0.80)
        test_idx  = set(dated_sorted[split_at:])
        tune_idx  = [i for i in df.index if i not in test_idx]
    else:
        tune_idx, _ = train_test_split(
            df.index.tolist(), test_size=0.20, random_state=42,
            stratify=df["did_say_word"].values,
        )

    df_tune = df.loc[tune_idx].reset_index(drop=True)
    n_yes_tune = int(df_tune["did_say_word"].sum())
    n_no_tune  = len(df_tune) - n_yes_tune
    spw_tune   = round(n_no_tune / max(n_yes_tune, 1), 3)

    if verbose:
        print(f"  Tuning on {len(df_tune)} rows  scale_pos_weight={spw_tune}")

    date_folds  = _date_grouped_folds(df_tune, n_splits=5)
    rng         = np.random.default_rng(123)
    best_score, best_params = -1.0, dict(_LGBM_PARAMS)

    if verbose:
        print(f"\nHyperparameter search: {n_trials} trials × 5-fold CV (date-grouped)")
        print(f"  Using gbdt for speed; DART will be used at final train() time.")
        print(f"{'=' * 54}")

    for trial in range(n_trials):
        # Use gbdt with early stopping for speed during search.
        # DART is used for the final model — it's too slow for 50×5 trials.
        params = {
            "boosting_type":     "gbdt",
            "objective":         "binary",
            "metric":            "binary_logloss",
            "extra_trees":       True,
            "max_bin":           63,
            "path_smooth":       1.0,
            "random_state":      42,
            "verbose":           -1,
            "scale_pos_weight":  spw_tune,
        }
        params["learning_rate"]     = float(rng.choice(_TUNE_SPACE["learning_rate"]))
        params["max_depth"]         = int(rng.choice(_TUNE_SPACE["max_depth"]))
        params["num_leaves"]        = int(rng.choice(_TUNE_SPACE["num_leaves"]))
        params["min_child_samples"] = int(rng.choice(_TUNE_SPACE["min_child_samples"]))
        params["feature_fraction"]  = float(rng.choice(_TUNE_SPACE["feature_fraction"]))
        params["bagging_fraction"]  = float(rng.choice(_TUNE_SPACE["bagging_fraction"]))
        params["bagging_freq"]      = 1
        params["reg_alpha"]         = float(rng.choice(_TUNE_SPACE["reg_alpha"]))
        params["reg_lambda"]        = float(rng.choice(_TUNE_SPACE["reg_lambda"]))
        params["min_split_gain"]    = float(rng.choice(_TUNE_SPACE["min_split_gain"]))
        params["num_leaves"]        = min(params["num_leaves"], 2 ** params["max_depth"])
        n_est_max                   = int(rng.choice(_TUNE_SPACE["n_estimators"]))

        fold_aucs: list[float] = []
        for trn_idx, val_idx in date_folds:
            if not val_idx:
                continue
            fold_priors = _compute_word_priors_from_arrays(
                df_tune.loc[trn_idx, "_word"].tolist(),
                df_tune.loc[trn_idx, "_event_type"].tolist(),
                df_tune.loc[trn_idx, "did_say_word"].tolist(),
            )
            X_fold_trn = _build_features_with_priors(df_tune.loc[trn_idx], fold_priors)
            X_fold_val = _build_features_with_priors(df_tune.loc[val_idx],  fold_priors)
            y_fold_trn = df_tune.loc[trn_idx, "did_say_word"].astype(int).values
            y_fold_val = df_tune.loc[val_idx,  "did_say_word"].astype(int).values
            w_fold_trn = df_tune.loc[trn_idx, "_weight"].astype(float).values

            dtrain = lgb.Dataset(
                X_fold_trn, label=y_fold_trn, weight=w_fold_trn,
                feature_name=FEATURES,
            )
            dval = lgb.Dataset(X_fold_val, label=y_fold_val, reference=dtrain)
            booster = lgb.train(
                params=params, train_set=dtrain,
                num_boost_round=n_est_max,
                valid_sets=[dval],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=30, verbose=False),
                    lgb.log_evaluation(period=0),
                ],
            )
            fold_aucs.append(roc_auc_score(y_fold_val, booster.predict(X_fold_val)))

        if not fold_aucs:
            continue
        mean_auc = float(np.mean(fold_aucs))
        if mean_auc > best_score:
            best_score  = mean_auc
            best_params = dict(params)
            best_params["n_estimators"] = n_est_max
            if verbose:
                print(f"  Trial {trial+1:>3}/{n_trials}: AUC={mean_auc:.4f} ★ "
                      f"(lr={params['learning_rate']}, "
                      f"depth={params['max_depth']}, "
                      f"leaves={params['num_leaves']}, "
                      f"n_est={n_est_max})")
        elif verbose and (trial + 1) % 10 == 0:
            print(f"  Trial {trial+1:>3}/{n_trials}: AUC={mean_auc:.4f}  "
                  f"(best: {best_score:.4f})")

    if verbose:
        print(f"\n  Best AUC: {best_score:.4f}")
        for k in sorted(best_params):
            if k in ("objective", "metric", "verbose", "random_state"):
                continue
            print(f"    {k:<22} = {best_params[k]}")

    _LGBM_PARAMS.update(best_params)
    return best_params


# ---------------------------------------------------------------------------
# Load / inference
# ---------------------------------------------------------------------------

_booster_cache: Optional[lgb.Booster] = None
_ensemble_cache: Optional[list[lgb.Booster]] = None
_calibrator_cache = None


def load_model() -> Optional[lgb.Booster]:
    if not MODEL_PATH.exists():
        return None
    return lgb.Booster(model_file=str(MODEL_PATH))


def load_ensemble() -> list[lgb.Booster]:
    out: list[lgb.Booster] = []
    if MODEL_PATH.exists():
        out.append(lgb.Booster(model_file=str(MODEL_PATH)))
    for i in range(1, 10):
        p = MODEL_PATH.with_name(f"kalshi_model_seed_{i}.lgb")
        if p.exists():
            out.append(lgb.Booster(model_file=str(p)))
    return out


def _get_booster() -> lgb.Booster:
    global _booster_cache
    if _booster_cache is None:
        b = load_model()
        if b is None:
            print("[kalshi_model] no saved model — training now ...")
            b = train(save=True, verbose=True)
        _booster_cache = b
    return _booster_cache


def _get_ensemble() -> list[lgb.Booster]:
    global _ensemble_cache
    if _ensemble_cache is None:
        ens = load_ensemble()
        if not ens:
            print("[kalshi_model] no saved model — training now ...")
            train(save=True, verbose=True)
            ens = load_ensemble()
        _ensemble_cache = ens
    return _ensemble_cache


def _get_calibrator():
    global _calibrator_cache
    if _calibrator_cache is None:
        _calibrator_cache = _load_calibrator()
    return _calibrator_cache


def predict_proba(
    speaker: str,
    word: str,
    event_type: str = "",
    kalshi_odds: float = 0.5,
    news_articles: Optional[list[dict]] = None,
    event_title: str = "",
) -> float:
    """
    Return P(speaker says word in this event) in [0, 1].

    Pipeline:
      1. Look up speaker profile features from DB
      2. Aggregate news relevancy features
      3. Look up word priors
      4. Run through LightGBM
      5. Apply isotonic calibration
      6. Apply context-aware veto gate for off-topic words
      7. Blend with hit_rate_lifetime when real data is scarce
    """
    profiles = db.get_cached_profile(speaker, word=word, event_type=event_type)
    if not profiles:
        profiles = db.get_cached_profile(speaker, word=word)
    if profiles and profiles[0]["n_samples_lifetime"] > 0:
        p = profiles[0]
        hit_rate_lifetime  = float(p["hit_rate_lifetime"] or 0.5)
        hit_rate_recent    = float(p["hit_rate_recent"]   or hit_rate_lifetime)
        momentum           = float(p["momentum"]          or 0.0)
        avg_freq           = float(p["avg_freq"]          or 1.0)
        recency            = float(p["recency"]           or 0.5)
        n_samples_lifetime = int(p["n_samples_lifetime"]  or 0)
    else:
        hit_rate_lifetime  = 0.5
        hit_rate_recent    = 0.5
        momentum           = 0.0
        avg_freq           = 1.0
        recency            = 0.5
        n_samples_lifetime = 0

    priors   = _get_word_priors()
    wg, we   = _lookup_priors(word, event_type, priors)
    et_prior = _lookup_event_type_prior(event_type, priors)
    cred     = _credibility(hit_rate_lifetime, n_samples_lifetime, et_prior)

    # Model is trained on warm rows only (n_samples_lifetime ≥ 3).
    # For cold-start words, return the event_type prior — we have no
    # reliable speaker signal and would never bet on these anyway.
    if n_samples_lifetime < 3:
        return round(et_prior, 4)

    ko = _mask_settlement_odds(kalshi_odds)
    if np.isnan(ko):
        ko = kalshi_odds

    tm_score_spacy = topic_match.compute_match_safe(event_title, word)
    tm_score_gate  = topic_match.compute_match_transformer(event_title, word)

    # Aggregate news relevancy features from provided articles (NaN if none)
    rel_max_val = rel_mean_val = rel_count_hi_val = rel_n_val = np.nan
    if news_articles:
        scores = [float(a.get("relevance_score", 0.0)) for a in news_articles]
        if scores:
            rel_max_val      = float(max(scores))
            rel_mean_val     = float(sum(scores) / len(scores))
            rel_count_hi_val = float(sum(1 for s in scores if s >= 0.5))
            rel_n_val        = float(len(scores))

    X = pd.DataFrame([{
        "hit_rate_lifetime":           hit_rate_lifetime,
        "hit_rate_recent":             hit_rate_recent,
        "momentum":                    momentum,
        "avg_freq":                    avg_freq,
        "recency":                     recency,
        "n_samples_lifetime":          n_samples_lifetime,
        "kalshi_odds":                 ko,
        "topic_match":                 tm_score_spacy,
        "hit_rate_word_global":        wg,
        "hit_rate_word_in_event_type": we,
        "hit_rate_credibility":        cred,
        "event_type_prior":            et_prior,
        "rel_max":                     rel_max_val,
        "rel_mean":                    rel_mean_val,
        "rel_count_hi":                rel_count_hi_val,
        "rel_n":                       rel_n_val,
    }], columns=FEATURES).astype(float)

    ensemble  = _get_ensemble()
    lgbm_prob = float(np.mean([b.predict(X)[0] for b in ensemble]))

    calibrator = _get_calibrator()
    if calibrator is not None:
        lgbm_prob = float(_apply_calibrator(calibrator, np.array([lgbm_prob]))[0])

    lgbm_prob = float(_post_process_probs(
        np.array([lgbm_prob]),
        np.array([kalshi_odds]),
    )[0])

    event_category = topic_match.classify_event(event_title) if event_title else "domestic_political"
    is_specialized = event_category in ("foreign_diplomatic", "ceremonial")

    if event_title and lgbm_prob > kalshi_odds:
        if is_specialized:
            if kalshi_odds < 0.25:
                yes_gate = 0.12
            elif kalshi_odds < 0.40:
                yes_gate = 0.18
            else:
                yes_gate = 0.05
        else:
            yes_gate = 0.05

        if tm_score_gate < yes_gate:
            lgbm_prob = kalshi_odds * 0.95

    hard_threshold = 0.05 if is_specialized else 0.03
    if event_title and tm_score_gate < hard_threshold and lgbm_prob > 0.55:
        lgbm_prob = 0.50

    n_real = _count_real_rows()
    if n_real >= _MIN_REAL_ROWS:
        return round(lgbm_prob, 4)

    alpha   = n_real / _MIN_REAL_ROWS
    blended = alpha * lgbm_prob + (1 - alpha) * hit_rate_lifetime
    return round(blended, 4)


def _count_real_rows() -> int:
    with db._connect() as conn:
        n_td = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        n_tl = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE outcome IN ('win','loss')"
        ).fetchone()[0]
    return n_td + n_tl


def retrain() -> lgb.Booster:
    global _booster_cache, _calibrator_cache, _word_priors_cache
    booster = train(save=True, verbose=False)
    _booster_cache     = booster
    _calibrator_cache  = _load_calibrator()
    _word_priors_cache = _load_word_priors()
    return booster


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_train() -> None:
    print("Training LightGBM model ...\n")
    train(save=True, verbose=True)


def _cli_eval() -> None:
    booster = load_model()
    if booster is None:
        print("No model found. Run --train first.")
        return
    calibrator = _load_calibrator()
    df, priors = build_training_dataset(verbose=True)

    dated_mask = df["event_date"].str.len() > 0
    if dated_mask.sum() >= 20:
        dated_sorted = sorted(
            df.index[dated_mask].tolist(),
            key=lambda i: df.loc[i, "event_date"],
        )
        split_at  = int(len(dated_sorted) * 0.80)
        test_idx  = list(dated_sorted[split_at:])
        train_idx = [i for i in df.index if i not in set(test_idx)]
    else:
        train_idx, test_idx = train_test_split(
            df.index.tolist(), test_size=0.20, random_state=42,
            stratify=df["did_say_word"].values,
        )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    test_priors = _compute_word_priors_from_arrays(
        df_train["_word"].tolist(),
        df_train["_event_type"].tolist(),
        df_train["did_say_word"].tolist(),
    )
    X_test = _build_features_with_priors(df_test, test_priors)
    y_test = df_test["did_say_word"].astype(int)

    test_probs_raw = booster.predict(X_test)
    test_probs_cal = (_apply_calibrator(calibrator, test_probs_raw)
                      if calibrator is not None else test_probs_raw)
    _print_metrics(booster, X_test, y_test, calibrator)
    _print_calibration(y_test.values, test_probs_raw, test_probs_cal)
    _print_importance(booster)
    print(f"\n  Optimal threshold : {_find_optimal_threshold(y_test.values, test_probs_cal):.3f}")


def _cli_info() -> None:
    n_real = _count_real_rows()
    with db._connect() as conn:
        n_prof = conn.execute(
            "SELECT COUNT(*) FROM speaker_profiles WHERE n_samples_lifetime >= 2"
        ).fetchone()[0]
    print(f"Real labeled rows  : {n_real}  (training_data + resolved trade_log)")
    print(f"Profile seeds      : {n_prof}  (speaker_profiles with >= 2 samples)")
    print(f"Model file exists  : {MODEL_PATH.exists()}  ({MODEL_PATH})")
    print(f"Calibrator exists  : {CALIBRATOR_PATH.exists()}  ({CALIBRATOR_PATH})")
    print(f"Word priors exist  : {WORD_PRIORS_PATH.exists()}  ({WORD_PRIORS_PATH})")
    print(f"Blend threshold    : {_MIN_REAL_ROWS} real rows -> full LightGBM trust")
    if MODEL_PATH.exists():
        _print_importance(load_model())


def _cli_tune() -> None:
    print("Hyperparameter search ...\n")
    tune_hyperparams(n_trials=50, verbose=True)
    print("\nRetraining with best params ...\n")
    train(save=True, verbose=True)


def _cli_calibrate() -> None:
    booster = load_model()
    if booster is None:
        print("No model found. Run --train first.")
        return
    calibrator = _load_calibrator()
    df, priors = build_training_dataset(verbose=True)

    dated_mask = df["event_date"].str.len() > 0
    if dated_mask.sum() >= 20:
        dated_sorted = sorted(
            df.index[dated_mask].tolist(),
            key=lambda i: df.loc[i, "event_date"],
        )
        split_at  = int(len(dated_sorted) * 0.80)
        test_idx  = list(dated_sorted[split_at:])
        train_idx = [i for i in df.index if i not in set(test_idx)]
    else:
        train_idx, test_idx = train_test_split(
            df.index.tolist(), test_size=0.20, random_state=42,
            stratify=df["did_say_word"].values,
        )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test  = df.loc[test_idx].reset_index(drop=True)

    test_priors = _compute_word_priors_from_arrays(
        df_train["_word"].tolist(),
        df_train["_event_type"].tolist(),
        df_train["did_say_word"].tolist(),
    )
    X_test = _build_features_with_priors(df_test, test_priors)
    y_test = df_test["did_say_word"].astype(int)

    test_probs_raw = booster.predict(X_test)
    test_probs_cal = (_apply_calibrator(calibrator, test_probs_raw)
                      if calibrator is not None else test_probs_raw)
    _print_calibration(y_test.values, test_probs_raw, test_probs_cal)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--train",     action="store_true",
                   help="5-fold CV train + calibrate + save")
    g.add_argument("--eval",      action="store_true",
                   help="Evaluate on held-out split")
    g.add_argument("--info",      action="store_true",
                   help="Dataset stats + feature importance")
    g.add_argument("--tune",      action="store_true",
                   help="Hyperparameter search (50 trials) + retrain")
    g.add_argument("--calibrate", action="store_true",
                   help="Calibration analysis only")
    args = p.parse_args()

    if args.train:
        _cli_train()
    elif args.eval:
        _cli_eval()
    elif args.info:
        _cli_info()
    elif args.tune:
        _cli_tune()
    elif args.calibrate:
        _cli_calibrate()
