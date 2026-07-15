"""
kalshi_model.py
---------------
"""

from __future__ import annotations

import argparse
import math
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

# topic_match lives at repo root in our layout; AI-Futures-Trader keeps its
# copy alongside kalshi_model.py in pipeline/_5_probability/ — try both so
# this module works dropped into either repo.
try:
    import topic_match
except ImportError:
    from pipeline._5_probability import topic_match  # type: ignore[no-redef]


def _import_news_scraper():
    """news_scraper is a top-level module in our layout, a `scrapers` package
    in AI-Futures-Trader — try both so this module works in either repo."""
    try:
        import news_scraper as _ns
        return _ns
    except ImportError:
        from scrapers import news_scraper as _ns
        return _ns

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MODEL_PATH       = Path(__file__).parent / "kalshi_model.lgb"
CALIBRATOR_PATH  = Path(__file__).parent / "kalshi_calibrator.pkl"
WORD_PRIORS_PATH = Path(__file__).parent / "kalshi_word_priors.pkl"
LR_MODEL_PATH    = Path(__file__).parent / "kalshi_lr_model.pkl"
WORD_EMB_CACHE_PATH = Path(__file__).parent / "word_embeddings_cache.pkl"

# Holdout cutoff — events on or after this date NEVER enter training.
HOLDOUT_CUTOFF = "2026-03-01"

# Bayesian shrinkage for word priors: alpha = n / (n + K).
_WORD_PRIOR_K_DEFAULT = 15.0

# Minimum real rows before we trust the model over the simple hit_rate fallback.
_MIN_REAL_ROWS = 50


def _speaker_k(n_obs: int) -> float:
    """Tiered K for hit_rate_credibility shrinkage (LF-058).
    Low-obs speakers get K=5 so their small sample is trusted faster."""
    if n_obs > 500:
        return 25.0
    if n_obs >= 50:
        return 15.0
    return 5.0

# Features used for training and inference.
FEATURES = [
    "hit_rate_lifetime",
    "hit_rate_recent",
    "momentum",
    "avg_freq",
    "recency",
    "n_samples_lifetime",
    "kalshi_odds",                   # NaN for settlement rows (≤0.04 or ≥0.96)
    "hit_rate_word_global",          # P(YES | word, all speakers) — shrunk
    "hit_rate_word_in_event_type",   # P(YES | word, event_type)   — shrunk
    "hit_rate_speaker_event_type",   # P(YES | speaker, word, event_type) — shrunk
    "hit_rate_credibility",          # speaker hit_rate shrunk toward event_type prior (tiered K)
    "event_type_prior",              # P(YES | event_type) baseline rate
    "word_rank",                     # rank by avg_freq in speaker vocab (1=most frequent)
    "market_vs_history",             # kalshi_odds - hit_rate_lifetime (NaN when no market price)
    "market_vs_word_prior",          # kalshi_odds - hit_rate_word_global (NaN when no market price)
    "days_since_last_event",         # days since speaker's previous event (NaN for first)
    "events_in_last_30d",            # distinct events in 30-day window before this event
    "topic_match",                   # transformer relevancy to event topic (NaN if not fetched)
    "rel_max",                       # news: max relevancy score (NaN if not fetched)
    "rel_mean",                      # news: mean relevancy score
    "rel_top3_mean",                 # news: mean of top-3 relevancy scores (94% NaN → learned NaN direction helps calibration)
    "rel_count_hi",                  # news: count of high-relevancy articles
    "rel_n",                         # news: total articles fetched for this word
    # Tried and rejected (see git history / session notes): word_semantic_proximity,
    # ko_velocity_24h/48h, news_decay_score, news_cooccur_rate, news_velocity,
    # news_title_polarity, news_tone_mean. Retested news_* twice, including
    # after improving GDELT retry patience (coverage only reached ~5-6%,
    # still 0 gain both times) — genuine data-availability ceiling for these
    # niche political mention markets, not a rate-limiting artifact. None
    # improved holdout Brier/AUC/accuracy vs this 23-feature baseline.
    # Feature-computation code stays available in the module (predict_proba's
    # optional ticker/news_articles params, the backfill script, db columns)
    # in case future data density makes them worth revisiting.
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
    "max_depth":         4,         # LF-056
    "learning_rate":     0.08,      # LF-056
    "min_child_samples": 30,
    "feature_fraction":  0.80,
    "bagging_fraction":  0.80,
    "bagging_freq":      1,
    "reg_alpha":         0.5,
    "reg_lambda":        0.5,       # LF-056 (was 1.5)
    "min_split_gain":    0.10,
    "path_smooth":       0.5,       # LF-056 (was 2.0)
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
# Half-life of 90 days: events from 3 months ago get ~50% weight.
_RECENCY_HALF_LIFE_DAYS = 90.0


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

_ENSEMBLE_SEEDS: list[int] = [42, 7, 13, 99, 2024, 1337, 17, 31, 55, 88, 144, 233]  # LF-007: 12 seeds


# ---------------------------------------------------------------------------
# Word priors — cross-speaker P(YES | word) and P(YES | word, event_type)
# ---------------------------------------------------------------------------

def _compute_word_priors() -> dict:
    """
    Compute Bayesian-shrunk word priors from training_data (pre-cutoff only).
    Also computes word_freq_rank (LF-005), speaker_event_dates (LF-014),
    and speaker_k tiers (LF-058). All derived from label-free columns —
    safe to inject into CV fold priors without label leakage.
    """
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT word, event_type, did_say_word, speaker, avg_freq, event_ticker, event_date "
            "FROM training_data "
            "WHERE event_date IS NULL OR event_date = '' OR event_date < ?",
            (HOLDOUT_CUTOFF,),
        ).fetchall()

    g_yes: dict = defaultdict(float)
    g_n:   dict = defaultdict(float)
    e_yes: dict = defaultdict(float)
    e_n:   dict = defaultdict(float)
    et_yes: dict = defaultdict(float)
    et_n:   dict = defaultdict(float)
    total_yes = total_n = 0.0

    sw_freq: dict = defaultdict(list)
    sw_outcomes: dict = defaultdict(list)
    sp_event_dates_set: dict = defaultdict(set)
    sp_n_total: dict = defaultdict(int)

    for row in rows:
        word  = row[0] or ""
        etype = row[1] or ""
        y     = float(row[2] or 0)
        sp    = row[3] or ""
        af    = float(row[4] or 0.0)
        edate = (row[6] or "")[:10]

        g_yes[word]          += y;  g_n[word]          += 1
        e_yes[(word, etype)] += y;  e_n[(word, etype)] += 1
        et_yes[etype]        += y;  et_n[etype]        += 1
        total_yes += y;  total_n += 1
        sw_freq[(sp, word)].append(af)
        sw_outcomes[(sp, word)].append(y)
        sp_n_total[sp] += 1
        if edate:
            sp_event_dates_set[sp].add(edate)

    global_prior = total_yes / total_n if total_n > 0 else 0.5

    word_global: dict[str, float] = {}
    for word in g_n:
        n     = g_n[word]
        raw   = g_yes[word] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K_DEFAULT)
        word_global[word] = alpha * raw + (1 - alpha) * global_prior

    word_etype: dict[tuple, float] = {}
    for (word, etype) in e_n:
        n   = e_n[(word, etype)]
        g   = word_global.get(word, global_prior)
        raw = e_yes[(word, etype)] / n if n > 0 else g
        alpha = n / (n + _WORD_PRIOR_K_DEFAULT)
        word_etype[(word, etype)] = alpha * raw + (1 - alpha) * g

    event_type_prior: dict[str, float] = {}
    for etype in et_n:
        n     = et_n[etype]
        raw   = et_yes[etype] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K_DEFAULT)
        event_type_prior[etype] = alpha * raw + (1 - alpha) * global_prior

    # LF-005: word_freq_rank — rank by avg_freq within speaker vocabulary
    sp_word_avg: dict = {
        (sp, w): sum(fs) / len(fs) for (sp, w), fs in sw_freq.items()
    }
    _sp_buckets: dict = defaultdict(list)
    for (sp, w), af in sp_word_avg.items():
        _sp_buckets[sp].append((af, w))
    word_freq_rank: dict = {}
    for sp, items in _sp_buckets.items():
        items.sort(key=lambda x: -x[0])
        for rank, (_, w) in enumerate(items, start=1):
            word_freq_rank[(sp, w)] = rank

    # LF-014: speaker_event_dates — sorted unique event dates per speaker
    speaker_event_dates: dict = {
        sp: sorted(dates) for sp, dates in sp_event_dates_set.items()
    }

    # LF-058: speaker_k — tiered K based on total obs count
    speaker_k: dict = {sp: _speaker_k(n) for sp, n in sp_n_total.items()}

    # word_variance — Var(did_say_word) per (speaker, word); 0.25 = max uncertainty default
    word_variance: dict[tuple, float] = {
        k: float(np.var(ys)) if len(ys) >= 2 else 0.25
        for k, ys in sw_outcomes.items()
    }

    return {
        "word_global":         word_global,
        "word_etype":          word_etype,
        "global_prior":        global_prior,
        "event_type_prior":    event_type_prior,
        "word_freq_rank":      word_freq_rank,
        "speaker_event_dates": speaker_event_dates,
        "speaker_k":           speaker_k,
        "word_variance":       word_variance,
    }


def _compute_word_priors_from_arrays(
    words: list[str],
    event_types: list[str],
    labels,
    speakers: Optional[list[str]] = None,
    event_dates: Optional[list[str]] = None,
    avg_freqs: Optional[list[float]] = None,
) -> dict:
    """
    Compute Bayesian-shrunk word priors from in-memory arrays (fold-local).
    Used inside CV folds so each val fold's labels never contaminate the
    prior-dependent features for that fold's validation rows.

    word_freq_rank, speaker_event_dates, and speaker_k are label-free;
    they are injected from the global priors dict by the caller after this returns.
    """
    g_yes: dict = defaultdict(float)
    g_n:   dict = defaultdict(float)
    e_yes: dict = defaultdict(float)
    e_n:   dict = defaultdict(float)
    et_yes: dict = defaultdict(float)
    et_n:   dict = defaultdict(float)
    total_yes = total_n = 0.0
    sp_n_total: dict = defaultdict(int)
    sw_outcomes_fold: dict = defaultdict(list)

    for i, (w, et, y) in enumerate(zip(words, event_types, labels)):
        w  = (w  or "")
        et = (et or "")
        y  = float(y)
        sp = (speakers[i] if speakers else "") or ""
        g_yes[w]       += y;  g_n[w]       += 1
        e_yes[(w, et)] += y;  e_n[(w, et)] += 1
        et_yes[et]     += y;  et_n[et]     += 1
        total_yes += y;  total_n += 1
        sp_n_total[sp] += 1
        sw_outcomes_fold[(sp, w)].append(y)

    global_prior = total_yes / total_n if total_n > 0 else 0.5

    word_global: dict[str, float] = {}
    for word in g_n:
        n     = g_n[word]
        raw   = g_yes[word] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K_DEFAULT)
        word_global[word] = alpha * raw + (1 - alpha) * global_prior

    word_etype: dict[tuple, float] = {}
    for (word, etype) in e_n:
        n   = e_n[(word, etype)]
        g   = word_global.get(word, global_prior)
        raw = e_yes[(word, etype)] / n if n > 0 else g
        alpha = n / (n + _WORD_PRIOR_K_DEFAULT)
        word_etype[(word, etype)] = alpha * raw + (1 - alpha) * g

    event_type_prior: dict[str, float] = {}
    for etype in et_n:
        n     = et_n[etype]
        raw   = et_yes[etype] / n if n > 0 else global_prior
        alpha = n / (n + _WORD_PRIOR_K_DEFAULT)
        event_type_prior[etype] = alpha * raw + (1 - alpha) * global_prior

    speaker_k: dict = {sp: _speaker_k(n) for sp, n in sp_n_total.items()}

    word_variance_fold: dict = {
        k: float(np.var(ys)) if len(ys) >= 2 else 0.25
        for k, ys in sw_outcomes_fold.items()
    }

    return {
        "word_global":      word_global,
        "word_etype":       word_etype,
        "global_prior":     global_prior,
        "event_type_prior": event_type_prior,
        "speaker_k":        speaker_k,
        "word_variance":    word_variance_fold,
        # word_freq_rank and speaker_event_dates injected from global priors by caller
    }


def _lookup_speaker_et_prior(speaker: str, word: str, event_type: str, priors: dict) -> float:
    swet = priors.get("speaker_word_etype", {})
    key  = (speaker or "", word or "", event_type or "")
    if key in swet:
        return swet[key]
    # Fallback: word+event_type prior, then word global, then global
    return priors.get("word_etype", {}).get(
        (word or "", event_type or ""),
        priors.get("word_global", {}).get(word or "",
        priors.get("global_prior", 0.5)))


def _build_features_with_priors(df_subset: pd.DataFrame, priors: dict) -> pd.DataFrame:
    """
    Build the FEATURES matrix for df_subset using the given priors.
    df_subset must have _word, _event_type, _speaker, and event_date columns.
    """
    gp       = priors["global_prior"]
    words    = list(df_subset["_word"].fillna(""))
    etypes   = list(df_subset["_event_type"].fillna(""))
    speakers = (
        list(df_subset["_speaker"].fillna(""))
        if "_speaker" in df_subset.columns else [""] * len(df_subset)
    )
    edates = (
        list(df_subset["event_date"].fillna(""))
        if "event_date" in df_subset.columns else [""] * len(df_subset)
    )

    wg_vals = [priors["word_global"].get(w, gp) for w in words]
    we_vals = [
        priors["word_etype"].get((w, et), priors["word_global"].get(w, gp))
        for w, et in zip(words, etypes)
    ]
    et_prior_vals = [priors["event_type_prior"].get(et, gp) for et in etypes]

    hl_arr = df_subset["hit_rate_lifetime"].values.astype(float)
    n_arr  = df_subset["n_samples_lifetime"].values.astype(float)

    # LF-058: speaker-tiered K for credibility
    sp_k_map = priors.get("speaker_k", {})
    cred_vals = [
        _credibility(float(hl), float(n), float(etp),
                     K=sp_k_map.get(sp, _WORD_PRIOR_K_DEFAULT))
        for hl, n, etp, sp in zip(hl_arr, n_arr, et_prior_vals, speakers)
    ]

    ko_arr = df_subset["kalshi_odds"].values.astype(float)

    # word_rank — rank by avg_freq in speaker vocab
    wfr_map  = priors.get("word_freq_rank", {})
    wfr_vals = [float(wfr_map.get((sp, w), 999)) for sp, w in zip(speakers, words)]

    # hit_rate_speaker_event_type — (speaker, word, event_type) shrunk prior
    swet_vals = [_lookup_speaker_et_prior(sp, w, et, priors)
                 for sp, w, et in zip(speakers, words, etypes)]

    # market_vs_history and market_vs_word_prior — NaN when no market price
    mvh_vals = [float(ko - hl) if not np.isnan(ko) else float("nan")
                for ko, hl in zip(ko_arr, hl_arr)]
    mvw_vals = [float(ko - wg) if not np.isnan(ko) else float("nan")
                for ko, wg in zip(ko_arr, wg_vals)]

    # events_in_last_30d + days_since_last_event via _speaker_activity (30-day window)
    sp_ed_map = priors.get("speaker_event_dates", {})
    act_vals  = [_speaker_activity(sp_ed_map.get(sp, []), ed)
                 for sp, ed in zip(speakers, edates)]
    days_since_last_evt_vals = [float("nan") if v[0] == 999.0 else float(v[0]) for v in act_vals]
    events_in_30d_vals       = [float(v[1]) for v in act_vals]

    def _news_col(col: str) -> np.ndarray:
        if col in df_subset.columns:
            vals = df_subset[col].values.astype(float)
            vals[vals == 0.0] = np.nan   # treat 0 as not-fetched
            return vals
        return np.full(len(df_subset), np.nan)

    def _raw_col(col: str) -> np.ndarray:
        """Like _news_col but without the 0->NaN treatment — these columns
        store real NULLs for missing data, and 0.0 is a legitimate value
        (e.g. neutral polarity, zero velocity)."""
        if col in df_subset.columns:
            return df_subset[col].values.astype(float)
        return np.full(len(df_subset), np.nan)

    wsp_map  = _get_word_semantic_proximity_map()
    wsp_vals = [float(wsp_map.get((sp, w), float("nan"))) for sp, w in zip(speakers, words)]

    return pd.DataFrame({
        "hit_rate_lifetime":             hl_arr,
        "hit_rate_recent":               df_subset["hit_rate_recent"].values.astype(float),
        "momentum":                      df_subset["momentum"].values.astype(float),
        "avg_freq":                      df_subset["avg_freq"].values.astype(float),
        "recency":                       df_subset["recency"].values.astype(float),
        "n_samples_lifetime":            n_arr,
        "kalshi_odds":                   ko_arr,
        "hit_rate_word_global":          wg_vals,
        "hit_rate_word_in_event_type":   we_vals,
        "hit_rate_speaker_event_type":   swet_vals,
        "hit_rate_credibility":          cred_vals,
        "event_type_prior":              et_prior_vals,
        "word_rank":                     wfr_vals,
        "market_vs_history":             mvh_vals,
        "market_vs_word_prior":          mvw_vals,
        "days_since_last_event":         days_since_last_evt_vals,
        "events_in_last_30d":            events_in_30d_vals,
        "topic_match":                   _news_col("topic_match"),
        "rel_max":                       _news_col("rel_max"),
        "rel_mean":                      _news_col("rel_mean"),
        "rel_top3_mean":                 _news_col("rel_top3_mean"),
        "rel_count_hi":                  _news_col("rel_count_hi"),
        "rel_n":                         _news_col("rel_n"),
        "news_decay_score":              _raw_col("news_decay_score"),
        "news_cooccur_rate":             _raw_col("news_cooccur_rate"),
        "news_velocity":                 _raw_col("news_velocity"),
        "news_title_polarity":           _raw_col("news_title_polarity"),
        "news_tone_mean":                _raw_col("news_tone_mean"),
        "word_semantic_proximity":       wsp_vals,
        "ko_velocity_24h":               _raw_col("ko_velocity_24h"),
        "ko_velocity_48h":               _raw_col("ko_velocity_48h"),
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
_speaker_activity_cache: Optional[dict] = None
_n_events_speaker_cache: Optional[dict] = None
_word_ranks_cache: Optional[dict] = None
_word_semantic_proximity_cache: Optional[dict] = None


def _get_speaker_activity_map() -> dict:
    global _speaker_activity_cache
    if _speaker_activity_cache is None:
        _speaker_activity_cache = _build_speaker_activity_map()
    return _speaker_activity_cache


def _compute_n_events_per_speaker() -> dict[str, int]:
    """Return {speaker: count_of_distinct_events} from pre-cutoff training_data."""
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT speaker, COUNT(DISTINCT event_ticker) as n "
            "FROM training_data "
            "WHERE event_date IS NULL OR event_date < ? "
            "GROUP BY speaker",
            (HOLDOUT_CUTOFF,),
        ).fetchall()
    return {r[0]: int(r[1]) for r in rows}


def _get_n_events_per_speaker() -> dict:
    global _n_events_speaker_cache
    if _n_events_speaker_cache is None:
        _n_events_speaker_cache = _compute_n_events_per_speaker()
    return _n_events_speaker_cache


def _get_word_ranks() -> dict:
    global _word_ranks_cache
    if _word_ranks_cache is None:
        _word_ranks_cache = _compute_word_ranks()
    return _word_ranks_cache


def _load_word_emb_cache() -> dict:
    """Load incremental word-embedding cache from disk (word -> np.ndarray)."""
    if WORD_EMB_CACHE_PATH.exists():
        try:
            with open(WORD_EMB_CACHE_PATH, "rb") as f:
                return pickle.load(f)
        except Exception:
            pass
    return {}


def _save_word_emb_cache(cache: dict) -> None:
    """Persist incremental word-embedding cache to disk."""
    try:
        with open(WORD_EMB_CACHE_PATH, "wb") as f:
            pickle.dump(cache, f)
    except Exception:
        pass


def _compute_word_semantic_proximity(cutoff: str = HOLDOUT_CUTOFF) -> dict:
    """
    Cosine similarity between a target word and its speaker's centroid embedding.

    For each speaker, embed their top-20 most-occurring words (by count across
    events) and average those embeddings -> speaker_centroid. Then for every
    (speaker, word) pair return cosine_sim(embed(word), speaker_centroid).

    Embeddings are cached in word_embeddings_cache.pkl — only new words are
    encoded on each call, since sentence-transformer encoding is slow relative
    to a retrain.

    Returns {} if sentence-transformers is unavailable (topic_match._get_st_model
    returns None) — callers fall back to NaN.
    """
    model = topic_match._get_st_model()
    if model is None:
        return {}

    with db._connect() as conn:
        rows = conn.execute(
            "SELECT speaker, word FROM training_data "
            "WHERE event_date IS NULL OR event_date < ?",
            (cutoff,),
        ).fetchall()

    from collections import Counter
    sp_word_counts: dict = defaultdict(Counter)
    sp_all_words:   dict = defaultdict(set)
    for speaker, word in rows:
        sp_word_counts[speaker][word] += 1
        sp_all_words[speaker].add(word)

    word_emb = _load_word_emb_cache()
    all_words = list({w for words in sp_all_words.values() for w in words})
    if not all_words:
        return {}

    new_words = [w for w in all_words if w not in word_emb]
    if new_words:
        new_embeddings = model.encode(new_words, convert_to_numpy=True,
                                       batch_size=256, show_progress_bar=False)
        for w, emb in zip(new_words, new_embeddings):
            word_emb[w] = emb
        _save_word_emb_cache(word_emb)

    result: dict = {}
    for speaker, wcount in sp_word_counts.items():
        top20 = [w for w, _ in wcount.most_common(20)]
        if not top20:
            continue
        centroid = np.mean([word_emb[w] for w in top20 if w in word_emb], axis=0)
        c_norm   = np.linalg.norm(centroid)
        if c_norm == 0:
            continue
        for word in sp_all_words[speaker]:
            if word not in word_emb:
                continue
            w_emb  = word_emb[word]
            w_norm = np.linalg.norm(w_emb)
            if w_norm == 0:
                result[(speaker, word)] = 0.0
            else:
                result[(speaker, word)] = float(np.dot(centroid, w_emb) / (c_norm * w_norm))
    return result


def _get_word_semantic_proximity_map() -> dict:
    global _word_semantic_proximity_cache
    if _word_semantic_proximity_cache is None:
        _word_semantic_proximity_cache = _compute_word_semantic_proximity()
    return _word_semantic_proximity_cache


def _word_semantic_proximity(speaker: str, word: str) -> float:
    """Lookup helper for predict_proba — NaN if not in the precomputed map."""
    return float(_get_word_semantic_proximity_map().get((speaker, word), float("nan")))


def _compute_ko_velocity(
    ticker: str, current_odds: float, ref_ts: Optional[int] = None,
) -> tuple[float, float]:
    """
    Price velocity: current_odds minus the YES price ~24h/48h earlier for the
    same market. ref_ts defaults to now (live inference); pass an explicit
    unix timestamp when backfilling historical rows. Returns (NaN, NaN) when
    the ticker is unknown or no candle data is available in that window.
    """
    if not ticker or np.isnan(current_odds):
        return float("nan"), float("nan")

    import kalshi_api as _ka
    if ref_ts is None:
        ref_ts = int(datetime.datetime.now(tz=datetime.timezone.utc).timestamp())

    try:
        p24 = _ka.get_price_at_ts(ticker, ref_ts - 24 * 3600)
        p48 = _ka.get_price_at_ts(ticker, ref_ts - 48 * 3600)
    except Exception:
        return float("nan"), float("nan")

    v24 = float(current_odds - p24) if p24 is not None else float("nan")
    v48 = float(current_odds - p48) if p48 is not None else float("nan")
    return v24, v48


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
                 K: float = _WORD_PRIOR_K_DEFAULT) -> float:
    n = max(0.0, float(n_samples))
    alpha = n / (n + K)
    return alpha * hit_rate + (1.0 - alpha) * et_prior


def _compute_activity_features(
    speaker: str, event_date: str, speaker_event_dates: dict
) -> tuple[float, float]:
    """
    Returns (days_since_last_event, events_per_90d) for a speaker at event_date.
    Uses only events strictly before event_date — no temporal leakage.
    LF-014.
    """
    if not event_date:
        return (30.0, 1.0)
    dates = speaker_event_dates.get(speaker, [])
    prior = [d for d in dates if d < event_date]
    if not prior:
        return (30.0, 1.0)
    try:
        cur  = datetime.date.fromisoformat(event_date[:10])
        last = datetime.date.fromisoformat(prior[-1])
        days_since = float(max(0, (cur - last).days))
        cutoff_90  = (cur - datetime.timedelta(days=90)).isoformat()
        events_90  = float(sum(1 for d in prior if d >= cutoff_90))
        return (days_since, max(events_90, 1.0))
    except ValueError:
        return (30.0, 1.0)


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


def _build_speaker_activity_map() -> dict[str, list[str]]:
    """
    Return {speaker: sorted list of unique event_dates (pre-cutoff)}.
    Used to compute days_since_last_event and events_in_last_30d per row.
    """
    import bisect
    with db._connect() as conn:
        rows = conn.execute(
            "SELECT DISTINCT speaker, event_date FROM training_data "
            "WHERE event_date IS NOT NULL AND event_date != '' AND event_date < ?",
            (HOLDOUT_CUTOFF,),
        ).fetchall()
    speaker_dates: dict[str, list[str]] = {}
    for speaker, date in rows:
        speaker_dates.setdefault(speaker, []).append(date)
    for sp in speaker_dates:
        speaker_dates[sp] = sorted(set(speaker_dates[sp]))
    return speaker_dates


def _speaker_activity(dates: list[str], event_date: str) -> tuple[float, int]:
    """
    Given a sorted list of event dates for a speaker and the current event date,
    return (days_since_last_event, events_in_last_30d).
    Only counts events STRICTLY BEFORE event_date to avoid leakage.
    """
    import bisect, datetime
    idx = bisect.bisect_left(dates, event_date)
    prev_dates = dates[:idx]  # strictly before
    if not prev_dates:
        return 999.0, 0
    try:
        ed = datetime.date.fromisoformat(event_date[:10])
        last_d = datetime.date.fromisoformat(prev_dates[-1][:10])
        days_since = float((ed - last_d).days)
        cutoff_30d = (ed - datetime.timedelta(days=30)).isoformat()
        n_last_30d = sum(1 for d in prev_dates if d >= cutoff_30d)
        return days_since, n_last_30d
    except (ValueError, IndexError):
        return 999.0, 0


def _rows_from_training_table(priors: dict, word_ranks: dict | None = None) -> list[dict]:
    rows = db.get_training_data()
    speaker_activity_map = _build_speaker_activity_map()
    n_evt_map = _compute_n_events_per_speaker()
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

        speaker = r.get("speaker", "")

        # word_rank from pre-computed ranks dict
        wrank = float(word_ranks.get((speaker, word), 999)) if word_ranks else 999.0

        # speaker activity features — NaN when no prior events (first event in DB)
        sp_dates = speaker_activity_map.get(speaker, [])
        if event_date and sp_dates:
            days_since, events_30d = _speaker_activity(sp_dates, event_date)
            ds_val = float("nan") if days_since == 999.0 else days_since
        else:
            ds_val, events_30d = float("nan"), 0

        n_evt_total = float(n_evt_map.get(speaker, 0))

        ko = _mask_settlement_odds(r.get("kalshi_odds"))
        mvh = float(ko - hl) if not np.isnan(ko) else float("nan")
        mvw = float(ko - wg) if not np.isnan(ko) else float("nan")

        out.append({
            # Stable features (no priors needed)
            "hit_rate_lifetime":              hl,
            "hit_rate_recent":                float(r.get("hit_rate_recent")   or 0.5),
            "momentum":                       float(r.get("momentum")          or 0.0),
            "avg_freq":                       float(r.get("avg_freq")          or 1.0),
            "recency":                        float(r.get("recency")           or 0.5),
            "n_samples_lifetime":             n_lifetime,
            "kalshi_odds":                    ko,
            # Prior-dependent features (recomputed per fold in CV)
            "hit_rate_word_global":           wg,
            "hit_rate_word_in_event_type":    we,
            "hit_rate_speaker_event_type":    et_prior,  # placeholder; rebuilt in _build_features_with_priors
            "hit_rate_credibility":           cred,
            "event_type_prior":               et_prior,
            "word_rank":                      wrank,
            "market_vs_history":              mvh,
            "market_vs_word_prior":           mvw,
            "days_since_last_event":          ds_val,
            "events_in_last_30d":             float(events_30d),
            # Context / news features — saved per row, passed through unchanged
            "topic_match":   float(r.get("topic_match")   or 0.0),
            "rel_max":       float(r.get("rel_max")       or 0.0),
            "rel_mean":      float(r.get("rel_mean")      or 0.0),
            "rel_top3_mean": float(r.get("rel_top3_mean") or 0.0),
            "rel_count_hi":  float(r.get("rel_count_hi")  or 0),
            "rel_n":         float(r.get("rel_n")         or 0),
            # Ported news/velocity features — real NULLs stay NaN (0 is a
            # legitimate value for these, unlike the rel_* columns above)
            "news_decay_score":    _to_nan(r.get("news_decay_score")),
            "news_cooccur_rate":   _to_nan(r.get("news_cooccur_rate")),
            "news_velocity":       _to_nan(r.get("news_velocity")),
            "news_title_polarity": _to_nan(r.get("news_title_polarity")),
            "news_tone_mean":      _to_nan(r.get("news_tone_mean")),
            "ko_velocity_24h":     _to_nan(r.get("ko_velocity_24h")),
            "ko_velocity_48h":     _to_nan(r.get("ko_velocity_48h")),
            # Labels / metadata
            "did_say_word": int(r["did_say_word"]),
            "event_date":   event_date,
            "_weight":      3.0 * _recency_weight(event_date),
            # Raw keys needed for per-fold prior recomputation
            "_word":        word,
            "_event_type":  et,
            "_speaker":     speaker,
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
        ko_tl      = _mask_settlement_odds(r.get("kalshi_odds"))
        n_evt_map  = _get_n_events_per_speaker()
        speaker    = r.get("speaker", "")

        out.append({
            "hit_rate_lifetime":           hl,
            "hit_rate_recent":             float(r.get("hit_rate_recent")   or 0.5),
            "momentum":                    float(r.get("momentum")          or 0.0),
            "avg_freq":                    float(r.get("avg_freq")          or 1.0),
            "recency":                     float(r.get("recency")           or 0.5),
            "n_samples_lifetime":          n_lifetime,
            "kalshi_odds":                 ko_tl,
            "hit_rate_word_global":        wg,
            "hit_rate_word_in_event_type": we,
            "hit_rate_speaker_event_type": et_prior,
            "hit_rate_credibility":        cred,
            "event_type_prior":            et_prior,
            "word_rank":                   float("nan"),
            "market_vs_history":           float(ko_tl - hl) if not np.isnan(ko_tl) else float("nan"),
            "market_vs_word_prior":        float(ko_tl - wg) if not np.isnan(ko_tl) else float("nan"),
            "days_since_last_event":       float("nan"),
            "events_in_last_30d":          float("nan"),
            # trade_log rows have no news context — use neutral defaults
            "topic_match":   0.0,
            "rel_max":       0.0,
            "rel_mean":      0.0,
            "rel_top3_mean": 0.0,
            "rel_count_hi":  0.0,
            "rel_n":         0.0,
            "news_decay_score":    float("nan"),
            "news_cooccur_rate":   float("nan"),
            "news_velocity":       float("nan"),
            "news_title_polarity": float("nan"),
            "news_tone_mean":      float("nan"),
            "ko_velocity_24h":     float("nan"),
            "ko_velocity_48h":     float("nan"),
            "did_say_word": did_say,
            "event_date":   "",
            "_weight":      5.0,
            "_word":        word,
            "_event_type":  et,
            "_speaker":     speaker,
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
        warm = int((df["n_samples_lifetime"] >= 2).sum())
        print(f"  Warm rows (n_samples_lifetime ≥ 2): {warm} / {n_real}")

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
    df = df[df["n_samples_lifetime"] >= 2].reset_index(drop=True)
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
        speakers=df_train["_speaker"].tolist() if "_speaker" in df_train.columns else None,
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
            speakers=df_train.loc[trn_idx, "_speaker"].tolist() if "_speaker" in df_train.columns else None,
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
        _MIN_CAL_IMPROVEMENT = 0.005   # require meaningful OOF improvement (Platt ~0.001 typical, rarely triggers)
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

    # ---- Logistic Regression ensemble member ----
    # Trained on the same X_train features. Blended equally with LGBM at inference.
    from sklearn.linear_model import LogisticRegression as _LR
    from sklearn.preprocessing import StandardScaler as _SS
    X_train_lr = _build_features_with_priors(df_train, priors)
    y_train_lr = df_train["did_say_word"].astype(int).values
    # Replace NaN with column median for LR (can't handle NaN)
    X_train_lr_filled = X_train_lr.copy()
    col_medians = {}
    for col in X_train_lr_filled.columns:
        med = X_train_lr_filled[col].median()
        fill_val = float(med) if not np.isnan(med) else 0.0
        col_medians[col] = fill_val
        X_train_lr_filled[col] = X_train_lr_filled[col].fillna(fill_val)
    lr_scaler = _SS()
    X_train_lr_scaled = lr_scaler.fit_transform(X_train_lr_filled.values)
    lr_model = _LR(C=1.0, max_iter=1000, random_state=42)
    lr_model.fit(X_train_lr_scaled, y_train_lr)
    if save:
        import pickle as _pkl
        with open(LR_MODEL_PATH, "wb") as _f:
            _pkl.dump({"model": lr_model, "scaler": lr_scaler,
                       "col_medians": col_medians}, _f)
        if verbose:
            print(f"  Saved LR model → {LR_MODEL_PATH}")

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
            df_test["n_samples_lifetime"] >= 2,
            min_edge=0.10,   # match run_pipeline live-trading default
            speakers_col=df_test["_speaker"] if "_speaker" in df_test.columns else None,
        )
        _print_importance(booster)
        print(f"\n  Ensemble size     : {len(boosters)}")
        print(f"  Threshold used    : {_optimal_threshold:.3f}  "
              f"(OOF, no test leakage)")

    return booster


# ---------------------------------------------------------------------------
# Calibration — Platt scaling (2-param sigmoid) on OOF predictions
# ---------------------------------------------------------------------------

def _fit_calibrator(oof_preds: np.ndarray, y_true: np.ndarray):
    """
    Fit Platt scaling: p_cal = sigmoid(a * logit(p) + b).
    2 parameters — can't overfit OOF the way isotonic regression does.
    a<1 compresses predictions toward 0.5 (fixes over-confidence).
    b<0 shifts predictions downward (fixes systematic over-prediction).
    Returns None if fewer than 30 samples.
    """
    from scipy.optimize import minimize
    if len(oof_preds) < 30:
        return None
    p = np.clip(oof_preds, 1e-7, 1 - 1e-7)
    logits = np.log(p) - np.log(1 - p)
    y = np.asarray(y_true, dtype=float)

    def nll(params):
        a, b = params
        s = a * logits + b
        log_p   = np.where(s >= 0, -np.log1p(np.exp(-s)), s - np.log1p(np.exp(s)))
        log_1mp = np.where(s >= 0, -s - np.log1p(np.exp(-s)), -np.log1p(np.exp(s)))
        return -np.mean(y * log_p + (1 - y) * log_1mp)

    res = minimize(nll, x0=[1.0, 0.0], method="Nelder-Mead",
                   options={"xatol": 1e-5, "fatol": 1e-6, "maxiter": 2000})
    a, b = float(res.x[0]), float(res.x[1])
    return ("platt", (a, b))


def _apply_calibrator(calibrator, probs: np.ndarray) -> np.ndarray:
    kind, model = calibrator
    if kind == "platt":
        a, b = model
        p = np.clip(probs, 1e-7, 1 - 1e-7)
        logits = np.log(p) - np.log(1 - p)
        return 1.0 / (1.0 + np.exp(-(a * logits + b)))
    return model.predict(probs)  # isotonic fallback


# Both blending and capping were found to HURT holdout calibration:
# raw LGBM Brier=0.1671 CalErr=0.0601 vs post-processed 0.1681/0.1095.
# The 0.65 cap blocked legitimate high-confidence YES bets.
# The 0.10 market blend shifted predictions toward over-priced markets.
_PROB_CLIP_HI: float = 0.99
_KALSHI_BLEND_W: float = 0.0


def _post_process_probs(
    probs: np.ndarray,
    kalshi_odds: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Clip only — blend is disabled (hurts holdout calibration)."""
    p = np.asarray(probs, dtype=float).copy()
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


_MAX_NO_BET_ODDS = 0.70   # skip NO bets when YES market > 70¢ (low payout, higher risk)

# Speakers where YES bets are blocked — model consistently over-predicts YES
# for these speakers based on holdout analysis. NO bets are still allowed.
_YES_BET_BLOCKED_SPEAKERS: set[str] = {"J.D. Vance", "Pete Hegseth"}

def _print_pseudo_trade(
    y_true, probs_cal, kalshi_odds_col, warm_mask,
    min_edge: float = 0.05,
    speakers_col=None,
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

        spk = ""
        if speakers_col is not None:
            spk = speakers_col.iloc[i] if hasattr(speakers_col, "iloc") else speakers_col[i]

        if ev_yes >= min_edge and ev_yes >= ev_no:
            if spk in _YES_BET_BLOCKED_SPEAKERS:
                # Downgrade to NO bet if it has edge, else skip
                if ev_no >= min_edge and odds <= _MAX_NO_BET_ODDS:
                    side, entry = "NO", (1.0 - odds)
                    won = (y == 0)
                else:
                    continue
            else:
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
                speakers=df_tune.loc[trn_idx, "_speaker"].tolist() if "_speaker" in df_tune.columns else None,
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


def _load_lr_model():
    """Load the saved logistic regression ensemble member (returns None if not found)."""
    if not LR_MODEL_PATH.exists():
        return None
    import pickle as _pkl
    with open(LR_MODEL_PATH, "rb") as f:
        return _pkl.load(f)


# ---------------------------------------------------------------------------
# News feature engineering — ported from AI-Futures-Trader
# (decay score, cooccurrence, velocity, title polarity, VADER tone)
# ---------------------------------------------------------------------------

_DECAY_LAMBDA     = math.log(2) / 3.0  # 3-day half-life
_NEWS_WINDOW_DAYS = 14

_POL_POS: frozenset = frozenset({
    "victory", "win", "wins", "won", "peace", "deal", "agreement",
    "growth", "record", "strong", "success", "breakthrough", "historic",
    "positive", "celebrate", "boom", "progress",
})
_POL_NEG: frozenset = frozenset({
    "crisis", "war", "collapse", "fail", "failure", "attack", "bomb",
    "disaster", "emergency", "scandal", "resign", "resigns", "threat",
    "shock", "slump", "conflict",
})


def _get_vader():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        return SentimentIntensityAnalyzer()
    except ImportError:
        return None


_VADER_SIA = _get_vader()

_NEWS_DEFAULTS: dict = {
    "news_has_any_news":   0,
    "news_decay_score":    None,
    "news_cooccur_rate":   None,
    "news_velocity":       None,
    "news_title_polarity": None,
    "news_tone_mean":      None,
}


def _to_nan(v):
    """None -> NaN, pass numbers through — for optional news feature values."""
    return float("nan") if v is None else float(v)


def _infer_art_age_days(pub, date_to, pos: int, n_total: int) -> int:
    """
    Estimate article age in days. GDELT seendate lags publication by 1-2 days,
    so dates near the event can be negative. Falls back to position-based
    estimate (position 0 = newest, position N-1 = oldest in DateDesc-sorted list).
    """
    if pub is not None:
        raw = (date_to.date() - pub.date()).days
        if 0 <= raw <= _NEWS_WINDOW_DAYS:
            return raw
    return int(pos / max(n_total - 1, 1) * _NEWS_WINDOW_DAYS)


def _compute_raw_news_features(
    articles: list[dict],
    speaker: str,
    word: str,
    date_to: Optional[datetime.datetime] = None,
) -> dict:
    """
    Compute decay/cooccurrence/velocity/polarity/tone news features from an
    in-memory article list. Title-only matching for count/decay/velocity,
    title+snippet for cooccurrence. All None when articles=[].
    """
    _ns = _import_news_scraper()

    if not articles:
        return dict(_NEWS_DEFAULTS)

    if date_to is None:
        date_to = datetime.datetime.now(tz=datetime.timezone.utc)

    articles_clean = [
        a for a in articles
        if _ns.classify_article_type(a.get("title", ""), a.get("source", ""))
        in ("news", "analysis")
    ]
    work = articles_clean if articles_clean else articles

    parts   = speaker.strip().split()
    surname = parts[-1].lower() if parts else speaker.lower()
    word_l  = word.lower()

    decay_score  = 0.0
    n_title_word = 0
    n_spkr = n_joint = 0
    n_title_3d = n_title_11d = 0
    n_total = len(work)

    for i, art in enumerate(work):
        title   = (art.get("title") or "").lower()
        snippet = (art.get("snippet") or "").lower()
        text    = title + (" " + snippet if snippet else "")
        in_word      = word_l in title
        in_word_text = word_l in text
        in_spkr_text = surname in text

        pub = art.get("published_at")
        if isinstance(pub, str):
            try:
                pub = datetime.datetime.fromisoformat(pub)
                if pub.tzinfo is None:
                    pub = pub.replace(tzinfo=datetime.timezone.utc)
            except ValueError:
                pub = None

        if in_word:
            n_title_word += 1
            age_days = _infer_art_age_days(pub, date_to, i, n_total)
            decay_score += math.exp(-_DECAY_LAMBDA * age_days)
            if age_days < 3:
                n_title_3d += 1
            elif age_days < 14:
                n_title_11d += 1

        if in_spkr_text:
            n_spkr += 1
            if in_word_text:
                n_joint += 1

    if n_title_word == 0:
        decay_score = 0.0

    cooccur_rate = (n_joint / n_spkr) if n_spkr > 0 else None

    rate_recent = n_title_3d / 3
    rate_older  = n_title_11d / 11
    if rate_older < 0.01 and rate_recent < 0.01:
        velocity = None
    else:
        velocity = max(-2.0, min(5.0, (rate_recent - rate_older) / max(rate_older, 0.1)))

    pol_scores = []
    for art in work:
        t = (art.get("title") or "").lower()
        if word_l not in t:
            continue
        twords = set(t.split())
        pos = len(twords & _POL_POS)
        neg = len(twords & _POL_NEG)
        if pos + neg > 0:
            pol_scores.append((pos - neg) / (pos + neg))
    title_polarity = sum(pol_scores) / len(pol_scores) if pol_scores else None

    tone_scores: list[float] = []
    if _VADER_SIA is not None:
        for art in work:
            raw_title = art.get("title") or ""
            if raw_title and word_l in raw_title.lower():
                tone_scores.append(_VADER_SIA.polarity_scores(raw_title)["compound"])
    tone_mean = sum(tone_scores) / len(tone_scores) if tone_scores else None

    return {
        "news_has_any_news":   1,
        "news_decay_score":    decay_score,
        "news_cooccur_rate":   cooccur_rate,
        "news_velocity":       velocity,
        "news_title_polarity": title_polarity,
        "news_tone_mean":      tone_mean,
    }


def predict_proba(
    speaker: str,
    word: str,
    event_type: str = "",
    kalshi_odds: float = 0.5,
    news_articles: Optional[list[dict]] = None,
    event_title: str = "",
    return_std: bool = False,
    ticker: str = "",
    event_ts: Optional[int] = None,
    news_word_relative_rank: float = float("nan"),
    ko_velocity_24h: float = float("nan"),
    ko_velocity_48h: float = float("nan"),
) -> float | tuple[float, float]:
    """
    Return P(speaker says word in this event) in [0, 1].

    Pipeline:
      1. Look up speaker profile features from DB
      2. Look up word priors
      3. Run through LightGBM
      4. Apply isotonic calibration
      5. Apply context-aware veto gate for off-topic words
      6. Blend with hit_rate_lifetime when real data is scarce

    If return_std=True, returns (prob, ensemble_std) where ensemble_std is
    the std-dev of the raw per-seed ensemble predictions (before LR blend,
    calibration, and veto gates) — a measure of how much the 10 LightGBM
    seeds disagree. 0.0 for cold-start/blended-fallback paths where no
    ensemble prediction was made.

    news_word_relative_rank / ko_velocity_24h / ko_velocity_48h: accepted for
    call-site compatibility with the AI-Futures-Trader pipeline (which passes
    these as kwargs), but not used — none of them survived a fair holdout
    test in this model (see FEATURES list comment) and aren't in FEATURES.
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

    # Model is trained on warm rows only (n_samples_lifetime ≥ 2).
    # For cold-start words, return the event_type prior — we have no
    # reliable speaker signal and would never bet on these anyway.
    if n_samples_lifetime < 2:
        return (round(et_prior, 4), 0.0) if return_std else round(et_prior, 4)

    ko = _mask_settlement_odds(kalshi_odds)
    if np.isnan(ko):
        ko = kalshi_odds

    tm_score_gate = topic_match.compute_match_transformer(event_title, word)

    # News relevancy features — aggregate from articles passed by the pipeline
    _ns = _import_news_scraper()
    if news_articles:
        nf = _ns.aggregate_relevancy_features(news_articles)
    else:
        nf = {"rel_max": 0.0, "rel_mean": 0.0, "rel_top3_mean": 0.0,
              "rel_count_hi": 0, "rel_n": 0}

    # Ported news features — decay score, cooccurrence, velocity, polarity, tone
    news_raw = _compute_raw_news_features(news_articles or [], speaker, word)

    # Price velocity — NaN unless a ticker is supplied (live inference / backfill)
    ko_vel_24h, ko_vel_48h = _compute_ko_velocity(ticker, ko, ref_ts=event_ts)

    # Speaker activity features (look up from DB)
    sp_act_map = _get_speaker_activity_map()
    sp_dates   = sp_act_map.get(speaker, [])
    import datetime as _dt
    today_str = _dt.date.today().isoformat()
    days_since_val, events_30d_val = _speaker_activity(sp_dates, today_str)
    ds_val_inf = float("nan") if days_since_val == 999.0 else float(days_since_val)

    # Interaction features
    mv_hist = float(ko - hit_rate_lifetime) if not np.isnan(ko) else float("nan")
    mv_word = float(ko - wg) if not np.isnan(ko) else float("nan")

    X = pd.DataFrame([{
        "hit_rate_lifetime":              hit_rate_lifetime,
        "hit_rate_recent":                hit_rate_recent,
        "momentum":                       momentum,
        "avg_freq":                       avg_freq,
        "recency":                        recency,
        "n_samples_lifetime":             n_samples_lifetime,
        "kalshi_odds":                    ko,
        "hit_rate_word_global":           wg,
        "hit_rate_word_in_event_type":    we,
        "hit_rate_speaker_event_type":    _lookup_speaker_et_prior(speaker, word, event_type, priors),
        "hit_rate_credibility":           cred,
        "event_type_prior":               et_prior,
        "word_rank":                      float(_get_word_ranks().get((speaker, word), float("nan"))),
        "market_vs_history":              mv_hist,
        "market_vs_word_prior":           mv_word,
        "days_since_last_event":          ds_val_inf,
        "events_in_last_30d":             float(events_30d_val),
        "topic_match":                    float(tm_score_gate),
        "rel_max":                        float(nf["rel_max"]),
        "rel_mean":                       float(nf["rel_mean"]),
        "rel_top3_mean":                  float(nf["rel_top3_mean"]),
        "rel_count_hi":                   float(nf["rel_count_hi"]),
        "rel_n":                          float(nf["rel_n"]),
        "news_decay_score":               _to_nan(news_raw["news_decay_score"]),
        "news_cooccur_rate":              _to_nan(news_raw["news_cooccur_rate"]),
        "news_velocity":                  _to_nan(news_raw["news_velocity"]),
        "news_title_polarity":            _to_nan(news_raw["news_title_polarity"]),
        "news_tone_mean":                 _to_nan(news_raw["news_tone_mean"]),
        "word_semantic_proximity":        _word_semantic_proximity(speaker, word),
        "ko_velocity_24h":                ko_vel_24h,
        "ko_velocity_48h":                ko_vel_48h,
    }], columns=FEATURES).astype(float)

    ensemble    = _get_ensemble()
    seed_preds  = np.array([b.predict(X)[0] for b in ensemble])
    lgbm_prob   = float(seed_preds.mean())
    ensemble_std = float(seed_preds.std())

    # Blend in LR ensemble member if available
    lr_bundle = _load_lr_model()
    if lr_bundle is not None:
        try:
            X_lr = X.copy()
            for col, med in lr_bundle["col_medians"].items():
                if col in X_lr.columns:
                    X_lr[col] = X_lr[col].fillna(med)
            X_lr_scaled = lr_bundle["scaler"].transform(X_lr.values)
            lr_prob = float(lr_bundle["model"].predict_proba(X_lr_scaled)[0][1])
            # Equal-weight blend: 10 LGBM + 1 LR
            lgbm_prob = (lgbm_prob * 10 + lr_prob) / 11
        except Exception:
            pass  # silently fall back to LGBM-only if LR fails

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
        return (round(lgbm_prob, 4), ensemble_std) if return_std else round(lgbm_prob, 4)

    alpha   = n_real / _MIN_REAL_ROWS
    blended = alpha * lgbm_prob + (1 - alpha) * hit_rate_lifetime
    return (round(blended, 4), ensemble_std) if return_std else round(blended, 4)


# ---------------------------------------------------------------------------
# AI-Futures-Trader pipeline compatibility
#
# Their pipeline/run_event.py does:
#   from pipeline._5_probability.kalshi_model import (
#       predict_proba, predict_proba_full, compute_event_word_ranks, ensemble_size,
#   )
# and calls predict_proba/predict_proba_full with a news_word_relative_rank
# kwarg (predict_proba already accepts and ignores it, above). These three
# functions cover the rest of that import so this module is a drop-in
# replacement for theirs — same trained model, their pipeline's gating/sizing
# stays untouched. Their db.get_cached_profile has the same signature/schema
# as ours, so no db changes are needed on their side.
# ---------------------------------------------------------------------------

def predict_proba_full(
    speaker: str,
    word: str,
    event_type: str = "",
    kalshi_odds: float = 0.5,
    news_articles: Optional[list[dict]] = None,
    event_title: str = "",
    news_word_relative_rank: float = float("nan"),
    ko_velocity_24h: float = float("nan"),
    ko_velocity_48h: float = float("nan"),
) -> tuple[float, float]:
    """Same as predict_proba(..., return_std=True) under their expected name."""
    return predict_proba(
        speaker=speaker, word=word, event_type=event_type,
        kalshi_odds=kalshi_odds, news_articles=news_articles,
        event_title=event_title, return_std=True,
        news_word_relative_rank=news_word_relative_rank,
        ko_velocity_24h=ko_velocity_24h, ko_velocity_48h=ko_velocity_48h,
    )


def compute_event_word_ranks(
    news_by_word: dict[str, list],
    speaker: str,
    event_type: str = "",
) -> dict[str, float]:
    """
    Compute news_word_relative_rank for all words in an event, called once
    before the per-word predict_proba loop (their pipeline's pattern).
    Returns {word: percentile_rank}, 0=least newsworthy, 1=most newsworthy.
    Not used by our own FEATURES list, but their run_event.py computes this
    before calling predict_proba and passes it in — provided for compatibility.
    """
    scores: dict[str, float] = {}
    for word, articles in news_by_word.items():
        feats = _compute_raw_news_features(articles, speaker, word)
        d = feats.get("news_decay_score")
        if d is not None:
            scores[word] = float(d)

    if not scores:
        return {}

    sorted_vals = sorted(scores.values())
    n = len(sorted_vals)
    return {
        word: (sorted_vals.index(score) / (n - 1) if n > 1 else 0.5)
        for word, score in scores.items()
    }


def ensemble_size() -> int:
    """Number of models loaded in the inference ensemble."""
    return len(_get_ensemble())


def _count_real_rows() -> int:
    with db._connect() as conn:
        n_td = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        n_tl = conn.execute(
            "SELECT COUNT(*) FROM trade_log WHERE outcome IN ('win','loss')"
        ).fetchone()[0]
    return n_td + n_tl


def retrain() -> lgb.Booster:
    global _booster_cache, _calibrator_cache, _word_priors_cache, \
           _ensemble_cache, _word_ranks_cache, _speaker_activity_cache, \
           _word_semantic_proximity_cache
    booster = train(save=True, verbose=False)
    _booster_cache           = booster
    _calibrator_cache        = _load_calibrator()
    _word_priors_cache       = _load_word_priors()
    _ensemble_cache          = None   # force reload of full ensemble
    _word_ranks_cache        = None   # force recompute after new data
    _speaker_activity_cache  = None   # force recompute after new data
    _word_semantic_proximity_cache = None  # force recompute after new data
    return booster


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _cli_train() -> None:
    print("Training LightGBM model ...\n")
    train(save=True, verbose=True)


def _cli_eval() -> None:
    ensemble = load_ensemble()
    if not ensemble:
        print("No model found. Run --train first.")
        return
    calibrator = _load_calibrator()
    df, priors = build_training_dataset(verbose=True)

    n_before = len(df)
    df = df[df["n_samples_lifetime"] >= 2].reset_index(drop=True)
    print(f"  Warm-only filter  : {n_before} → {len(df)} rows")

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
        speakers=df_train["_speaker"].tolist() if "_speaker" in df_train.columns else None,
    )
    X_test = _build_features_with_priors(df_test, test_priors)
    y_test = df_test["did_say_word"].astype(int)

    # Use full ensemble (same as live inference), not just seed-0
    test_probs_raw = np.mean([b.predict(X_test) for b in ensemble], axis=0)
    test_probs_cal = (_apply_calibrator(calibrator, test_probs_raw)
                      if calibrator is not None else test_probs_raw)
    _print_metrics_from_probs(y_test.values, test_probs_cal,
                              ensemble_size=len(ensemble))
    _print_calibration(y_test.values, test_probs_raw, test_probs_cal)
    _print_importance(ensemble[0])
    print(f"\n  Ensemble size     : {len(ensemble)}")
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
        speakers=df_train["_speaker"].tolist() if "_speaker" in df_train.columns else None,
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
