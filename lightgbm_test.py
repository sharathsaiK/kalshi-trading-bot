"""
lightgbm_test.py
----------------
Proof-of-concept: synthetic Kalshi trade dataset → LightGBM classifier.

Answers:
  1. Which parameter combinations make LightGBM say "yes" to a trade?
  2. How does it rank feature importance?
  3. What do the evaluation metrics look like?

Run:
    python3 lightgbm_test.py
"""

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix, classification_report,
)

RNG = np.random.default_rng(42)
N   = 600  # synthetic samples


# ---------------------------------------------------------------------------
# 1. Generate synthetic training data
# ---------------------------------------------------------------------------

def make_dataset(n: int) -> pd.DataFrame:
    """
    Each row = one (speaker, event, word) prediction opportunity.
    Features mirror what the real pipeline produces.
    Label = did_say_word (1 = yes, 0 = no).
    """
    # --- Speaker profile features ---
    hit_rate_lifetime = RNG.beta(2, 3, n).clip(0.05, 0.95)   # avg ~0.40
    hit_rate_recent   = (hit_rate_lifetime
                         + RNG.normal(0, 0.12, n)).clip(0.0, 1.0)
    momentum          = hit_rate_recent - hit_rate_lifetime    # negative = cooling off
    avg_freq          = RNG.gamma(2, 2, n).clip(0.5, 15.0)    # avg times said per speech
    recency           = RNG.beta(2, 2, n)                      # 0=said long ago, 1=said recently
    n_samples_lifetime = RNG.integers(3, 60, n)
    n_samples_recent   = RNG.integers(0, 15, n)

    # --- News relevancy features (from aggregate_relevancy_features) ---
    rel_mean      = RNG.beta(3, 4, n).clip(0.1, 0.9)
    rel_max       = (rel_mean + RNG.uniform(0.05, 0.25, n)).clip(0.0, 1.0)
    rel_top3_mean = (rel_mean + RNG.uniform(0.0, 0.15, n)).clip(0.0, 1.0)
    rel_count_hi  = RNG.integers(0, 15, n)
    rel_n         = RNG.integers(0, 25, n)

    # --- Market / contract features ---
    kalshi_odds = RNG.beta(2, 3, n).clip(0.05, 0.92)   # market implied prob
    ev_score    = RNG.normal(0.08, 0.12, n)             # expected value; positive = edge

    # --- Event type (encoded) ---
    event_types = RNG.choice(
        [0, 1, 2, 3, 4],
        n,
        p=[0.10, 0.30, 0.15, 0.35, 0.10],  # fomc, sotu, debate, speech, press_conf
    )

    # --- Label: did_say_word ---
    # Ground truth probability is a weighted combo of signals + noise.
    true_prob = (
        0.38 * hit_rate_lifetime
        + 0.22 * hit_rate_recent
        + 0.12 * (momentum + 1) / 2       # normalize to [0,1]
        + 0.12 * rel_mean
        + 0.08 * recency
        + 0.05 * (avg_freq / 15).clip(0, 1)
        + 0.03 * (rel_count_hi / 15).clip(0, 1)
        + RNG.normal(0, 0.07, n)           # noise
    ).clip(0.0, 1.0)

    did_say_word = (true_prob > 0.48).astype(int)

    return pd.DataFrame({
        # Speaker profile
        "hit_rate_lifetime":  hit_rate_lifetime,
        "hit_rate_recent":    hit_rate_recent,
        "momentum":           momentum,
        "avg_freq":           avg_freq,
        "recency":            recency,
        "n_samples_lifetime": n_samples_lifetime,
        "n_samples_recent":   n_samples_recent,
        # News
        "rel_max":            rel_max,
        "rel_mean":           rel_mean,
        "rel_top3_mean":      rel_top3_mean,
        "rel_count_hi":       rel_count_hi,
        "rel_n":              rel_n,
        # Market
        "kalshi_odds":        kalshi_odds,
        "ev_score":           ev_score,
        "event_type":         event_types,
        # Label
        "did_say_word":       did_say_word,
    })


df = make_dataset(N)

FEATURES = [c for c in df.columns if c != "did_say_word"]
X = df[FEATURES]
y = df["did_say_word"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

print(f"Dataset: {N} rows  |  said word: {y.sum()} ({y.mean():.1%})  |  didn't: {(1-y).sum()}")
print(f"Train: {len(X_train)}  |  Test: {len(X_test)}\n")


# ---------------------------------------------------------------------------
# 2. Train LightGBM
# ---------------------------------------------------------------------------

params = {
    "objective":      "binary",
    "metric":         "binary_logloss",
    "n_estimators":   300,
    "learning_rate":  0.05,
    "max_depth":      5,
    "num_leaves":     20,
    "min_child_samples": 15,
    "feature_fraction":  0.8,
    "bagging_fraction":  0.8,
    "bagging_freq":      5,
    "reg_alpha":      0.1,
    "reg_lambda":     0.1,
    "random_state":   42,
    "verbose":       -1,
}

model = lgb.LGBMClassifier(**params)
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)],
)
print(f"Best iteration: {model.best_iteration_}\n")


# ---------------------------------------------------------------------------
# 3. Evaluation metrics
# ---------------------------------------------------------------------------

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)[:, 1]

print("=" * 55)
print("EVALUATION METRICS")
print("=" * 55)
print(f"  Accuracy  : {accuracy_score(y_test, y_pred):.3f}")
print(f"  Precision : {precision_score(y_test, y_pred):.3f}   (of trades we bet YES, how many were right)")
print(f"  Recall    : {recall_score(y_test, y_pred):.3f}   (of real YES events, how many did we catch)")
print(f"  F1        : {f1_score(y_test, y_pred):.3f}")
print(f"  AUC-ROC   : {roc_auc_score(y_test, y_pred_prob):.3f}   (1.0 = perfect, 0.5 = random)")

cm = confusion_matrix(y_test, y_pred)
print(f"\nConfusion Matrix:")
print(f"                  Predicted NO   Predicted YES")
print(f"  Actual NO   :     {cm[0][0]:>5}           {cm[0][1]:>5}   (false positives = bad bets)")
print(f"  Actual YES  :     {cm[1][0]:>5}           {cm[1][1]:>5}   (false negatives = missed trades)")


# ---------------------------------------------------------------------------
# 4. Feature importance
# ---------------------------------------------------------------------------

importance = pd.Series(
    model.feature_importances_,
    index=FEATURES,
).sort_values(ascending=False)

print(f"\n{'=' * 55}")
print("FEATURE IMPORTANCE (how much each feature influenced decisions)")
print(f"{'=' * 55}")
for feat, score in importance.items():
    bar = "█" * (score // 5)
    print(f"  {feat:<22} {score:>4}  {bar}")


# ---------------------------------------------------------------------------
# 5. Case studies — what makes LightGBM say YES?
# ---------------------------------------------------------------------------

EVENT_LABELS = {0: "fomc", 1: "sotu", 2: "debate", 3: "speech", 4: "press_conf"}

case_studies = pd.DataFrame([
    # Strong YES — high hit rate, positive momentum, hot news coverage
    {"name": "Trump/tariff (SOTU, hot topic)",
     "hit_rate_lifetime": 0.80, "hit_rate_recent": 0.90, "momentum": 0.10,
     "avg_freq": 6.0, "recency": 0.95, "n_samples_lifetime": 25, "n_samples_recent": 8,
     "rel_max": 0.88, "rel_mean": 0.74, "rel_top3_mean": 0.82, "rel_count_hi": 9, "rel_n": 18,
     "kalshi_odds": 0.70, "ev_score": 0.15, "event_type": 1},

    # Moderate YES — decent history, some news signal
    {"name": "Powell/inflation (FOMC, moderate)",
     "hit_rate_lifetime": 0.60, "hit_rate_recent": 0.65, "momentum": 0.05,
     "avg_freq": 3.5, "recency": 0.70, "n_samples_lifetime": 18, "n_samples_recent": 5,
     "rel_max": 0.65, "rel_mean": 0.52, "rel_top3_mean": 0.60, "rel_count_hi": 4, "rel_n": 10,
     "kalshi_odds": 0.55, "ev_score": 0.08, "event_type": 0},

    # Edge case — high market odds but weak profile
    {"name": "Unknown speaker (market hype, no history)",
     "hit_rate_lifetime": 0.25, "hit_rate_recent": 0.30, "momentum": 0.05,
     "avg_freq": 1.5, "recency": 0.20, "n_samples_lifetime": 4, "n_samples_recent": 1,
     "rel_max": 0.72, "rel_mean": 0.60, "rel_top3_mean": 0.68, "rel_count_hi": 6, "rel_n": 14,
     "kalshi_odds": 0.75, "ev_score": 0.05, "event_type": 3},

    # Strong NO — low history, negative momentum, cold news
    {"name": "Speaker cooling off (low signal)",
     "hit_rate_lifetime": 0.35, "hit_rate_recent": 0.20, "momentum": -0.15,
     "avg_freq": 1.2, "recency": 0.10, "n_samples_lifetime": 10, "n_samples_recent": 3,
     "rel_max": 0.30, "rel_mean": 0.22, "rel_top3_mean": 0.27, "rel_count_hi": 1, "rel_n": 4,
     "kalshi_odds": 0.30, "ev_score": -0.05, "event_type": 3},

    # Trap — high EV but model sees weak fundamentals
    {"name": "High EV trap (good odds, bad profile)",
     "hit_rate_lifetime": 0.20, "hit_rate_recent": 0.15, "momentum": -0.05,
     "avg_freq": 0.8, "recency": 0.05, "n_samples_lifetime": 8, "n_samples_recent": 2,
     "rel_max": 0.40, "rel_mean": 0.30, "rel_top3_mean": 0.36, "rel_count_hi": 2, "rel_n": 6,
     "kalshi_odds": 0.20, "ev_score": 0.22, "event_type": 3},

    # Borderline — mixed signals
    {"name": "Borderline (mixed signals)",
     "hit_rate_lifetime": 0.50, "hit_rate_recent": 0.48, "momentum": -0.02,
     "avg_freq": 2.5, "recency": 0.50, "n_samples_lifetime": 12, "n_samples_recent": 4,
     "rel_max": 0.55, "rel_mean": 0.44, "rel_top3_mean": 0.51, "rel_count_hi": 3, "rel_n": 8,
     "kalshi_odds": 0.45, "ev_score": 0.03, "event_type": 3},
])

names = case_studies.pop("name")
probs = model.predict_proba(case_studies[FEATURES])[:, 1]
decisions = ["✅ BET YES" if p >= 0.50 else "❌ SKIP" for p in probs]

print(f"\n{'=' * 55}")
print("CASE STUDIES — Would LightGBM bet?")
print(f"{'=' * 55}")
for name, prob, decision in zip(names, probs, decisions):
    print(f"\n  {decision}  ({prob:.1%} confidence)")
    print(f"  → {name}")

print(f"\n{'=' * 55}")
print("KEY TAKEAWAY")
print(f"{'=' * 55}")
top2 = importance.head(2).index.tolist()
print(f"  Most important features: {top2[0]} + {top2[1]}")
print(f"  Model threshold: predict YES if P(say_word) >= 50%")
print(f"  In production, raise threshold to ~60-65% to reduce false positives")
