# Kalshi Prediction Market Bot

LightGBM model that predicts whether a named speaker will say a target word during a live event, traded as YES/NO contracts on Kalshi.

**Tracked speakers:** Donald Trump, J.D. Vance, Jerome Powell, Marco Rubio, Michael Barr, Pete Hegseth

---

## Core Files

| File | Purpose |
|---|---|
| `run_pipeline.py` | Live trading orchestrator. Fetches open Kalshi markets, runs predictions, and logs bet recommendations. |
| `kalshi_model.py` | LightGBM model: training, CV, 12-seed ensemble, Platt calibration, feature engineering, and `predict_proba()` for inference. |
| `pseudo_trade.py` | Fixed-holdout evaluator (post-2026-03-01). Measures Brier, AUC, P&L, and ROI/bet against the holdout set. Run this after every model change. |
| `backtest_trades.py` | Per-event backtest. Usage: `python3 backtest_trades.py <TICKER>` |
| `db.py` | SQLite layer. Tables: `speaker_profiles`, `training_data`, `training_data_holdout`, `trade_log`. |

## Data Collection

| File | Purpose |
|---|---|
| `harvest_training_data.py` | Scrapes settled Kalshi markets and populates `training_data`. Use `--holdout` flag for post-cutoff events. |
| `kalshi_api.py` | Kalshi REST client — events, markets, candlestick prices. |
| `news_scraper.py` | Fetches news relevancy scores for word/event pairs (Guardian, NYT, NewsAPI, FMP). |
| `backfill_news.py` | Backfills news features for existing training or holdout rows. |
| `backfill_topic_match.py` | Backfills `topic_match` transformer scores for existing rows. |
| `kalshi_word_counter.py` | Counts word occurrences in event transcripts to determine `did_say_word`. |
| `transcript_bot.py` | Fetches and parses event transcripts (YouTube captions, etc.). |

## Model Utilities

| File | Purpose |
|---|---|
| `topic_match.py` | Transformer-based semantic similarity between a word and event title (used as a feature). |
| `queries.py` | Shared SQL queries used across multiple scripts. |
| `connection.py` | Database connection helper. |
| `maintenance.py` | DB cleanup and maintenance tasks. |
| `rebuild_profiles_from_training.py` | Rebuilds `speaker_profiles` table from scratch using `training_data`. |
| `restore_kalshi_odds.py` | Restores missing `kalshi_odds` values from API for existing training rows. |
| `fix_training_prices.py` | Corrects malformed price entries in `training_data`. |
| `profile_agent.py` | Builds and updates speaker-word profiles. |

## Other

| File | Purpose |
|---|---|
| `backtest.py` | Older backtest script (superseded by `backtest_trades.py`). |
| `lightgbm_test.py` | Sandbox for testing LightGBM behaviour in isolation. |
| `kalshi_word_counter.py` | Word-frequency counter used to compute speaker vocab stats. |

---

## Quick Start

```bash
cd ~/Projects/kalshi
source venv/bin/activate

# Train model and evaluate on holdout
python3 pseudo_trade.py

# Evaluate only (no retrain)
python3 pseudo_trade.py --no-train

# Run live pipeline for a specific event
python3 run_pipeline.py --event KXTRUMPMENTION-26MAY28

# Collect more holdout data
python3 harvest_training_data.py --holdout
```

## Current Holdout Performance (post-2026-03-01)

Three betting modes available — same model, different selectivity and sizing strategy.

| Metric | Default Mode | High-Confidence Mode | Volume + Multiplier Mode |
|---|---|---|---|
| YES edge threshold | ≥ 0.40 | ≥ 0.40 | ≥ 0.22 |
| NO edge threshold | ≥ 0.15 | ≥ 0.15 | ≥ 0.10 |
| Max NO prob cap | off | 0.15 | off |
| Contract sizing | flat Kelly | flat Kelly | Kelly × confidence (max 3×) |
| **Bets placed** | **74** | **36** | **113** |
| **Bet accuracy** | **82.4%** | **91.7%** | **76.1%** |
| **ROI / bet** | **+30.7¢** | **+26.5¢** | **+23.0¢ flat / +25.0¢ scaled** |
| **Total P&L** | **+2,275¢** | **+954¢** | **+2,598¢ flat / +5,815¢ scaled** |
| Projected P&L @301 bets | +9,254¢ | +8,009¢ | +6,920¢ flat / **+15,500¢ scaled** |
| AUC-ROC | 0.808 | 0.808 | 0.808 |
| Brier score | 0.1774 | 0.1774 | 0.1774 |

> **Which mode to use?** Volume + Multiplier is the live trading mode — lower thresholds maximise bet volume, and the confidence multiplier (capped at 3×, bankroll-safe) sizes each bet proportionally to edge strength. Projected P&L at 301 bets is +15,500¢, nearly double Default Mode. High-Confidence mode is best if minimising individual bet risk is the priority.

### Mode Comparison

![Mode comparison chart](assets/mode_comparison.png)

### Model Calibration

![Calibration chart](assets/calibration.png)
