"""
api_server.py
-------------
Flask REST API that exposes kalshi.db and model data to the dashboard.
Run:  python3 api_server.py
Port: 8765
"""

from __future__ import annotations

import json
import os
import pickle
import subprocess
import sys
import threading
from datetime import datetime
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, request
from flask_cors import CORS

import db

BASE = Path(__file__).parent
CONFIG_PATH = BASE / "pipeline_config.json"
HARVEST_LOG = BASE / "harvest.log"
PIPELINE_LOG = BASE / "pipeline.log"
METRICS_PATH = BASE / "model_metrics.json"
MODEL_SEEDS = 11

# ── Pipeline defaults (mirrors run_pipeline.py constants) ────────────────────
_DEFAULTS: dict[str, Any] = {
    "yesMinEdge":      0.22,
    "noMinEdge":       0.10,
    "yesProbFloor":    0.72,
    "noProbCeil":      0.30,
    "confMaxMult":     3.0,
    "bankroll":        1000.0,
    "kellyFraction":   0.25,
    "maxPositionPct":  0.10,
    "baseRatePct":     0.05,
    "nSamplesMin":     3,
    "maxSpread":       0.10,
    "minVolume":       100.0,
    "minTimeClose":    30,
    "noOddsCeil":      0.65,
    "apiPing":         "0.4",
    "newsPing":        "4",
    "harvestPing":     "24",
    "retrainRows":     "25",
    "transcriptRefresh": "6",
    "minTranscriptChars": "5000",
    "autoRetrain":     True,
    "liveMode":        False,
}

app = Flask(__name__)
CORS(app)

# ── Shared training state ─────────────────────────────────────────────────────
_train_state: dict[str, Any] = {"running": False, "seed": 0, "progress": 0, "done": False}
_train_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            pass
    return {}


def _save_config(data: dict) -> None:
    merged = {**_DEFAULTS, **data}
    CONFIG_PATH.write_text(json.dumps(merged, indent=2))


def _settings() -> dict:
    return {**_DEFAULTS, **_load_config()}


def _model_feature_importance() -> list[dict]:
    """Load seed-1 model and extract feature importance."""
    try:
        import lightgbm as lgb
        from kalshi_model import FEATURES

        seed1 = BASE / "kalshi_model_seed_1.lgb"
        if not seed1.exists():
            return []
        m = lgb.Booster(model_file=str(seed1))
        gains = m.feature_importance(importance_type="gain")
        total = gains.sum() or 1.0
        feat_names = m.feature_name()
        result = []
        for name, g in zip(feat_names, gains):
            cat = (
                "profile" if name in ("hit_rate_lifetime", "hit_rate_recent", "momentum",
                                      "avg_freq", "recency", "n_samples_lifetime", "word_rank")
                else "word"   if "word" in name or "prior" in name
                else "market" if "kalshi" in name or "market_vs" in name
                else "news"   if name.startswith("rel_")
                else "event"  if "event" in name or "days_since" in name or "events_in" in name
                else "other"
            )
            result.append({"feature": name, "importance": round(float(g) / total * 100, 2), "category": cat})
        result.sort(key=lambda x: x["importance"], reverse=True)
        return result
    except Exception as e:
        return [{"feature": f"error: {e}", "importance": 0.0, "category": "other"}]


def _model_mtime() -> str:
    p = BASE / "kalshi_model_seed_1.lgb"
    if p.exists():
        ts = datetime.utcfromtimestamp(p.stat().st_mtime)
        return ts.strftime("%Y-%m-%d %H:%M UTC")
    return "unknown"


def _parse_log_line(line: str) -> dict | None:
    """Try to parse a timestamped log line, fall back to raw."""
    line = line.strip()
    if not line:
        return None
    # Detect [LEVEL] tag
    level = "INFO"
    if "[error]" in line.lower() or "ERROR" in line:
        level = "ERROR"
    elif "warn" in line.lower() or "WARNING" in line:
        level = "WARN"

    # Detect source from content
    source = "BOT"
    lower = line.lower()
    if "harvest" in lower or "holdout" in lower:
        source = "HARVEST"
    elif "train" in lower or "lgb" in lower or "brier" in lower:
        source = "TRAIN"
    elif "trade" in lower or "bet" in lower or "kelly" in lower:
        source = "TRADE"
    elif "news" in lower or "article" in lower:
        source = "NEWS"
    elif "transcript" in lower:
        source = "TRANS"
    elif "model" in lower or "predict" in lower:
        source = "MODEL"
    elif "api" in lower or "http" in lower or "kalshi" in lower:
        source = "API"

    now = datetime.utcnow()
    return {
        "id": hash(line) & 0x7FFFFFFF,
        "time": now.strftime("%H:%M:%S"),
        "level": level,
        "source": source,
        "message": line,
    }


def _tail_log(path: Path, n: int = 300) -> list[dict]:
    if not path.exists():
        return []
    try:
        lines = path.read_text(errors="replace").splitlines()[-n:]
        out = []
        for i, line in enumerate(lines):
            parsed = _parse_log_line(line)
            if parsed:
                parsed["id"] = i
                out.append(parsed)
        return out
    except Exception:
        return []


# ── Routes: system/logs ───────────────────────────────────────────────────────

@app.get("/api/system/logs")
def system_logs():
    logs = _tail_log(HARVEST_LOG)
    if PIPELINE_LOG.exists():
        logs += _tail_log(PIPELINE_LOG, 200)
    logs.sort(key=lambda x: x["id"])
    for i, l in enumerate(logs):
        l["id"] = i
    return jsonify(logs)


# ── Routes: system/settings ───────────────────────────────────────────────────

@app.get("/api/system/settings")
def system_settings_get():
    return jsonify(_settings())


@app.post("/api/system/settings")
def system_settings_post():
    data = request.get_json(force=True) or {}
    _save_config(data)
    return jsonify({"ok": True, "settings": _settings()})


# ── Routes: model/training-stats ─────────────────────────────────────────────

@app.get("/api/model/training-stats")
def model_training_stats():
    with db._connect() as conn:
        total_train = conn.execute("SELECT COUNT(*) FROM training_data").fetchone()[0]
        total_holdout = conn.execute("SELECT COUNT(*) FROM training_data_holdout").fetchone()[0]

        # Pre-cutoff = training_data only (all rows are pre-cutoff by design)
        speakers_train = conn.execute(
            "SELECT speaker, COUNT(*) as cnt FROM training_data GROUP BY speaker ORDER BY cnt DESC"
        ).fetchall()

        # Count real rows (rows where there was a kalshi_odds price)
        real_rows = conn.execute(
            "SELECT COUNT(*) FROM training_data WHERE kalshi_odds > 0.04 AND kalshi_odds < 0.96"
        ).fetchone()[0]

        # Holdout stats
        holdout_stats = conn.execute(
            "SELECT speaker, COUNT(*) as cnt FROM training_data_holdout GROUP BY speaker ORDER BY cnt DESC"
        ).fetchall()

    by_speaker = [{"speaker": r[0], "rows": r[1]} for r in speakers_train]
    holdout_by_speaker = [{"speaker": r[0], "rows": r[1]} for r in holdout_stats]

    return jsonify({
        "totalRows": total_train + total_holdout,
        "preCutoffRows": total_train,
        "holdoutRows": total_holdout,
        "realRows": real_rows,
        "syntheticRows": max(0, total_train - real_rows),
        "bySpeaker": by_speaker,
        "holdoutBySpeaker": holdout_by_speaker,
        "modelMtime": _model_mtime(),
        "features": len([]), # filled below
        "seeds": MODEL_SEEDS,
        "architecture": f"{MODEL_SEEDS}-seed DART ensemble + LR",
        "holdoutCutoff": "2026-03-01",
    })


# ── Routes: model/profiles ────────────────────────────────────────────────────

@app.get("/api/model/profiles")
def model_profiles():
    speaker = request.args.get("speaker")
    search = request.args.get("search", "").lower()
    limit = int(request.args.get("limit", 500))

    with db._connect() as conn:
        if speaker and speaker != "All":
            rows = conn.execute(
                "SELECT * FROM speaker_profiles WHERE speaker=? ORDER BY n_samples_lifetime DESC LIMIT ?",
                (speaker, limit),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM speaker_profiles ORDER BY n_samples_lifetime DESC LIMIT ?",
                (limit,),
            ).fetchall()

    result = [dict(r) for r in rows]

    if search:
        result = [r for r in result
                  if search in r.get("word", "").lower()
                  or search in r.get("speaker", "").lower()]

    return jsonify(result)


# ── Routes: model/transcripts ─────────────────────────────────────────────────

@app.get("/api/model/transcripts")
def model_transcripts():
    speaker = request.args.get("speaker")
    search = request.args.get("search", "").lower()
    limit = int(request.args.get("limit", 100))
    offset = int(request.args.get("offset", 0))

    sp = speaker if (speaker and speaker != "All") else None
    rows = db.get_transcripts(speaker=sp, limit=limit, offset=offset)

    if search:
        rows = [r for r in rows if search in r.get("full_text", "").lower()
                or search in r.get("event_ticker", "").lower()]

    # Trim full_text to a preview for the list view
    for r in rows:
        text = r.get("full_text", "")
        r["chars"] = len(text)
        r["preview"] = text[:120].replace("\n", " ") if text else ""
        del r["full_text"]

    return jsonify(rows)


# ── Routes: model/metrics ─────────────────────────────────────────────────────

@app.get("/api/model/metrics")
def model_metrics():
    if METRICS_PATH.exists():
        try:
            return jsonify(json.loads(METRICS_PATH.read_text()))
        except Exception:
            pass
    return jsonify({"accuracy": None, "auc": None, "brier": None, "history": []})


# ── Routes: model/features ────────────────────────────────────────────────────

@app.get("/api/model/features")
def model_features():
    return jsonify(_model_feature_importance())


# ── Routes: model/train ───────────────────────────────────────────────────────

@app.get("/api/model/train-status")
def model_train_status():
    with _train_lock:
        return jsonify(dict(_train_state))


@app.post("/api/model/train")
def model_train():
    with _train_lock:
        if _train_state["running"]:
            return jsonify({"ok": False, "error": "Training already running"}), 409
        _train_state.update({"running": True, "seed": 0, "progress": 0, "done": False})

    def _run():
        try:
            venv_python = str(BASE / "venv" / "bin" / "python3")
            proc = subprocess.Popen(
                [venv_python, str(BASE / "pseudo_trade.py")],
                cwd=str(BASE),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            seed = 0
            for line in proc.stdout:
                line = line.strip()
                if "seed" in line.lower() or "lgb" in line.lower():
                    seed = min(seed + 1, MODEL_SEEDS)
                    with _train_lock:
                        _train_state["seed"] = seed
                        _train_state["progress"] = round(seed / MODEL_SEEDS * 100)
            proc.wait()
        finally:
            with _train_lock:
                _train_state.update({"running": False, "seed": MODEL_SEEDS, "progress": 100, "done": True})

    threading.Thread(target=_run, daemon=True).start()
    return jsonify({"ok": True})


# ── Entrypoint ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("API_PORT", 8765))
    print(f"[api_server] listening on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=False, threaded=True)
