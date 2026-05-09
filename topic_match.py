"""
topic_match.py
--------------
Semantic context awareness for the Kalshi bot.

Computes how well a target word fits the topic of a specific event.
Uses sentence-transformers (all-MiniLM-L6-v2) to measure cosine similarity
between the event title/description and the target word.

Replaced spaCy word vectors (en_core_web_md) because word2vec assigns
inflated scores to politically-flavored phrases regardless of context —
e.g. "Save America Act" scored 0.55 against "bilateral meeting with Japan"
because "America" is geopolitically similar to "Japan" in GloVe space.
Sentence-transformers encode full phrase meaning, so domestic-legislation
phrases score near-zero against foreign-meeting titles.

Public API:
  compute_match(event_title: str, word: str) -> float    # 0..1
  compute_match_safe(...) -> float                        # never raises
  classify_event(event_title: str) -> str                # event category
"""

from __future__ import annotations

import functools
import re
from typing import Optional

# Lazy-loaded sentence-transformer model
_MODEL = None

# Strip Kalshi's templated boilerplate from event titles so we can extract
# the actual descriptive part. Examples:
#   "What will Trump say during his Cabinet Meeting?"
#       → "Cabinet Meeting"
#   "What will Trump say during the Memphis Safety Roundtable?"
#       → "Memphis Safety Roundtable"
#   "Who will Trump mention during his State of the Union address?"
#       → "State of the Union address"
_TEMPLATE_RE = re.compile(
    r"^\s*(?:what|who|where|how|when)\s+will\s+\w+\s+(?:say|mention|talk\s+about)"
    r"\s+(?:in|during|at)?\s+(?:his|her|the|a|an|in)?\s*",
    re.IGNORECASE,
)


def _clean_title(title: str) -> str:
    """Strip the templated 'What will X say during ...' prefix."""
    if not title:
        return ""
    cleaned = _TEMPLATE_RE.sub("", title).strip()
    cleaned = re.sub(r"[?.!]+$", "", cleaned).strip()
    return cleaned or title


_NLP = None


def _get_nlp():
    """Load the spaCy md model lazily (only when first needed)."""
    global _NLP
    if _NLP is None:
        try:
            import spacy
            _NLP = spacy.load("en_core_web_md")
        except (ImportError, OSError):
            _NLP = False
    return _NLP if _NLP else None


@functools.lru_cache(maxsize=4096)
def _embed(text: str):
    """Cache spaCy embeddings."""
    nlp = _get_nlp()
    if nlp is None or not text:
        return None
    return nlp(text)


def compute_match(event_title: str, word: str) -> float:
    """
    spaCy cosine similarity between event title and target word, in [0, 1].
    Used as a LightGBM training/inference feature — model was trained on these scores.
    Returns 0.5 (neutral) if spaCy is unavailable or text is empty.
    """
    if not event_title or not word:
        return 0.5

    cleaned_title = _clean_title(event_title)
    title_doc = _embed(cleaned_title.lower().strip())
    word_doc  = _embed(word.lower().strip())

    if title_doc is None or word_doc is None:
        return 0.5
    if not title_doc.has_vector or not word_doc.has_vector:
        return 0.5
    if title_doc.vector_norm == 0 or word_doc.vector_norm == 0:
        return 0.5

    sim = float(title_doc.similarity(word_doc))
    return max(0.0, min(1.0, sim))


def compute_match_safe(event_title: Optional[str], word: Optional[str]) -> float:
    """Like compute_match but never raises — returns 0.5 on any failure."""
    try:
        return compute_match(event_title or "", word or "")
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Transformer-based match — used only for the veto gate, NOT as a model feature
# ---------------------------------------------------------------------------

_ST_MODEL = None


def _get_st_model():
    """Load sentence-transformer lazily on first use."""
    global _ST_MODEL
    if _ST_MODEL is None:
        try:
            from sentence_transformers import SentenceTransformer
            _ST_MODEL = SentenceTransformer("all-MiniLM-L6-v2")
        except (ImportError, Exception):
            _ST_MODEL = False
    return _ST_MODEL if _ST_MODEL else None


@functools.lru_cache(maxsize=4096)
def _embed_transformer(text: str):
    model = _get_st_model()
    if model is None or not text:
        return None
    return model.encode(text, convert_to_tensor=True)


def compute_match_transformer(event_title: Optional[str], word: Optional[str]) -> float:
    """
    Cosine similarity using sentence-transformers (all-MiniLM-L6-v2).

    Used exclusively for the YES-bet veto gate in kalshi_model.predict_proba —
    NOT as a LightGBM feature (the model was trained on spaCy scores).

    Score range: roughly 0.0–0.35 for most pairs. Negative raw cosines are
    clamped to 0. Returns 0.5 (neutral / don't veto) on any failure.
    """
    if not event_title or not word:
        return 0.5
    try:
        from sentence_transformers import util as st_util
        cleaned = _clean_title(event_title or "")
        t_emb = _embed_transformer(cleaned.strip())
        w_emb = _embed_transformer((word or "").strip())
        if t_emb is None or w_emb is None:
            return 0.5
        sim = float(st_util.cos_sim(t_emb, w_emb))
        return max(0.0, min(1.0, sim))
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# Event category classifier
# ---------------------------------------------------------------------------

_FOREIGN_KW = {
    "bilateral", "summit", "g7", "g20", "g-7", "g-20",
    "diplomatic", "foreign minister", "prime minister",
    "chancellor", "president of",
    "japan", "china", "uk ", "france", "germany", "italy",
    "canada", "india", "south korea", "north korea", "korea",
    "israel", "ukraine", "russia", "saudi", "mexico", "brazil",
    "modi", "macron", "scholz", "kishida", "netanyahu", "zelensky",
    "putin", "xi jinping", "trudeau", "meloni",
    "nato summit", "un general assembly", "world economic forum",
    "davos", "oval office meeting with", "meeting with",
}
_CEREMONIAL_KW = {
    "mother's day", "mothers day", "father's day", "fathers day",
    "opening night", "opening day",
    "championship", "super bowl", "world series",
    "nba finals", "nfl", "mlb", "nhl", "olympic",
    "inauguration", "commencement", "graduation",
    "award ceremony", "awards", "gala",
    "memorial day", "veterans day", "veterans",
    "thanksgiving", "christmas", "easter",
    "independence day", "fourth of july",
    "national prayer", "prayer breakfast",
    "kentucky derby", "daytona",
}
_RALLY_KW = {
    "rally", "maga rally", "save america rally", "campaign rally",
    "campaign event", "fundraiser", "town hall", "political rally",
}
_PRESS_CONF_KW = {
    "press conference", "press briefing", "media briefing", "gaggle",
    "q&a", "q & a",
}


def classify_event(event_title: str) -> str:
    """
    Classify event title into a broad category using keyword matching.

    Returns one of:
      "foreign_diplomatic"  — bilateral meetings, foreign summits, world leaders
      "ceremonial"          — sports events, holidays, award ceremonies
      "rally"               — MAGA/campaign rallies
      "press_conf"          — press conferences / briefings
      "domestic_political"  — cabinet meetings, roundtables, domestic speeches (default)
    """
    if not event_title:
        return "domestic_political"

    text = _clean_title(event_title).lower()

    if any(kw in text for kw in _FOREIGN_KW):
        return "foreign_diplomatic"
    if any(kw in text for kw in _CEREMONIAL_KW):
        return "ceremonial"
    if any(kw in text for kw in _RALLY_KW):
        return "rally"
    if any(kw in text for kw in _PRESS_CONF_KW):
        return "press_conf"
    return "domestic_political"


if __name__ == "__main__":
    test_cases = [
        ("Chicago Opening Night baseball event", "Biden"),
        ("bilateral meeting with Japan", "Save Act Save America Act"),
        ("bilateral meeting with Japan", "Taiwan"),
        ("Military Mothers Day Event", "Epic Fury"),
        ("Military Mothers Day Event", "Hegseth"),
        ("Cabinet Meeting", "Tariff"),
        ("Memphis Safety Roundtable", "ICE"),
    ]
    print(f"{'Event':<45} {'Word':<35} {'Match':>6}")
    print("-" * 90)
    for title, word in test_cases:
        m = compute_match(title, word)
        print(f"{title:<45} {word:<35} {m:>6.3f}")
