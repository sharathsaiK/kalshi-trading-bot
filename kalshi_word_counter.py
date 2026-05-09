"""
kalshi_word_counter.py
----------------------
Kalshi-compliant word counter using spaCy.
"""

import re
import spacy
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Speaker-turn extraction
# ---------------------------------------------------------------------------

# Matches lines like "TRUMP:", "THE PRESIDENT:", "MR. VANCE:", "Q:" etc.
_TURN_LABEL_RE = re.compile(
    r'^([A-Z][A-Z0-9\s\.\-\']{0,50}?):\s+',
    re.MULTILINE,
)


def _speaker_labels(speaker_name: str) -> list[re.Pattern]:
    """
    Return compiled patterns that match transcript speaker labels for this person.
    Covers common transcript conventions across WH, press conferences, and debates.
    """
    parts  = speaker_name.strip().split()
    surname    = parts[-1].upper()  if parts         else ""
    firstname  = parts[0].upper()   if len(parts) > 1 else ""
    fullname   = speaker_name.upper()

    variants: list[str] = [
        re.escape(fullname),                                    # DONALD TRUMP
        re.escape(surname),                                     # TRUMP
        rf"THE\s+PRESIDENT",                                    # THE PRESIDENT
        rf"PRESIDENT\s+{re.escape(surname)}",                   # PRESIDENT TRUMP
        rf"MR\.\s*{re.escape(surname)}",                        # MR. TRUMP
        rf"MRS\.\s*{re.escape(surname)}",
        rf"MS\.\s*{re.escape(surname)}",
        rf"THE\s+VICE\s+PRESIDENT",                             # THE VICE PRESIDENT
        rf"VICE\s+PRESIDENT\s+{re.escape(surname)}",            # VICE PRESIDENT VANCE
        rf"SECRETARY\s+{re.escape(surname)}",                   # SECRETARY POWELL
        rf"CHAIRMAN\s+{re.escape(surname)}",                    # CHAIRMAN POWELL
        rf"SENATOR\s+{re.escape(surname)}",
        rf"REPRESENTATIVE\s+{re.escape(surname)}",
    ]
    if firstname:
        variants.append(rf"{re.escape(firstname)}\s+{re.escape(surname)}")

    return [
        re.compile(rf'(?:^|\n)({v})\s*:', re.IGNORECASE)
        for v in variants
    ]


def extract_speaker_turns(text: str, speaker_name: str) -> str:
    """
    If the transcript is in Q&A / speaker-turn format, return only the lines
    spoken by `speaker_name`. Leaves continuous-prose transcripts untouched.

    This is critical for accuracy: a reporter asking "Will you sanction nuclear
    Iran?" should NOT add 'nuclear' or 'Iran' to Trump's word count.
    """
    # Step 1 — detect whether this is a turn-based transcript.
    # A turn-based transcript has many lines that start with "LABEL:" where
    # LABEL is short, all-caps (or title-case) text. We require at least 4
    # such lines before bothering to parse turns.
    turn_labels = _TURN_LABEL_RE.findall(text)
    if len(turn_labels) < 4:
        return text  # narrative prose — return as-is

    # Step 2 — split the transcript into (label, content) segments.
    # We rebuild the text from scratch, skipping non-target segments.
    speaker_pats = _speaker_labels(speaker_name)

    # Build a combined split pattern covering ALL speaker labels in the doc.
    # We split on any "^LABEL: " so we can iterate segment by segment.
    split_pat = re.compile(
        r'(?m)^(?:[A-Z][A-Z0-9\s\.\-\']{0,50}?):\s+',
    )
    segments = split_pat.split(text)
    labels   = [m.group(0).strip().rstrip(':').strip()
                for m in split_pat.finditer(text)]

    # segments[0] is preamble text before the first speaker label.
    target_chunks: list[str] = []
    for label, segment in zip(labels, segments[1:]):
        is_target = any(p.search(label) for p in speaker_pats)
        if is_target:
            target_chunks.append(segment.strip())

    if not target_chunks:
        # No turns found for this speaker — return original to avoid blanking.
        return text

    extracted = "\n\n".join(target_chunks)
    return extracted


# ---------------------------------------------------------------------------
# Default Kalshi target words (from SOTU/mention market briefs)
# ---------------------------------------------------------------------------
DEFAULT_TARGETS = [
    "America", "border", "tariff", "tax", "God", "inflation",
    "war", "Social Security", "China", "market", "economy",
    "energy", "trillion", "fentanyl", "drill", "DOGE",
    "deportation", "immigrant", "wall", "recession", "cut",
    "Iran", "Russia", "Ukraine", "Israel", "Gaza", "NATO",
    "Palestine", "trade", "oil", "drug", "crime", "military",
    "peace", "freedom", "fire",
]

# Signal weights for sentence scoring
DEFAULT_WEIGHTS = {
    "recession": 5,
    "default":   5,
    "tariff":    4,
    "inflation": 3,
    "trillion":  3,
    "china":     3,
    "border":    2,
    "tax":       2,
    "economy":   2,
    "market":    2,
}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------
@dataclass
class CountResult:
    counts: dict[str, int]
    negations: list[dict]
    top_sentences: list[dict]
    total_tokens: int
    total_sentences: int


# ---------------------------------------------------------------------------
# Core counter
# ---------------------------------------------------------------------------
class KalshiCounter:
    """
    Kalshi-compliant word counter.

    Parameters
    ----------
    targets : list[str]
        Words/phrases to track. Case-insensitive. Multi-word phrases
        (e.g. "Social Security") are supported.
    model : str
        spaCy model name. Defaults to "blank" (fastest, tokenizer only).
        Use "en_core_web_sm" or "en_core_web_md" for negation + NER.
    weights : dict[str, int]
        Per-word weights for sentence signal scoring.
    top_n : int
        Number of top-scoring sentences to return.
    """

    def __init__(
        self,
        targets: Optional[list[str]] = None,
        model: str = "blank",
        weights: Optional[dict[str, int]] = None,
        top_n: int = 10,
    ):
        raw = [t.lower() for t in (targets or DEFAULT_TARGETS)]
        # Expand "A / B / C" alternatives into individual targets so each
        # variant is counted separately (Kalshi resolves YES if ANY fires).
        # We track which originals produced which alternatives for reporting.
        self._alt_map: dict[str, str] = {}  # expanded → original label
        expanded: list[str] = []
        for t in raw:
            if " / " in t:
                for alt in t.split(" / "):
                    alt = alt.strip()
                    expanded.append(alt)
                    self._alt_map[alt] = t
            else:
                expanded.append(t)
                self._alt_map[t] = t
        self.targets = expanded
        self.single_targets = {t for t in self.targets if " " not in t}
        self.phrase_targets = [t for t in self.targets if " " in t]
        self.weights = weights or DEFAULT_WEIGHTS
        self.top_n = top_n

        if model == "blank":
            self.nlp = spacy.blank("en")
            # Add sentencizer so we can split sentences without a full pipeline
            self.nlp.add_pipe("sentencizer")
            self._has_dep = False
        else:
            self.nlp = spacy.load(model)
            self._has_dep = True  # full model supports dep_ parsing for negation

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def count(self, text: str) -> CountResult:
        """Run the full Kalshi pipeline on a transcript string."""
        doc = self.nlp(text)
        tokens = list(doc)
        sentences = list(doc.sents)

        counts = self._count_tokens(tokens)

        # Merge alternative counts back to the original "A / B / C" label
        merged: defaultdict = defaultdict(int)
        for alt, n in counts.items():
            merged[self._alt_map.get(alt, alt)] += n

        negations = self._find_negations(sentences, merged) if self._has_dep else []
        top_sentences = self._score_sentences(sentences)

        return CountResult(
            counts=dict(merged),
            negations=negations,
            top_sentences=top_sentences,
            total_tokens=len(tokens),
            total_sentences=len(sentences),
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _token_matches(self, token, target: str) -> bool:
        """
        Returns True if this token is a Kalshi-valid match for `target`.

        Handles:
          - exact match
          - plural: target + s / target + es
          - possessive: target + 's
          - open/hyphenated compound: adjacent hyphen means the surrounding
            tokens are checked — NOT this method (see _count_tokens)
        """
        lower = token.text.lower()

        # Contractions (e.g. "n't", "wo") — never match
        if token.text.startswith("'") or token.text in ("n't", "wo", "ca", "'s"):
            return False

        # Exact
        if lower == target:
            return True
        # Plural -s  (tariff → tariffs)
        if lower == target + "s":
            return True
        # Plural -es  (border → borders already caught above, but also:)
        if lower == target + "es":
            return True
        # Plural -ies for targets ending in -y  (economy → economies)
        if target.endswith("y") and lower == target[:-1] + "ies":
            return True
        # Possessive 's  (spaCy blank keeps "America" + "'s" as two tokens)
        # The "'s" token is separate, so the base token itself matches "america"
        # (already covered by exact match on the base token)

        return False

    def _count_tokens(self, tokens: list) -> defaultdict:
        counts = defaultdict(int)
        n = len(tokens)

        i = 0
        while i < n:
            token = tokens[i]

            # --- Hyphenated open compound check ---
            # Pattern: word - word  (three consecutive tokens)
            # e.g. "pro", "-", "Palestine"
            # Rule: if the middle token is "-", check whether either side
            # matches a target. Closed compounds ("firefighter") are a single
            # token and won't appear here.
            # We only fire when i points at the "-" token to avoid
            # the flanking word tokens also matching in the normal path below.
            if (
                token.text == "-"
                and i > 0
                and i < n - 1
            ):
                left  = tokens[i - 1].text.lower()
                right = tokens[i + 1].text.lower()
                matched = set()
                for target in self.single_targets:
                    if (left == target or right == target) and target not in matched:
                        counts[target] += 1
                        matched.add(target)
                # Skip the right-side token so it isn't double-counted below
                i += 2
                continue

            # Skip a token that was already consumed as the LEFT side of a hyphen
            # (the next token is "-")
            if i + 1 < n and tokens[i + 1].text == "-":
                i += 1
                continue

            # --- Single-token matching ---
            for target in self.single_targets:
                if self._token_matches(token, target):
                    counts[target] += 1

            i += 1

        # --- Multi-word phrase matching (sliding window) ---
        words = [t.text.lower() for t in tokens]
        for phrase in self.phrase_targets:
            phrase_tokens = phrase.split()
            pn = len(phrase_tokens)
            for j in range(len(words) - pn + 1):
                if words[j : j + pn] == phrase_tokens:
                    counts[phrase] += 1

        return counts

    def _find_negations(self, sentences, counts: defaultdict) -> list[dict]:
        """
        Find sentences where a ROOT verb is negated AND contains a target word.
        Only available when using a full spaCy model (dep_ parsing required).
        """
        hit_targets = set(counts.keys())
        results = []

        for sent in sentences:
            sent_text_lower = sent.text.lower()
            # Check if sentence contains any counted target
            affected = [t for t in hit_targets if t in sent_text_lower]
            if not affected:
                continue

            # Check for negated ROOT verb
            for token in sent:
                if token.dep_ == "ROOT":
                    neg_children = [c for c in token.children if c.dep_ == "neg"]
                    if neg_children:
                        results.append({
                            "sentence": sent.text.strip(),
                            "negated_verb": token.text,
                            "keywords_affected": affected,
                        })
                        break

        return results

    def _score_sentences(self, sentences) -> list[dict]:
        """Score sentences by weighted keyword density and return top N."""
        scored = []

        for sent in sentences:
            lower = sent.text.lower()
            score = 0
            hits = []

            for target in self.single_targets | set(self.phrase_targets):
                if target in lower:
                    w = self.weights.get(target, 1)
                    score += w
                    hits.append(target)

            if score > 0:
                scored.append({
                    "score": score,
                    "sentence": sent.text.strip(),
                    "keywords": hits,
                })

        scored.sort(key=lambda x: -x["score"])
        return scored[: self.top_n]


# ---------------------------------------------------------------------------
# CLI — run directly on a text file or stdin
# ---------------------------------------------------------------------------
def _print_results(result: CountResult, label: str = ""):
    width = 55
    print("=" * width)
    print(f"  KALSHI WORD COUNTER{' — ' + label if label else ''}")
    print(f"  Tokens: {result.total_tokens:,}  |  Sentences: {result.total_sentences:,}")
    print("=" * width)

    if not result.counts:
        print("  No target words found.")
        return

    print(f"\n  {'Word':<22} {'Count':>6}  Bar")
    print("  " + "-" * 42)
    for word, count in sorted(result.counts.items(), key=lambda x: -x[1]):
        bar = "█" * min(count, 30)
        print(f"  {word:<22} {count:>6}  {bar}")

    absent_targets = []  # filled externally if needed
    print(f"\n  Total mentions : {sum(result.counts.values())}")
    print(f"  Unique targets : {len(result.counts)}")

    if result.negations:
        print(f"\n  {'NEGATION ALERTS':^{width - 4}}")
        print("  " + "-" * 42)
        for neg in result.negations:
            kw = ", ".join(neg["keywords_affected"])
            print(f"  [{neg['negated_verb']}] {neg['sentence'][:80]}...")
            print(f"       → keywords affected: {kw}")

    if result.top_sentences:
        print(f"\n  TOP SIGNAL SENTENCES")
        print("  " + "-" * 42)
        for i, s in enumerate(result.top_sentences[:5], 1):
            kw = ", ".join(s["keywords"])
            print(f"  #{i} score={s['score']}  [{kw}]")
            print(f"     {s['sentence'][:100]}")

    print()


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        with open(sys.argv[1], "r") as f:
            text = f.read()
        label = sys.argv[1]
    else:
        print("Reading from stdin (Ctrl+D to finish)...")
        text = sys.stdin.read()
        label = "stdin"

    # Optional: pass custom targets as comma-separated second arg
    targets = None
    if len(sys.argv) > 2:
        targets = [t.strip() for t in sys.argv[2].split(",")]

    counter = KalshiCounter(targets=targets)
    result = counter.count(text)
    _print_results(result, label)
