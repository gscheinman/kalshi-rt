"""
Review sentiment intensity scoring.

Converts binary Fresh/Rotten into a 0.0-1.0 intensity score using
numeric ratings when available, falling back to keyword analysis.
"""
import re

# Letter grade mapping (to 0-1 scale)
LETTER_GRADES = {
    "A+": 0.97, "A": 0.93, "A-": 0.90,
    "B+": 0.87, "B": 0.83, "B-": 0.80,
    "C+": 0.67, "C": 0.63, "C-": 0.60,
    "D+": 0.47, "D": 0.43, "D-": 0.40,
    "F": 0.15,
}

# Keywords for quote-based sentiment analysis
STRONG_POSITIVE = {
    "masterpiece", "stunning", "extraordinary", "brilliant", "best",
    "perfect", "transcendent", "flawless", "magnificent", "superb",
    "phenomenal", "riveting", "unforgettable", "astonishing", "triumph",
}
MILD_POSITIVE = {
    "solid", "enjoyable", "decent", "fine", "good", "pleasant",
    "entertaining", "charming", "engaging", "worthwhile", "likable",
    "fun", "appealing", "satisfying", "competent",
}
STRONG_NEGATIVE = {
    "disaster", "unwatchable", "terrible", "worst", "abysmal",
    "atrocious", "dreadful", "horrendous", "painful", "insufferable",
    "embarrassing", "trainwreck", "catastrophe", "grotesque", "unbearable",
}
MILD_NEGATIVE = {
    "disappointing", "mediocre", "forgettable", "bland", "dull",
    "uninspired", "lackluster", "tepid", "tiresome", "flat",
    "predictable", "generic", "unremarkable", "pedestrian", "weak",
}


def parse_numeric_rating(text):
    """
    Parse a numeric rating string into a 0.0-1.0 float.

    Handles formats like:
      "3/4" -> 0.75
      "B+" -> 0.87
      "3.5/5" -> 0.7
      "7/10" -> 0.7
      star characters -> count / 5

    Returns None if the text can't be parsed.
    """
    if not text:
        return None

    text = text.strip()

    # Letter grades (case-insensitive)
    upper = text.upper().strip()
    if upper in LETTER_GRADES:
        return LETTER_GRADES[upper]

    # Star characters: count full and half stars out of 5
    if "★" in text or "☆" in text or "⭐" in text or "½" in text:
        full = text.count("★") + text.count("⭐")  # filled stars
        half = text.count("½")  # half symbol
        total = full + half * 0.5
        if total > 0:
            return min(1.0, total / 5.0)

    # Fraction pattern: "N/D" or "N.N/D"
    frac_match = re.search(r"(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)", text)
    if frac_match:
        num = float(frac_match.group(1))
        denom = float(frac_match.group(2))
        if denom > 0:
            return min(1.0, num / denom)

    # Plain number out of 10 (e.g., "7" or "7.5" when no denominator)
    plain_match = re.match(r"^(\d+(?:\.\d+)?)$", text)
    if plain_match:
        val = float(plain_match.group(1))
        if val <= 10:
            return val / 10.0

    return None


def analyze_quote(quote, is_fresh):
    """
    Lightweight keyword-based sentiment scoring of a review quote.

    Returns float 0.0 (harsh rotten) to 1.0 (glowing fresh).
    Uses the is_fresh flag as a baseline anchor.
    """
    if not quote:
        return 0.65 if is_fresh else 0.35

    words = set(re.findall(r"[a-z]+", quote.lower()))

    strong_pos = len(words & STRONG_POSITIVE)
    mild_pos = len(words & MILD_POSITIVE)
    strong_neg = len(words & STRONG_NEGATIVE)
    mild_neg = len(words & MILD_NEGATIVE)

    # Compute a score shift from baseline
    pos_signal = strong_pos * 0.12 + mild_pos * 0.05
    neg_signal = strong_neg * 0.12 + mild_neg * 0.05

    if is_fresh:
        base = 0.65
        score = base + pos_signal - neg_signal
    else:
        base = 0.35
        score = base - neg_signal + pos_signal * 0.5  # positive words in rotten review shift less

    return max(0.0, min(1.0, score))


def score_intensity(review):
    """
    Score review intensity from 0.0 (harsh rotten) to 1.0 (glowing fresh).

    Uses numeric rating if available, falls back to keyword analysis of quote.
    """
    is_fresh = review.get("sentiment") == "Fresh"

    # Try numeric rating first
    rating_text = review.get("rating_text", "")
    numeric = parse_numeric_rating(rating_text)
    if numeric is not None:
        return numeric

    # Fall back to quote analysis
    quote = review.get("quote", "")
    if quote:
        return analyze_quote(quote, is_fresh)

    # Default based on sentiment
    return 0.65 if is_fresh else 0.35
