"""
Review completion estimator.

Estimates what fraction of total expected reviews are already in,
and how the settlement close date affects score stability. This
information is used to widen or narrow confidence intervals:
- A movie with 30 of expected 200 reviews has lots of uncertainty remaining
- A movie with 100 of expected 110 reviews is basically locked in

Also estimates the expected direction of drift: early reviews may
systematically differ from late reviews, which matters for predicting
the settlement score.
"""
from datetime import datetime, timezone

# Expected total review counts by movie tier.
# Based on analysis of 3,591 movies in backtest dataset (May 2026).
# Kalshi only lists major releases, so we use the big-movie tiers.
EXPECTED_REVIEWS = {
    # (min_current_reviews, expected_total) -- stepped lookup
    # Franchise blockbusters (Star Wars, Marvel, etc.) get 150-250
    # Major studio releases get 80-150
    # Smaller wide releases get 50-100
    "franchise_blockbuster": 200,  # Star Wars, Marvel, DC, etc.
    "major_studio": 120,
    "wide_release": 90,
    "limited": 50,
}

# Franchise keywords that indicate a blockbuster-tier movie
FRANCHISE_KEYWORDS = {
    "star wars", "marvel", "avengers", "spider-man", "spider man",
    "batman", "superman", "dc", "x-men", "fantastic four",
    "mission impossible", "fast furious", "jurassic", "transformers",
    "harry potter", "lord of the rings", "dune", "pirates",
    "james bond", "007", "indiana jones", "toy story", "pixar",
    "frozen", "minions", "despicable me", "shrek", "kung fu panda",
    "predator", "alien", "terminator", "ghostbusters",
}

# Genre-based expected review counts
GENRE_REVIEW_EXPECTATIONS = {
    "Action": 120,
    "Adventure": 120,
    "Sci-Fi": 110,
    "Animation": 100,
    "Comedy": 90,
    "Drama": 85,
    "Horror": 80,
    "Thriller": 85,
    "Documentary": 50,
    "Musical": 80,
    "Romance": 75,
}


def estimate_completion(n_reviews, movie_title=None, genres=None, close_time=None):
    """
    Estimate what fraction of total reviews are in and the expected final count.

    Returns dict with:
        expected_total: estimated final review count
        completion_pct: fraction of reviews already in (0.0 to 1.0)
        days_to_close: days until market closes (None if unknown)
        score_stability: 0.0 (very unstable) to 1.0 (locked in)
    """
    # Estimate expected total reviews
    expected = _estimate_expected_total(n_reviews, movie_title, genres)

    completion = min(1.0, n_reviews / expected) if expected > 0 else 1.0

    # Time-based stability: closer to settlement = more stable
    days_to_close = None
    time_stability = 0.5  # default
    if close_time:
        now = datetime.now(timezone.utc)
        if isinstance(close_time, str):
            try:
                close_time = datetime.fromisoformat(close_time.replace("Z", "+00:00"))
            except ValueError:
                close_time = None

        if close_time:
            delta = (close_time - now).total_seconds() / 86400
            days_to_close = max(0, delta)
            # Within 1 day of close = very stable, 7+ days = less stable
            time_stability = min(1.0, max(0.2, 1.0 - days_to_close / 10.0))

    # Combined score stability: weighted average of review completion and time
    # Review completion matters more (0.7 weight) because it directly
    # determines how much the score can move
    score_stability = 0.7 * completion + 0.3 * time_stability

    return {
        "expected_total": expected,
        "completion_pct": round(completion, 3),
        "days_to_close": round(days_to_close, 1) if days_to_close is not None else None,
        "score_stability": round(score_stability, 3),
    }


def _estimate_expected_total(n_reviews, movie_title=None, genres=None):
    """Estimate how many total reviews a movie will receive."""
    # Check if it's a franchise blockbuster
    if movie_title:
        title_lower = movie_title.lower()
        for keyword in FRANCHISE_KEYWORDS:
            if keyword in title_lower:
                return max(EXPECTED_REVIEWS["franchise_blockbuster"], n_reviews * 1.1)

    # Genre-based estimate
    if genres:
        genre_estimates = [GENRE_REVIEW_EXPECTATIONS.get(g, 80) for g in genres]
        genre_avg = sum(genre_estimates) / len(genre_estimates)
    else:
        genre_avg = 90  # default for unknown genre

    # If current reviews already exceed our genre estimate, it's a bigger movie
    # than typical for its genre. Scale up.
    if n_reviews > genre_avg:
        # Movie is already past our genre average -- extrapolate
        # At 80% of the way to completion, the estimate should be ~1.25x current
        return int(n_reviews * 1.3)
    elif n_reviews > genre_avg * 0.7:
        # Getting close to our genre estimate, probably slightly more to come
        return int(genre_avg * 1.1)
    else:
        return int(genre_avg)


def completion_adjusted_discount(n_reviews, base_discount, movie_title=None,
                                  genres=None, close_time=None):
    """
    Adjust the correlation discount based on review completion.

    When most reviews are in, the score is stable and we should trust
    the current weighted mean more (higher effective N = less discount).
    When few reviews are in, there's more uncertainty (keep the discount).

    Returns adjusted correlation discount (higher = less shrinkage = tighter CI).
    """
    comp = estimate_completion(n_reviews, movie_title, genres, close_time)
    stability = comp["score_stability"]

    # At stability=1.0 (fully complete), discount approaches 0.95 (nearly no shrinkage)
    # At stability=0.0 (very early), use base_discount as-is
    # Linear interpolation
    adjusted = base_discount + (0.95 - base_discount) * stability

    return min(0.95, adjusted)
