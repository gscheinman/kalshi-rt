"""
Movie-specific prior estimation.

Replaces the flat 63% population mean with a genre/franchise/rating-adjusted
baseline. Kalshi only lists major releases, so this prior should reflect
the distribution of big-movie scores, not all movies.

Genre base rates from Kaggle RT dataset (movies with 50+ reviews):
  Action/Adventure: ~58%     Animation: ~65%
  Comedy: ~55%               Drama: ~67%
  Horror: ~59%               Sci-Fi: ~56%
  Romance: ~62%              Documentary: ~85%
  Musical: ~68%              Thriller: ~60%

Franchise/sequel penalty: sequels average ~5% lower than originals.
Rating adjustment: R-rated films average ~2% higher than PG-13.

These are UNVALIDATED estimates from aggregate data. They'll be replaced
with fitted values once we have enough backtest results.
"""
import re
import config

GENRE_BASE_RATES = {
    "action": 0.58,
    "adventure": 0.58,
    "animation": 0.65,
    "comedy": 0.55,
    "drama": 0.67,
    "horror": 0.59,
    "sci-fi": 0.56,
    "science fiction": 0.56,
    "romance": 0.62,
    "documentary": 0.85,
    "musical": 0.68,
    "mystery & thriller": 0.60,
    "thriller": 0.60,
    "mystery": 0.60,
    "fantasy": 0.58,
    "kids & family": 0.62,
    "war": 0.64,
    "western": 0.60,
    "biography": 0.70,
    "history": 0.68,
    "crime": 0.62,
    "sport": 0.60,
    "music": 0.66,
}

SEQUEL_INDICATORS = [
    r"\b(part|chapter|vol\.?|volume)\s*\d",
    r"\b[IVX]{2,}\b",
    r"\b\d{1,2}\b(?!.*\d{4})",
    r":\s*(the\s+)?(final|last|next|new|rise|return|revenge|resurrection|awakening|legacy|reckoning)",
    r"\b(sequel|reboot|remake)\b",
]

FRANCHISE_KEYWORDS = [
    "mission: impossible", "fast", "furious", "avengers", "spider-man",
    "batman", "superman", "x-men", "transformers", "jurassic", "star wars",
    "harry potter", "lord of the rings", "pirates of the caribbean",
    "john wick", "deadpool", "guardians", "thor", "captain", "ant-man",
    "black panther", "doctor strange", "shang-chi", "eternals",
    "fantastic four", "blade", "thunderbolts",
]

SEQUEL_PENALTY = -0.05
R_RATED_ADJUSTMENT = 0.02


def estimate_prior(movie_summary):
    """Estimate a movie-specific prior tomatometer probability.

    Args:
        movie_summary: dict from rt_page.get_movie_summary() with optional
                      genres, directors, content_rating, title fields.

    Returns:
        float between 0 and 1 representing the prior expected tomatometer.
    """
    if not movie_summary:
        return config.POPULATION_MEAN

    genres = movie_summary.get("genres", [])
    title = movie_summary.get("title", "")
    content_rating = movie_summary.get("content_rating", "")

    prior = _genre_prior(genres)

    if _is_sequel_or_franchise(title):
        prior += SEQUEL_PENALTY

    if content_rating == "R":
        prior += R_RATED_ADJUSTMENT
    elif content_rating == "G":
        prior -= 0.02

    return max(0.15, min(0.95, prior))


def _genre_prior(genres):
    """Compute weighted average of genre base rates."""
    if not genres:
        return config.POPULATION_MEAN

    rates = []
    for g in genres:
        key = g.lower().strip()
        if key in GENRE_BASE_RATES:
            rates.append(GENRE_BASE_RATES[key])

    if not rates:
        return config.POPULATION_MEAN

    return sum(rates) / len(rates)


def _is_sequel_or_franchise(title):
    """Detect if a movie is likely a sequel or franchise entry."""
    if not title:
        return False

    title_lower = title.lower()

    for kw in FRANCHISE_KEYWORDS:
        if kw in title_lower:
            return True

    for pattern in SEQUEL_INDICATORS:
        if re.search(pattern, title, re.IGNORECASE):
            return True

    return False
