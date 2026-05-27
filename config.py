"""
Central configuration for all model parameters.

Every number here must have a justification. If marked UNVALIDATED,
it needs backtesting before real money is deployed.
"""

# --- Model: Beta-binomial distribution ---

# Population mean tomatometer across all movies.
# Source: Kaggle RT dataset, ~9000 movies. VALIDATED.
POPULATION_MEAN = 0.63

# Prior parameters for Beta distribution.
# alpha=2.5, beta=1.5 encode a weak prior centered at ~62.5%
# (alpha / (alpha + beta) = 2.5/4.0 = 0.625), matching POPULATION_MEAN.
# Effective prior sample size = alpha + beta = 4 (very weak).
#
# NOTE: previous values (4.0, 0.5) implied a prior mean of 88.9% which
# systematically inflated low-N predictions. Reverted 2026-05-20 after
# calibration audit showed 30+ percentage point overconfidence in the
# 35-65% bucket. The grid-search "optimization" that produced 4.0/0.5
# was fitting Kaggle's high-coverage tail, not Kalshi-relevant outcomes.
# UNVALIDATED against Kalshi resolutions.
PRIOR_ALPHA = 2.5
PRIOR_BETA = 1.5

# Correlation discount: reviews of the same movie aren't independent.
# If 10 critics see the same film, effective sample size < 10.
# 0.7 means effective_N = 0.7 * weighted_N.
# UNVALIDATED -- should be estimated from review variance analysis.
CORRELATION_DISCOUNT = 0.6

# Drift adjustment: applies a predicted-final-score blend to the raw model
# mean. DISABLED 2026-05-26 after audit showed the formula has a structural
# bias toward 50% (any score >50% gets pulled down, any <50% pulled up).
#
# Empirical test: at naive=85% the formula predicts the remaining critics
# will Fresh-rate at 68.8%, dragging the score down by 8.8pp. That's not
# modelling drift -- it's mean reversion masquerading as drift.
#
# Real-world cost: Power Ballad (87.9% naive) was predicted at 76-83%
# by the model and we shorted it via BUY NO Above 80. Score has actually
# climbed to 89%. The drift mechanism was -100% wrong in direction.
#
# Re-enable only after we have a corrected formula (likely keyed on
# top-critic-vs-non-top differential, not raw agreement rate) AND
# 10+ resolved Kalshi markets confirming it improves out-of-sample MAE.
DRIFT_ADJUSTMENT_ENABLED = False

# Mean reversion: how quickly we trust reviews over the population mean.
# At N reviews, shrinkage = min(1.0, N / SHRINKAGE_N).
# At N=30, we fully trust the review signal.
# UNVALIDATED -- should be fitted from backtest by review count bucket.
SHRINKAGE_N = 15

# Kalshi threshold brackets we model.
# Base set: every 5% from 5 to 95. Kalshi also creates granular thresholds
# (47%, 52%, 56-59%, 62%, etc.) for popular movies. Those are added dynamically
# by predict_distribution when market data is available.
KALSHI_THRESHOLDS = list(range(5, 100, 5))


# --- Sizing: Kelly criterion ---

# Fraction of full Kelly to use. 0.25 = quarter Kelly.
# Quarter Kelly gives ~50% of full Kelly's return with ~25% of the variance.
# VALIDATED (standard quant practice).
KELLY_FRACTION = 0.25

# Maximum fraction of bankroll per single position.
# Even if Kelly says bet more, never exceed this.
# VALIDATED (standard risk management).
MAX_POSITION_PCT = 0.05

# Default bankroll for sizing calculations.
DEFAULT_BANKROLL = 1000

# Fallback minimum edge if no tiered rule applies.
MIN_EDGE = 0.12

# Minimum model win probability to consider a trade.
# Tightened 2026-05-20 from 0.25 to 0.40 to avoid acting on model output
# in the most miscalibrated probability zone.
MIN_WIN_PROB = 0.40

# Tiered minimum edge by review count at entry.
# Hypothesis: edge mostly lives in the 5-40 review window. Higher review
# counts have less informational asymmetry vs the market, so we demand
# more edge to take a trade there. Tier is selected by review_count.
# Format: list of (min_reviews, max_reviews, min_edge). First match wins.
# trade_enabled=False on the 0-5 bucket disables that zone entirely.
MIN_EDGE_BY_REVIEW_COUNT = [
    # (lo, hi, min_edge, trade_enabled)
    (0,    5,   0.99, False),  # too noisy, skip
    (5,   40,   0.12, True),   # prime hunting ground -- start tight, loosen after
                               # 10+ resolutions show real edge at 12% (then try 10%, then 8%)
    (40,  80,   0.15, True),   # possible edge, tighter
    (80,  9999, 0.20, True),   # skeptical zone, only fire on huge edges
]

# Volume-based tightening: on top of the review-count tier, demand even
# more edge if the market is thick enough that consensus is well-informed.
HIGH_VOLUME_THRESHOLD = 50_000
HIGH_VOLUME_MIN_EDGE_BUMP = 0.05  # adds 5pp to the review-count tier's min_edge

# Sanity guard: extreme claimed edge is almost certainly model error.
# Graded by market liquidity -- the more money in the market, the more
# the consensus knows, so we demand even smaller "edges" before flagging
# the claim as a model bug.
# Format: (min_volume, max_edge_allowed). First match wins; higher
# volume -> tighter cap. Set max_edge to 0 to disable a tier.
SANITY_GRADED = [
    (50_000, 0.10),   # thick markets: anything over 10pp is suspect
    (5_000,  0.15),   # mid liquidity: 15pp cap
    (0,      0.20),   # thin markets: 20pp cap (rare model errors only)
]

# Legacy single-threshold sanity check (kept for backwards compatibility,
# but SANITY_GRADED takes precedence).
SANITY_MAX_EDGE_ON_LIQUID = 0.15
SANITY_LIQUID_VOLUME_MIN = 10_000


def _coerce_float(v, default=0.0):
    """Kalshi sometimes returns numbers as strings. Coerce safely."""
    if v is None:
        return default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def min_edge_for(review_count, volume=0):
    """Return (min_edge, trade_enabled) for a given review count + volume.

    Used by alpha/portfolio engines to enforce tiered filtering.
    Falls back to (1.0, False) for malformed input -- fail safe, no trade.
    """
    rc = int(_coerce_float(review_count))
    vol = _coerce_float(volume)
    for lo, hi, edge, enabled in MIN_EDGE_BY_REVIEW_COUNT:
        if lo <= rc < hi:
            if vol and vol >= HIGH_VOLUME_THRESHOLD:
                edge = edge + HIGH_VOLUME_MIN_EDGE_BUMP
            return edge, enabled
    # Should never hit this -- tier table covers 0-9999. Fail safe if it does.
    return 1.0, False


def sanity_blocks(edge, volume):
    """Return True if the claimed edge should be blocked as likely model error.

    Graded by liquidity: thicker markets get tighter edge caps because the
    consensus has incorporated more information.
    """
    edge = _coerce_float(edge, default=-1)
    if edge < 0:
        return False
    vol = _coerce_float(volume)
    for vol_floor, max_edge in SANITY_GRADED:
        if vol >= vol_floor:
            if max_edge > 0 and edge > max_edge:
                return True
            return False
    return False



# --- Confidence-based sizing multipliers ---
# Kelly is scaled by these multipliers based on model confidence.
# HIGH confidence = plenty of known critics, trust the model more.
# LOW confidence = few reviews or unknown critics, discount heavily.
# UNVALIDATED -- should be fitted from backtest P&L by confidence bucket.
CONFIDENCE_KELLY_MULTIPLIER = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.0,    # disabled 2026-05-20: too few critics known to trust the model
    "NONE": 0.0,
}

# --- Time-to-settlement sizing ---
# As settlement approaches, model uncertainty decreases (more reviews in).
# Scale position size based on days until Monday 10 AM ET resolution.
SETTLEMENT_SIZING = {
    "max_multiplier": 1.3,
    "base_multiplier": 0.7,
    "full_confidence_days": 1,
    "base_confidence_days": 7,
}

# --- Portfolio risk limits ---
# Tightened 2026-05-20 from 0.10 to 0.05 per event while calibration is broken.
MAX_EVENT_EXPOSURE_PCT = 0.05
MAX_TOTAL_EXPOSURE_PCT = 0.30


# --- Fees ---

# Kalshi fee on profits (not on principal).
# Source: Kalshi fee schedule, confirmed against live trades.
# VALIDATED.
KALSHI_FEE_RATE = 0.07


# --- Confidence levels ---

# Thresholds for confidence classification.
# HIGH: enough data that predictions should be reliable.
# MEDIUM: moderate data, some uncertainty.
# LOW: few reviews or mostly unknown critics.
CONFIDENCE_HIGH_MIN_REVIEWS = 20
CONFIDENCE_HIGH_MIN_KNOWN_PCT = 0.6
CONFIDENCE_MEDIUM_MIN_REVIEWS = 10
CONFIDENCE_MEDIUM_MIN_KNOWN_PCT = 0.4


# --- Scraping ---

# Rate limiting delays (seconds).
RT_SCRAPE_DELAY = 0.3
KALSHI_RATE_LIMIT_DELAY = 0.15

# Max review pages to fetch per movie (20 reviews per page).
MAX_REVIEW_PAGES = 5


# --- Critic database ---

# Minimum reviews for a critic to be included in the database.
MIN_CRITIC_REVIEWS = 5

# Fuzzy match threshold for critic name lookup.
FUZZY_MATCH_THRESHOLD = 0.85

# "Big movie" = movie with this many critics in our dataset.
# Kalshi only lists major releases, so a critic's accuracy on big movies
# is more predictive than their overall agreement rate.
# 80 = roughly the threshold where a movie is a wide release with broad coverage.
BIG_MOVIE_REVIEW_THRESHOLD = 80

# Which weight function to use for critic reviews.
# Options: "agreement_60", "spread", "big_spread", "cross_threshold_acc",
#          "inv_mae", "big_cross_threshold_acc", "spread_x_accuracy"
# VALIDATED -- spread wins on Brier score (0.1373 vs 0.1388 for agreement_60).
# Brier matters more than MAE for threshold betting calibration.
CRITIC_WEIGHT_FUNCTION = "spread"
