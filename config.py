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
# alpha=2.5, beta=1.5 encode a weak prior centered at ~63%.
# Effective prior sample size = alpha + beta = 4 (very weak).
# VALIDATED from backtest optimization.
PRIOR_ALPHA = 4.0
PRIOR_BETA = 0.5

# Correlation discount: reviews of the same movie aren't independent.
# If 10 critics see the same film, effective sample size < 10.
# 0.7 means effective_N = 0.7 * weighted_N.
# UNVALIDATED -- should be estimated from review variance analysis.
CORRELATION_DISCOUNT = 0.6

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

# Minimum edge (model prob - market price) to consider a trade.
# Below this, transaction costs and model uncertainty eat the edge.
# UNVALIDATED -- should be set based on backtest P&L by edge bucket.
MIN_EDGE = 0.05

# Minimum model win probability to consider a trade.
# Filters out long-shot bets where model is very uncertain.
# UNVALIDATED -- should be validated from backtest.
MIN_WIN_PROB = 0.25



# --- Confidence-based sizing multipliers ---
# Kelly is scaled by these multipliers based on model confidence.
# HIGH confidence = plenty of known critics, trust the model more.
# LOW confidence = few reviews or unknown critics, discount heavily.
# UNVALIDATED -- should be fitted from backtest P&L by confidence bucket.
CONFIDENCE_KELLY_MULTIPLIER = {
    "HIGH": 1.0,
    "MEDIUM": 0.6,
    "LOW": 0.3,
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
MAX_EVENT_EXPOSURE_PCT = 0.10
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
