"""
Calibration module. Applies Platt scaling to raw model probabilities
based on backtested performance at different review counts.

The calibration params below were derived from the Stefanoleone992 Kaggle dataset
(1.13M reviews, 8,997 movies with 30+ reviews). They correct for systematic
over/under-confidence at different sample sizes.

To regenerate: run calibrate.py against the historical dataset.
"""
import math

# Platt scaling params: (a, b) per review-count bucket
# calibrated_p = 1 / (1 + exp(-(a * logit(raw_p) + b)))
#
# Refit 2026-05-21 under corrected prior (PRIOR_ALPHA=2.5, PRIOR_BETA=1.5).
# The previous params were fit against the broken 88.9%-mean prior and were
# compressing already-overconfident probabilities back toward 50%. With the
# prior fixed, the raw model is closer to correctly calibrated and only
# needs mild adjustment -- 'a' jumped from ~0.5 to ~0.7-1.05.
#
# Sample counts per bucket (from backtest/optimize.py --param platt):
#   N=5:  70,813    N=10: 34,656    N=15: 21,299
#   N=20: 15,257    N=25: 11,381    N=30:  9,177
#
# UNVALIDATED against Kalshi outcomes -- needs 20+ resolved markets to confirm.
PLATT_PARAMS = {
    5:  (0.6878, 0.7135),
    10:  (0.7775, 0.5675),
    15:  (0.9493, 0.3353),
    20:  (0.9692, 0.2684),
    25:  (0.9882, 0.2065),
    30:  (1.0545, 0.1510),
}


def get_params_for_n(n_reviews):
    """Get calibration params for the closest review-count bucket."""
    buckets = sorted(PLATT_PARAMS.keys())
    for b in reversed(buckets):
        if n_reviews >= b:
            return PLATT_PARAMS[b]
    return PLATT_PARAMS[buckets[0]]


def calibrate_probability(raw_p, n_reviews):
    """Apply Platt scaling to a raw model probability."""
    raw_p = max(0.001, min(0.999, raw_p))
    a, b = get_params_for_n(n_reviews)
    logit = math.log(raw_p / (1 - raw_p))
    scaled_logit = a * logit + b
    return 1.0 / (1.0 + math.exp(-scaled_logit))


def calibrate_thresholds(threshold_probs, n_reviews):
    """Calibrate all threshold probabilities."""
    return {
        t: calibrate_probability(p, n_reviews)
        for t, p in threshold_probs.items()
        if p is not None
    }
