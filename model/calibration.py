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
# Fitted from backtest optimization
PLATT_PARAMS = {
    5:  (0.5395, 0.7375),
    10:  (0.4934, 0.649),
    15:  (0.4725, 0.5146),
    20:  (0.4938, 0.3656),
    25:  (0.5334, 0.091),
    30:  (0.5965, 0.0207),
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
