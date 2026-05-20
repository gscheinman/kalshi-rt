from scipy.stats import beta as beta_dist
from model.sentiment import score_intensity
from model.learned_weights import learned_predictor
from model.prior import estimate_prior
from model.review_completion import estimate_completion, completion_adjusted_discount
from model.drift import predict_drift, drift_adjusted_mean
import config


def predict_distribution(reviews, critic_db, movie_summary=None, close_time=None,
                         extra_thresholds=None):
    """
    Given early reviews, produce a Beta posterior distribution over the
    final Tomatometer score and threshold probabilities.

    Two paths:
      1. ML path (when trained model available): LightGBM predicts what each
         review implies about the final score. These per-review predictions
         are aggregated into the Beta posterior.
      2. Fallback path: hand-crafted agreement-rate weighting with calibration
         blending. Used when ML model hasn't been trained yet.

    Both paths feed into the same Beta-binomial framework for uncertainty
    quantification and threshold probability output.

    reviews: list of dicts with keys: critic_name, sentiment ("Fresh"/"Rotten"),
             optionally top_critic (bool), publication (str)
    critic_db: CriticDatabase instance

    Returns dict with model outputs.
    """
    if not reviews:
        return _empty_result()

    n = len(reviews)
    known_count = 0
    details = []

    # Extract movie metadata for completion estimation
    movie_title = movie_summary.get("title") if movie_summary else None
    genres = movie_summary.get("genres") if movie_summary else None

    # ML path disabled: showed systematic upward bias (+11.6pp on Star Wars
    # where naive=59.6% but ML predicted 71.2%). Needs retraining on real
    # Kalshi market outcomes before re-enabling.
    # TODO: Re-enable after accumulating 20+ resolved market snapshots and
    # retraining with market-aware loss function.
    ml_scores = None
    # if learned_predictor.available:
    #     ml_scores = learned_predictor.predict_review_scores(reviews, critic_db)

    if ml_scores:
        # ML path: use per-review score predictions
        weighted_score_sum = 0.0
        weight_sum = 0.0

        for i, r in enumerate(reviews):
            name = r["critic_name"]
            critic = critic_db.get_critic(name)
            if critic:
                known_count += 1

            pred_score, confidence = ml_scores[i]
            weighted_score_sum += confidence * pred_score
            weight_sum += confidence

            details.append({
                "critic_name": name,
                "sentiment": r["sentiment"],
                "publication": r.get("publication", ""),
                "weight": round(confidence, 3),
                "known": critic is not None,
                "tier": critic["tier"] if critic else "Unknown",
                "intensity": round(pred_score, 3),
                "implied_tomato": round(pred_score * 100, 1),
                "ml_predicted": True,
            })

        raw_mean = weighted_score_sum / weight_sum if weight_sum > 0 else config.POPULATION_MEAN
        effective_total = weight_sum * config.CORRELATION_DISCOUNT

    else:
        # Fallback path: hand-crafted weighting
        weighted_fresh = 0.0
        weighted_total = 0.0
        calibrated_estimates = []

        for r in reviews:
            name = r["critic_name"]
            is_fresh = r["sentiment"] == "Fresh"
            is_top = r.get("top_critic", False)
            weight = critic_db.get_weight(name, is_top_critic=is_top)
            critic = critic_db.get_critic(name)
            intensity = score_intensity(r)

            # Binary Fresh/Rotten with critic weight only.
            # Intensity is NOT used for the weighted tomatometer because it
            # conflates "review strength" with "probability" and systematically
            # biases the estimate (Fresh at 0.65 pulls the mean below naive).
            # Instead, weight differences between critics handle differential
            # trust, and the binary outcome matches what the tomatometer actually
            # measures.
            weighted_fresh += weight * (1.0 if is_fresh else 0.0)
            weighted_total += weight

            implied = critic_db.get_implied_tomatometer(
                name, r["sentiment"], r.get("rating_text", "")
            )
            if implied is not None:
                calibrated_estimates.append((weight, implied / 100.0))

            if critic:
                known_count += 1

            details.append({
                "critic_name": name,
                "sentiment": r["sentiment"],
                "publication": r.get("publication", ""),
                "weight": round(weight, 3),
                "known": critic is not None,
                "tier": critic["tier"] if critic else "Unknown",
                "intensity": round(intensity, 3),
                "implied_tomato": round(implied, 1) if implied is not None else None,
                "ml_predicted": False,
            })

        raw_mean = weighted_fresh / weighted_total if weighted_total > 0 else config.POPULATION_MEAN

        # Correlation discount adjusted by review completion.
        # When most expected reviews are in, the score is stable -> less discount.
        # When few reviews are in, more uncertainty -> keep the discount.
        corr_discount = completion_adjusted_discount(
            n, config.CORRELATION_DISCOUNT,
            movie_title=movie_title, genres=genres, close_time=close_time,
        )
        effective_total = weighted_total * corr_discount

        # Blend in per-critic calibrated estimates when available.
        # Scale blend down with review count: calibration matters most at low N,
        # but at high N the direct weighted count is more reliable.
        # At N=99, review_discount = max(0.1, 1 - 99/50) = 0.1 -> blend capped at 4%
        # At N=10, review_discount = max(0.1, 1 - 10/50) = 0.8 -> blend up to 32%
        if calibrated_estimates:
            cal_weight_sum = sum(w for w, _ in calibrated_estimates)
            cal_mean = sum(w * est for w, est in calibrated_estimates) / cal_weight_sum
            cal_coverage = len(calibrated_estimates) / n
            review_discount = max(0.1, 1.0 - n / 50.0)
            blend_weight = min(0.4, cal_coverage * 0.5) * review_discount
            raw_mean = raw_mean * (1 - blend_weight) + cal_mean * blend_weight

    # Mean reversion: shrink toward movie-specific prior at low review counts
    prior_mean = estimate_prior(movie_summary)
    shrinkage = min(1.0, n / config.SHRINKAGE_N)
    adjusted_mean = raw_mean * shrinkage + prior_mean * (1.0 - shrinkage)

    # Drift prediction: adjust mean based on expected remaining reviewers
    naive_pct_tmp = sum(1 for r in reviews if r["sentiment"] == "Fresh") / n * 100
    comp = estimate_completion(n, movie_title=movie_title, genres=genres, close_time=close_time)
    expected_remaining = max(0, comp["expected_total"] - n)
    drift_info = predict_drift(reviews, critic_db, naive_pct_tmp, expected_remaining=expected_remaining)
    if drift_info["confidence"] > 0.1 and comp["completion_pct"] < 0.85:
        adjusted_mean = drift_adjusted_mean(adjusted_mean, drift_info, n)

    # Beta posterior
    alpha = config.PRIOR_ALPHA + adjusted_mean * effective_total
    beta = config.PRIOR_BETA + (1.0 - adjusted_mean) * effective_total

    dist = beta_dist(alpha, beta)
    mean = dist.mean()
    ci_low, ci_high = dist.ppf(0.025), dist.ppf(0.975)

    # Threshold probabilities -- include both standard thresholds and any
    # extra thresholds from actual Kalshi markets (47%, 52%, 56-59%, 62%, etc.)
    all_thresholds = set(config.KALSHI_THRESHOLDS)
    if extra_thresholds:
        all_thresholds.update(extra_thresholds)
    threshold_probs = {}
    for t in sorted(all_thresholds):
        threshold_probs[t] = float(1.0 - dist.cdf(t / 100.0))

    naive_pct = naive_pct_tmp  # already computed above for drift
    known_pct = known_count / n if n > 0 else 0

    if n >= config.CONFIDENCE_HIGH_MIN_REVIEWS and known_pct >= config.CONFIDENCE_HIGH_MIN_KNOWN_PCT:
        confidence = "HIGH"
    elif n >= config.CONFIDENCE_MEDIUM_MIN_REVIEWS and known_pct >= config.CONFIDENCE_MEDIUM_MIN_KNOWN_PCT:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"

    # comp already computed above for drift

    return {
        "n_reviews": n,
        "n_fresh": sum(1 for r in reviews if r["sentiment"] == "Fresh"),
        "n_rotten": sum(1 for r in reviews if r["sentiment"] != "Fresh"),
        "n_known": known_count,
        "known_pct": round(known_pct * 100, 1),
        "naive_pct": round(naive_pct, 1),
        "model_mean": round(mean * 100, 1),
        "model_ci": [round(ci_low * 100, 1), round(ci_high * 100, 1)],
        "confidence": confidence,
        "threshold_probs": threshold_probs,
        "prior_mean": round(prior_mean * 100, 1),
        "alpha": alpha,
        "beta": beta,
        "corr_discount": round(corr_discount, 3),
        "review_completion": comp,
        "drift": drift_info,
        "critic_details": details,
    }


def _empty_result():
    probs = {t: None for t in config.KALSHI_THRESHOLDS}
    return {
        "n_reviews": 0, "n_fresh": 0, "n_rotten": 0, "n_known": 0,
        "known_pct": 0, "naive_pct": 0, "model_mean": config.POPULATION_MEAN * 100,
        "model_ci": [0, 100], "confidence": "NONE",
        "threshold_probs": probs, "alpha": config.PRIOR_ALPHA, "beta": config.PRIOR_BETA,
        "critic_details": [],
    }
