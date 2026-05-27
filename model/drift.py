"""
Score drift prediction from remaining reviewer pool.

When early reviews are in but more are expected, predicts which direction
the tomatometer will drift based on:
1. Who has already reviewed (and their tendencies)
2. Who is likely to still review (big-movie critics who haven't weighed in)
3. The current score's distance from 50% (borderline movies drift more)

This is a key edge over the market: the market sees the current tomatometer,
but we can model the expected FUTURE tomatometer by knowing the reviewer pool.
"""
from data.critics import CriticDatabase


def predict_drift(reviews, critic_db, current_score, expected_remaining=None):
    """
    Predict score drift from remaining reviewers.

    Args:
        reviews: list of review dicts (already-submitted reviews)
        critic_db: CriticDatabase instance
        current_score: current tomatometer as 0-100 float
        expected_remaining: estimated number of remaining reviews (optional)

    Returns dict with:
        expected_remaining_fresh_rate: predicted fresh rate of future reviews (0-1)
        predicted_drift_pp: predicted drift in percentage points
        predicted_final_score: predicted final tomatometer
        n_pool: number of big-movie critics in the remaining pool
        confidence: how much to trust this drift estimate
    """
    if not reviews or current_score is None:
        return _empty_drift()

    quality = current_score / 100.0  # normalize to 0-1

    # Build set of critics who have already reviewed
    reviewed_names = set()
    for r in reviews:
        reviewed_names.add(r.get("critic_name", "").lower())

    # Find critics likely to review this movie who haven't yet.
    # Two tiers:
    #   Tier 1: Critics with big_movie_agreement_rate (they've reviewed big movies)
    #   Tier 2: Top critics or high-volume critics (50+ reviews) -- likely to review
    #           major releases even if we don't have their big-movie stats
    remaining_pool = []
    for name in critic_db._names:
        if name.lower() in reviewed_names:
            continue
        critic = critic_db.get_critic(name)
        if not critic:
            continue

        # Tier 1: has big-movie agreement rate (best signal)
        big_agree = critic.get("big_movie_agreement_rate")
        if big_agree is not None:
            remaining_pool.append({
                "name": name,
                "agreement_rate": big_agree,
                "top_critic": critic.get("top_critic", False),
                "total_reviews": critic.get("total_reviews", 0),
            })
            continue

        # Tier 2: top critic or prolific reviewer (likely reviews big movies)
        is_top = critic.get("top_critic", False)
        total = critic.get("total_reviews", 0)
        agree_rate = critic.get("agreement_rate")
        if agree_rate is not None and (is_top or total >= 50):
            remaining_pool.append({
                "name": name,
                "agreement_rate": agree_rate,
                "top_critic": is_top,
                "total_reviews": total,
            })

    if not remaining_pool:
        return _empty_drift()

    # Predict each remaining critic's fresh probability
    # P(Fresh | movie_quality=q, agreement_rate=a) = q*a + (1-q)*(1-a)
    # This is: "agree with consensus" where consensus = Fresh if q > 0.5
    fresh_probs = []
    for critic in remaining_pool:
        a = critic["agreement_rate"]
        p_fresh = quality * a + (1.0 - quality) * (1.0 - a)
        fresh_probs.append(p_fresh)

    expected_fresh_rate = sum(fresh_probs) / len(fresh_probs)

    # Estimate remaining review count if not provided
    if expected_remaining is None:
        # Heuristic: big movies get ~100-150 more reviews after the first batch
        # Scale by how many big-movie critics are in the pool
        expected_remaining = min(len(remaining_pool), 120)

    # Predict final score
    current_fresh = round(quality * len(reviews))
    current_total = len(reviews)
    new_fresh = expected_remaining * expected_fresh_rate
    predicted_final = (current_fresh + new_fresh) / (current_total + expected_remaining) * 100

    drift_pp = predicted_final - current_score

    # Confidence in drift estimate
    # Higher when: more big-movie critics in pool, bigger expected remaining count
    pool_confidence = min(1.0, len(remaining_pool) / 200)  # saturates at 200 critics
    volume_confidence = min(1.0, expected_remaining / 50)  # saturates at 50 remaining
    confidence = pool_confidence * 0.6 + volume_confidence * 0.4

    return {
        "expected_remaining_fresh_rate": round(expected_fresh_rate, 4),
        "predicted_drift_pp": round(drift_pp, 2),
        "predicted_final_score": round(predicted_final, 1),
        "n_pool": len(remaining_pool),
        "n_reviewed": len(reviews),
        "expected_remaining": expected_remaining,
        "confidence": round(confidence, 3),
    }


def drift_adjusted_mean(raw_mean, drift_info, n_reviews):
    """
    Blend the raw model mean with the drift prediction.

    CURRENTLY DISABLED via config.DRIFT_ADJUSTMENT_ENABLED. The underlying
    formula in predict_drift() has a structural bias toward 50%: at any
    score >50% it predicts downward drift; at any score <50% it predicts
    upward drift. That's mean reversion mislabeled as drift, not real signal.

    Specifically: p_fresh = q*a + (1-q)*(1-a) where q is naive Fresh rate
    and a is critic agreement rate. With typical a~0.65, this formula
    sends p_fresh toward 50% regardless of q.

    The function is preserved for diagnostic logging (we still record
    drift_info on each snapshot). It just doesn't influence the model
    output until we have a corrected formula validated against real
    Kalshi outcomes.

    Returns raw_mean unchanged when DRIFT_ADJUSTMENT_ENABLED is False.
    """
    import config
    if not config.DRIFT_ADJUSTMENT_ENABLED:
        return raw_mean

    if not drift_info or drift_info.get("confidence", 0) < 0.1:
        return raw_mean

    predicted_final = drift_info["predicted_final_score"] / 100.0
    drift_confidence = drift_info["confidence"]

    n_factor = max(0.05, 1.0 - n_reviews / 120.0)
    blend_weight = min(0.3, n_factor * drift_confidence * 0.4)

    adjusted = raw_mean * (1.0 - blend_weight) + predicted_final * blend_weight
    return adjusted


def _empty_drift():
    return {
        "expected_remaining_fresh_rate": None,
        "predicted_drift_pp": 0.0,
        "predicted_final_score": None,
        "n_pool": 0,
        "n_reviewed": 0,
        "expected_remaining": 0,
        "confidence": 0.0,
    }
