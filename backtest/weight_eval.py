"""
Empirical evaluation of critic weight functions.

The model weights each critic's review to build a prediction. The current
weight is "agreement rate" -- how often the critic's Fresh/Rotten matches
the 60% consensus threshold. But Kalshi markets span 25% to 95%, so
accuracy at the 60% line is only loosely relevant.

This module tests multiple candidate weight functions against historical
data to find which one produces the best predictions across ALL thresholds.

The winner is whichever weight function minimizes Brier score across all
Kalshi thresholds (25, 30, 35... 95) on held-out movie data.

Usage:
    python -m backtest.weight_eval
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

from scipy.stats import beta as beta_dist

from backtest.evaluate import load_movie_reviews
from backtest.metrics import brier_score, mean_absolute_error
from data.critics import CriticDatabase
from model.sentiment import score_intensity
import config

ROOT = Path(__file__).parent.parent
REVIEWS_PATH = ROOT / "critic_reviews.json"


def load_critic_calibration():
    """Load per-critic calibration data from the scraper output."""
    if not REVIEWS_PATH.exists():
        return {}
    with open(REVIEWS_PATH) as f:
        return json.load(f)


def compute_critic_metrics(critic_cal, movies):
    """For each critic, compute various candidate weight metrics from their
    historical review data.

    Returns dict: critic_name -> {metric_name: value}
    """
    # Build per-movie review count (how many critics reviewed each movie)
    movie_review_counts = defaultdict(int)
    for critic_name, cdata in critic_cal.items():
        for r in cdata.get("reviews", []):
            ems = r.get("movie_ems_id")
            if ems:
                movie_review_counts[ems] += 1

    metrics = {}
    for critic_name, cdata in critic_cal.items():
        reviews = cdata.get("reviews", [])
        if len(reviews) < 5:
            continue

        agreement_rate = cdata.get("agreement_rate")
        big_agreement = cdata.get("big_movie_agreement_rate")
        avg_fresh = cdata.get("avg_tomato_when_fresh")
        avg_rotten = cdata.get("avg_tomato_when_rotten")
        big_avg_fresh = cdata.get("big_avg_tomato_when_fresh")
        big_avg_rotten = cdata.get("big_avg_tomato_when_rotten")

        # --- Candidate 1: Agreement rate at 60% (current) ---
        c1_agreement = agreement_rate if agreement_rate else 0.5

        # --- Candidate 2: Fresh/Rotten spread ---
        # How much does knowing Fresh vs Rotten narrow down the score?
        if avg_fresh is not None and avg_rotten is not None:
            c2_spread = abs(avg_fresh - avg_rotten) / 100.0
        else:
            c2_spread = 0.0

        # --- Candidate 3: Big-movie spread ---
        if big_avg_fresh is not None and big_avg_rotten is not None:
            c3_big_spread = abs(big_avg_fresh - big_avg_rotten) / 100.0
        else:
            c3_big_spread = c2_spread

        # --- Candidate 4: Cross-threshold accuracy ---
        # For each review where we know the final score, check if the critic's
        # implied score (avg_fresh or avg_rotten) correctly predicts above/below
        # at every Kalshi threshold.
        threshold_correct = 0
        threshold_total = 0
        mae_errors = []

        for r in reviews:
            ems = r.get("movie_ems_id")
            if not ems or ems not in movies:
                continue
            actual = movies[ems]["tomatometer"]

            is_fresh = r.get("sentiment") == "POSITIVE"
            implied = avg_fresh if is_fresh else avg_rotten
            if implied is None:
                continue

            mae_errors.append(abs(implied - actual))

            for t in range(25, 96, 5):
                predicted_above = implied > t
                actual_above = actual > t
                if predicted_above == actual_above:
                    threshold_correct += 1
                threshold_total += 1

        c4_cross_threshold = threshold_correct / threshold_total if threshold_total > 0 else 0.5

        # --- Candidate 5: Inverse MAE ---
        # Lower prediction error = higher weight
        if mae_errors:
            mae = sum(mae_errors) / len(mae_errors)
            c5_inv_mae = max(0.1, 1.0 - mae / 100.0)
        else:
            c5_inv_mae = 0.5

        # --- Candidate 6: Big-movie cross-threshold accuracy ---
        big_correct = 0
        big_total = 0
        for r in reviews:
            ems = r.get("movie_ems_id")
            if not ems or ems not in movies:
                continue
            if movie_review_counts.get(ems, 0) < config.BIG_MOVIE_REVIEW_THRESHOLD:
                continue
            actual = movies[ems]["tomatometer"]

            is_fresh = r.get("sentiment") == "POSITIVE"
            implied = big_avg_fresh if (is_fresh and big_avg_fresh) else \
                      big_avg_rotten if (not is_fresh and big_avg_rotten) else \
                      avg_fresh if is_fresh else avg_rotten
            if implied is None:
                continue

            for t in range(25, 96, 5):
                predicted_above = implied > t
                actual_above = actual > t
                if predicted_above == actual_above:
                    big_correct += 1
                big_total += 1

        c6_big_cross = big_correct / big_total if big_total > 0 else c4_cross_threshold

        # --- Candidate 7: Spread * accuracy (composite) ---
        # Signal strength weighted by how accurate that signal is
        c7_composite = c2_spread * c4_cross_threshold

        metrics[critic_name] = {
            "agreement_60": c1_agreement,
            "spread": c2_spread,
            "big_spread": c3_big_spread,
            "cross_threshold_acc": c4_cross_threshold,
            "inv_mae": c5_inv_mae,
            "big_cross_threshold_acc": c6_big_cross,
            "spread_x_accuracy": c7_composite,
            "n_reviews": len(reviews),
        }

    return metrics


def predict_with_weight_fn(reviews, critic_db, critic_metrics, weight_fn_name):
    """Run prediction using a specific weight function instead of agreement rate."""
    if not reviews:
        return None

    weighted_fresh = 0.0
    weighted_total = 0.0

    for r in reviews:
        name = r["critic_name"]
        intensity = score_intensity(r)

        # Look up the candidate weight
        cm = critic_metrics.get(name)
        if cm and weight_fn_name in cm:
            weight = cm[weight_fn_name]
        else:
            # Fallback: use the median of this weight across all known critics
            all_vals = [m[weight_fn_name] for m in critic_metrics.values() if weight_fn_name in m]
            weight = sorted(all_vals)[len(all_vals) // 2] if all_vals else 0.5

        # Clamp to avoid zero/negative weights
        weight = max(0.01, weight)

        weighted_fresh += weight * intensity
        weighted_total += weight

    n = len(reviews)
    effective_total = weighted_total * config.CORRELATION_DISCOUNT
    shrinkage = min(1.0, n / config.SHRINKAGE_N)
    raw_mean = weighted_fresh / weighted_total if weighted_total > 0 else config.POPULATION_MEAN
    adjusted_mean = raw_mean * shrinkage + config.POPULATION_MEAN * (1.0 - shrinkage)

    alpha = config.PRIOR_ALPHA + adjusted_mean * effective_total
    beta = config.PRIOR_BETA + (1.0 - adjusted_mean) * effective_total

    dist = beta_dist(alpha, beta)
    mean = dist.mean()

    threshold_probs = {}
    for t in config.KALSHI_THRESHOLDS:
        threshold_probs[t] = float(1.0 - dist.cdf(t / 100.0))

    return {
        "model_mean": round(mean * 100, 1),
        "threshold_probs": threshold_probs,
        "naive_pct": sum(1 for r in reviews if r["sentiment"] == "Fresh") / n * 100,
    }


def evaluate_weight_function(weight_fn_name, critic_db, critic_metrics, movies,
                             checkpoints=None):
    """Run full backtest with a specific weight function. Returns Brier score."""
    if checkpoints is None:
        checkpoints = [10, 15, 20, 25, 30]

    threshold_pairs = []
    mae_pairs = []

    for ems_id, movie in movies.items():
        actual = movie["tomatometer"]
        all_reviews = movie["reviews"]

        for cp in checkpoints:
            if len(all_reviews) < cp:
                continue

            subset = all_reviews[:cp]
            pred = predict_with_weight_fn(subset, critic_db, critic_metrics, weight_fn_name)
            if not pred:
                continue

            mae_pairs.append((pred["model_mean"], actual))

            for t, prob in pred["threshold_probs"].items():
                outcome = 1 if actual > t else 0
                threshold_pairs.append((prob, outcome))

    if not threshold_pairs:
        return {"brier": 1.0, "mae": 100.0, "n_predictions": 0}

    return {
        "brier": brier_score(threshold_pairs),
        "mae": mean_absolute_error(mae_pairs),
        "n_predictions": len(mae_pairs),
    }


WEIGHT_FUNCTIONS = [
    ("agreement_60", "Agreement rate at 60% threshold (current)"),
    ("spread", "Fresh/Rotten tomatometer spread"),
    ("big_spread", "Big-movie Fresh/Rotten spread"),
    ("cross_threshold_acc", "Cross-threshold accuracy (25-95%)"),
    ("inv_mae", "Inverse MAE (prediction accuracy)"),
    ("big_cross_threshold_acc", "Big-movie cross-threshold accuracy"),
    ("spread_x_accuracy", "Spread * cross-threshold accuracy"),
]


def main():
    print("Loading critic database...", flush=True)
    critic_db = CriticDatabase()
    print(f"  {critic_db.count} critics loaded", flush=True)

    print("Loading calibration data...", flush=True)
    critic_cal = load_critic_calibration()
    if not critic_cal:
        print("No critic_reviews.json found. Run the scraper first.")
        sys.exit(1)
    print(f"  {len(critic_cal)} critics with calibration data", flush=True)

    print("Loading movie data...", flush=True)
    movies = load_movie_reviews()
    print(f"  {len(movies)} movies", flush=True)

    if not movies:
        print("No movies available for evaluation.")
        sys.exit(1)

    # Build movie lookup for calibration computation (need tomatometers)
    movie_lookup = {}
    for ems_id, movie in movies.items():
        movie_lookup[ems_id] = movie

    print("\nComputing critic metrics...", flush=True)
    critic_metrics = compute_critic_metrics(critic_cal, movie_lookup)
    print(f"  {len(critic_metrics)} critics with computed metrics", flush=True)

    # Show distribution of each metric
    print("\n--- Metric distributions ---")
    for fn_name, desc in WEIGHT_FUNCTIONS:
        vals = [m[fn_name] for m in critic_metrics.values() if fn_name in m]
        if vals:
            vals.sort()
            print(f"  {fn_name}: median={vals[len(vals)//2]:.3f}, "
                  f"min={vals[0]:.3f}, max={vals[-1]:.3f}, n={len(vals)}")

    # Evaluate each weight function
    print("\n--- Evaluating weight functions ---\n")
    results = []
    for fn_name, desc in WEIGHT_FUNCTIONS:
        print(f"  Testing: {fn_name} ({desc})...", flush=True)
        r = evaluate_weight_function(fn_name, critic_db, critic_metrics, movies)
        results.append((fn_name, desc, r))
        print(f"    Brier: {r['brier']:.4f}  MAE: {r['mae']:.2f}%  "
              f"({r['n_predictions']} predictions)")

    # Rank by Brier score (lower is better)
    results.sort(key=lambda x: x[2]["brier"])

    print("\n" + "=" * 70)
    print("RESULTS (ranked by Brier score, lower = better)")
    print("=" * 70)
    baseline_brier = results[-1][2]["brier"]
    for rank, (fn_name, desc, r) in enumerate(results, 1):
        improvement = ((baseline_brier - r["brier"]) / baseline_brier * 100) if baseline_brier > 0 else 0
        marker = " <-- WINNER" if rank == 1 else ""
        print(f"  {rank}. {fn_name}")
        print(f"     {desc}")
        print(f"     Brier: {r['brier']:.4f}  MAE: {r['mae']:.2f}%  "
              f"(+{improvement:.1f}% vs worst){marker}")

    winner = results[0]
    print(f"\nBest weight function: {winner[0]}")
    print(f"  {winner[1]}")
    print(f"  Brier score: {winner[2]['brier']:.4f}")
    print(f"  Score MAE: {winner[2]['mae']:.2f}%")

    # Save results
    output = {
        "winner": winner[0],
        "rankings": [
            {"rank": i+1, "name": fn, "description": desc,
             "brier": round(r["brier"], 4), "mae": round(r["mae"], 2)}
            for i, (fn, desc, r) in enumerate(results)
        ],
    }
    output_path = ROOT / "backtest" / "weight_eval_results.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == "__main__":
    main()
