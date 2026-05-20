"""
Parameter optimizer for the RT prediction model.

Uses backtest results to fit UNVALIDATED parameters in config.py:
- PRIOR_ALPHA, PRIOR_BETA (Beta prior shape)
- CORRELATION_DISCOUNT (effective sample size multiplier)
- SHRINKAGE_N (mean reversion strength)
- Platt scaling params in calibration.py

Also fits calibration parameters from predicted vs actual threshold outcomes.

Usage:
    python -m backtest.optimize
    python -m backtest.optimize --param shrinkage  # optimize single param
"""
import json
import math
import sys
from itertools import product
from pathlib import Path

from backtest.evaluate import load_movie_reviews, CHECKPOINTS
from backtest.metrics import brier_score, mean_absolute_error
from data.critics import CriticDatabase
from model.calibration import calibrate_thresholds
import config

ROOT = Path(__file__).parent.parent
OPTIMIZED_CONFIG_PATH = ROOT / "backtest" / "optimized_params.json"
OPTIMIZED_PLATT_PATH = ROOT / "backtest" / "optimized_platt.json"


def predict_with_params(reviews, critic_db, prior_alpha, prior_beta,
                        correlation_discount, shrinkage_n):
    """Run the prediction model with custom parameters (not config defaults)."""
    from scipy.stats import beta as beta_dist
    from model.sentiment import score_intensity

    if not reviews:
        return None

    weighted_fresh = 0.0
    weighted_total = 0.0
    known_count = 0

    for r in reviews:
        name = r["critic_name"]
        is_fresh = r["sentiment"] == "Fresh"
        is_top = r.get("top_critic", False)
        weight = critic_db.get_weight(name, is_top_critic=is_top)
        intensity = score_intensity(r)
        weighted_fresh += weight * intensity
        weighted_total += weight
        if critic_db.get_critic(name):
            known_count += 1

    n = len(reviews)
    effective_total = weighted_total * correlation_discount
    shrinkage = min(1.0, n / shrinkage_n)
    raw_mean = weighted_fresh / weighted_total if weighted_total > 0 else config.POPULATION_MEAN
    adjusted_mean = raw_mean * shrinkage + config.POPULATION_MEAN * (1.0 - shrinkage)

    alpha = prior_alpha + adjusted_mean * effective_total
    beta = prior_beta + (1.0 - adjusted_mean) * effective_total

    dist = beta_dist(alpha, beta)
    mean = dist.mean()

    threshold_probs = {}
    for t in config.KALSHI_THRESHOLDS:
        threshold_probs[t] = float(1.0 - dist.cdf(t / 100.0))

    naive_pct = sum(1 for r in reviews if r["sentiment"] == "Fresh") / n * 100

    return {
        "model_mean": round(mean * 100, 1),
        "naive_pct": round(naive_pct, 1),
        "threshold_probs": threshold_probs,
        "n_reviews": n,
        "n_known": known_count,
    }


def evaluate_params(critic_db, movies, prior_alpha, prior_beta,
                    correlation_discount, shrinkage_n, checkpoints=None):
    """Run backtest with specific parameters and return aggregate score.

    Returns a composite score (lower is better) combining:
    - MAE of score predictions (weight: 1.0)
    - Brier score of threshold predictions (weight: 100.0, scaled to similar range)
    """
    if checkpoints is None:
        checkpoints = [10, 15, 20, 25, 30]

    total_mae_pairs = []
    total_threshold_pairs = []

    for ems_id, movie in movies.items():
        actual = movie["tomatometer"]
        all_reviews = movie["reviews"]

        for cp in checkpoints:
            if len(all_reviews) < cp:
                continue

            subset = all_reviews[:cp]
            pred = predict_with_params(
                subset, critic_db, prior_alpha, prior_beta,
                correlation_discount, shrinkage_n,
            )
            if not pred or pred["n_reviews"] == 0:
                continue

            total_mae_pairs.append((pred["model_mean"], actual))

            calibrated = calibrate_thresholds(pred["threshold_probs"], pred["n_reviews"])
            for t, prob in calibrated.items():
                outcome = 1 if actual > t else 0
                total_threshold_pairs.append((prob, outcome))

    if not total_mae_pairs:
        return float("inf")

    mae = mean_absolute_error(total_mae_pairs)
    brier = brier_score(total_threshold_pairs) if total_threshold_pairs else 0.25

    return mae + brier * 100


def grid_search_params(critic_db, movies):
    """Grid search over UNVALIDATED parameters to minimize prediction error."""
    param_grid = {
        "prior_alpha": [1.0, 1.5, 2.0, 2.5, 3.0, 4.0],
        "prior_beta": [0.5, 1.0, 1.5, 2.0, 2.5],
        "correlation_discount": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "shrinkage_n": [15, 20, 25, 30, 40, 50],
    }

    best_score = float("inf")
    best_params = None
    total = 1
    for v in param_grid.values():
        total *= len(v)

    print(f"Grid search: {total} parameter combinations", flush=True)
    tested = 0

    for pa in param_grid["prior_alpha"]:
        for pb in param_grid["prior_beta"]:
            for cd in param_grid["correlation_discount"]:
                for sn in param_grid["shrinkage_n"]:
                    score = evaluate_params(
                        critic_db, movies, pa, pb, cd, sn,
                    )
                    tested += 1
                    if score < best_score:
                        best_score = score
                        best_params = {
                            "PRIOR_ALPHA": pa,
                            "PRIOR_BETA": pb,
                            "CORRELATION_DISCOUNT": cd,
                            "SHRINKAGE_N": sn,
                        }
                        print(f"  [{tested}/{total}] New best: score={score:.3f} "
                              f"alpha={pa} beta={pb} corr={cd} shrink={sn}", flush=True)
                    elif tested % 100 == 0:
                        print(f"  [{tested}/{total}] ...", flush=True)

    return best_params, best_score


def fit_platt_params(critic_db, movies):
    """Fit Platt scaling parameters from backtest data.

    For each review-count bucket, collect (raw_probability, actual_outcome) pairs,
    then fit a * logit(p) + b via gradient descent on log-loss.
    """
    buckets = {5: [], 10: [], 15: [], 20: [], 25: [], 30: []}

    for ems_id, movie in movies.items():
        actual = movie["tomatometer"]
        all_reviews = movie["reviews"]

        for cp in buckets.keys():
            if len(all_reviews) < cp:
                continue

            from model.distribution import predict_distribution
            subset = all_reviews[:cp]
            pred = predict_distribution(subset, critic_db)
            if pred["n_reviews"] == 0:
                continue

            for t, raw_p in pred["threshold_probs"].items():
                if raw_p is None:
                    continue
                outcome = 1 if actual > t else 0
                buckets[cp].append((raw_p, outcome))

    platt_params = {}
    for bucket_n, pairs in sorted(buckets.items()):
        if len(pairs) < 20:
            platt_params[bucket_n] = [1.0, 0.0]
            continue

        a, b = _fit_platt_single(pairs)
        platt_params[bucket_n] = [round(a, 4), round(b, 4)]
        print(f"  N={bucket_n}: a={a:.4f}, b={b:.4f} ({len(pairs)} samples)", flush=True)

    return platt_params


def _fit_platt_single(pairs, lr=0.01, epochs=1000):
    """Fit Platt scaling a, b via gradient descent on cross-entropy loss."""
    a, b = 1.0, 0.0

    for _ in range(epochs):
        grad_a, grad_b = 0.0, 0.0
        for raw_p, y in pairs:
            raw_p = max(0.001, min(0.999, raw_p))
            logit = math.log(raw_p / (1 - raw_p))
            z = a * logit + b
            sig = 1.0 / (1.0 + math.exp(-z))
            err = sig - y
            grad_a += err * logit
            grad_b += err

        n = len(pairs)
        a -= lr * grad_a / n
        b -= lr * grad_b / n

    return a, b


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Optimize model parameters from backtest")
    parser.add_argument("--param", type=str, help="Optimize a single parameter group: model, platt, or all")
    args = parser.parse_args()

    param = args.param or "all"

    print("Loading critic database...", flush=True)
    critic_db = CriticDatabase()
    print(f"  {critic_db.count} critics loaded", flush=True)

    print("Loading movie review data...", flush=True)
    movies = load_movie_reviews()
    print(f"  {len(movies)} movies available", flush=True)

    if not movies:
        print("No movies available. Run the critic scraper first.")
        sys.exit(1)

    if param in ("model", "all"):
        print("\n--- Optimizing model parameters ---", flush=True)
        best_params, best_score = grid_search_params(critic_db, movies)
        print(f"\nBest parameters (score={best_score:.3f}):")
        for k, v in best_params.items():
            current = getattr(config, k)
            delta = "CHANGED" if v != current else "same"
            print(f"  {k}: {current} -> {v}  ({delta})")

        with open(OPTIMIZED_CONFIG_PATH, "w") as f:
            json.dump({"params": best_params, "score": best_score}, f, indent=2)
        print(f"\nSaved to {OPTIMIZED_CONFIG_PATH}")

    if param in ("platt", "all"):
        print("\n--- Fitting Platt calibration parameters ---", flush=True)
        platt = fit_platt_params(critic_db, movies)
        print(f"\nFitted Platt params:")
        for n, (a, b) in sorted(platt.items()):
            print(f"  N={n}: a={a}, b={b}")

        with open(OPTIMIZED_PLATT_PATH, "w") as f:
            json.dump(platt, f, indent=2)
        print(f"\nSaved to {OPTIMIZED_PLATT_PATH}")

    print("\nTo apply these parameters, update config.py and model/calibration.py")
    print("Or run: python -m backtest.apply_params")


if __name__ == "__main__":
    main()
