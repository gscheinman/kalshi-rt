"""
End-to-end model training and evaluation pipeline.

Loads scraper data, trains the ML critic weight model, compares it against
the hand-crafted approach, and reports which produces better predictions
for Kalshi-style threshold betting.

Usage:
    python -m backtest.train_model

This should be run after the critic scraper completes (needs critic_reviews.json
and movie_scores.json).
"""
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist

from backtest.evaluate import load_movie_reviews, CHECKPOINTS
from backtest.metrics import brier_score, mean_absolute_error, simulated_pnl, sharpe_ratio
from data.critics import CriticDatabase
from model.learned_weights import train_model, LearnedPredictor, MODEL_PATH
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
from model.sentiment import score_intensity
import config

ROOT = Path(__file__).parent.parent
REVIEWS_PATH = ROOT / "critic_reviews.json"
MOVIES_CACHE = Path.home() / ".cache" / "kalshi-rt" / "movie_scores.json"


def load_scraper_data():
    """Load critic calibration and movie score data from scraper output."""
    if not REVIEWS_PATH.exists():
        print(f"ERROR: {REVIEWS_PATH} not found. Run scraper first.")
        sys.exit(1)

    with open(REVIEWS_PATH) as f:
        critic_cal = json.load(f)

    movie_scores = {}
    if MOVIES_CACHE.exists():
        with open(MOVIES_CACHE) as f:
            movie_scores = json.load(f)

    return critic_cal, movie_scores


def evaluate_model_on_movies(movies, critic_db, use_ml=False, checkpoints=None):
    """Run predictions on historical movies and compute trading-relevant metrics.

    Returns dict with Brier score, MAE, simulated P&L, etc.
    Split by all movies and big movies (Kalshi-relevant).
    """
    if checkpoints is None:
        checkpoints = [10, 15, 20, 25, 30]

    all_threshold_pairs = []
    big_threshold_pairs = []
    all_mae_pairs = []
    big_mae_pairs = []
    all_trades = []
    big_trades = []

    for ems_id, movie in movies.items():
        actual = movie["tomatometer"]
        all_reviews = movie["reviews"]
        is_big = movie.get("total_reviews", len(all_reviews)) >= config.BIG_MOVIE_REVIEW_THRESHOLD

        for cp in checkpoints:
            if len(all_reviews) < cp:
                continue

            subset = all_reviews[:cp]
            pred = predict_distribution(subset, critic_db)
            if pred["n_reviews"] == 0:
                continue

            calibrated = calibrate_thresholds(pred["threshold_probs"], pred["n_reviews"])

            all_mae_pairs.append((pred["model_mean"], actual))
            if is_big:
                big_mae_pairs.append((pred["model_mean"], actual))

            for t, prob in calibrated.items():
                outcome = 1 if actual > t else 0
                all_threshold_pairs.append((prob, outcome))
                if is_big:
                    big_threshold_pairs.append((prob, outcome))

                # Simulated trades (bet when edge > 5% vs fair price of 50c)
                if prob > 0.55:
                    trade = {
                        "direction": "BUY YES", "threshold": t, "cost": 0.50,
                        "contracts": 1.0, "actual_score": actual, "model_prob": prob,
                    }
                    all_trades.append(trade)
                    if is_big:
                        big_trades.append(trade)
                elif prob < 0.45:
                    trade = {
                        "direction": "BUY NO", "threshold": t, "cost": 0.50,
                        "contracts": 1.0, "actual_score": actual, "model_prob": 1 - prob,
                    }
                    all_trades.append(trade)
                    if is_big:
                        big_trades.append(trade)

    all_pnl = simulated_pnl(all_trades)
    big_pnl = simulated_pnl(big_trades)

    return {
        "all": {
            "brier": brier_score(all_threshold_pairs),
            "mae": mean_absolute_error(all_mae_pairs),
            "n_movies": len(all_mae_pairs),
            "pnl": all_pnl["total_pnl"] if all_pnl else 0,
            "roi": all_pnl["roi"] if all_pnl else 0,
            "win_rate": all_pnl["win_rate"] if all_pnl else 0,
            "trades": all_pnl["total_trades"] if all_pnl else 0,
            "sharpe": sharpe_ratio([t["pnl"] for t in all_pnl["details"]]) if all_pnl else None,
        },
        "big": {
            "brier": brier_score(big_threshold_pairs) if big_threshold_pairs else None,
            "mae": mean_absolute_error(big_mae_pairs) if big_mae_pairs else None,
            "n_movies": len(big_mae_pairs),
            "pnl": big_pnl["total_pnl"] if big_pnl else 0,
            "roi": big_pnl["roi"] if big_pnl else 0,
            "win_rate": big_pnl["win_rate"] if big_pnl else 0,
            "trades": big_pnl["total_trades"] if big_pnl else 0,
            "sharpe": sharpe_ratio([t["pnl"] for t in big_pnl["details"]]) if big_pnl else None,
        },
    }


def print_comparison(label, metrics):
    """Print metrics for a model variant."""
    for subset in ("all", "big"):
        m = metrics[subset]
        tag = "ALL MOVIES" if subset == "all" else "BIG MOVIES (Kalshi-relevant)"
        print(f"\n  {tag}:")
        print(f"    Predictions: {m['n_movies']}")
        if m["brier"] is not None:
            print(f"    Brier score: {m['brier']:.4f} (lower = better, 0.25 = random)")
        if m["mae"] is not None:
            print(f"    Score MAE:   {m['mae']:.2f}%")
        if m["trades"] > 0:
            print(f"    Sim trades:  {m['trades']} ({m['win_rate']:.1f}% win rate)")
            print(f"    Sim P&L:     ${m['pnl']:.2f} ({m['roi']:.1f}% ROI)")
            if m["sharpe"]:
                print(f"    Sharpe:      {m['sharpe']:.2f}")


def main():
    print("=" * 70)
    print("KALSHI RT MODEL TRAINING PIPELINE")
    print("=" * 70)

    print("\nLoading data...", flush=True)
    critic_cal, movie_scores = load_scraper_data()
    print(f"  {len(critic_cal)} critics with calibration data")
    scored = sum(1 for v in movie_scores.values() if v.get("tomatometer") is not None)
    print(f"  {scored} movies with tomatometers")

    critic_db = CriticDatabase()
    print(f"  {critic_db.count} critics in database")

    movies = load_movie_reviews()
    print(f"  {len(movies)} movies available for evaluation")

    if not movies:
        print("\nNo movies with reviews and scores. Run scraper phases 1-3 first.")
        sys.exit(1)

    # Step 1: Evaluate hand-crafted model (baseline)
    print("\n" + "=" * 70)
    print("STEP 1: BASELINE (hand-crafted weights)")
    print("=" * 70)
    baseline = evaluate_model_on_movies(movies, critic_db, use_ml=False)
    print_comparison("Baseline", baseline)

    # Step 2: Train ML model
    print("\n" + "=" * 70)
    print("STEP 2: TRAINING ML MODEL")
    print("=" * 70)
    result = train_model(critic_cal, movie_scores)
    if result is None:
        print("ML training failed. Not enough data.")
        return

    # Step 3: Evaluate ML model
    print("\n" + "=" * 70)
    print("STEP 3: ML MODEL EVALUATION")
    print("=" * 70)
    ml_metrics = evaluate_model_on_movies(movies, critic_db, use_ml=True)
    print_comparison("ML Model", ml_metrics)

    # Step 4: Compare
    print("\n" + "=" * 70)
    print("COMPARISON: ML vs BASELINE")
    print("=" * 70)

    for subset, label in [("all", "All Movies"), ("big", "Big Movies (Kalshi)")]:
        b = baseline[subset]
        m = ml_metrics[subset]
        print(f"\n  {label}:")

        if b["brier"] and m["brier"]:
            brier_diff = b["brier"] - m["brier"]
            pct = brier_diff / b["brier"] * 100 if b["brier"] > 0 else 0
            winner = "ML" if brier_diff > 0 else "Baseline"
            print(f"    Brier: {b['brier']:.4f} -> {m['brier']:.4f} "
                  f"({'+' if brier_diff > 0 else ''}{pct:.1f}% -- {winner} wins)")

        if b["mae"] and m["mae"]:
            mae_diff = b["mae"] - m["mae"]
            winner = "ML" if mae_diff > 0 else "Baseline"
            print(f"    MAE:   {b['mae']:.2f}% -> {m['mae']:.2f}% "
                  f"({'+' if mae_diff > 0 else ''}{mae_diff:.2f}% -- {winner} wins)")

        if b["trades"] > 0 and m["trades"] > 0:
            pnl_diff = m["pnl"] - b["pnl"]
            print(f"    P&L:   ${b['pnl']:.2f} -> ${m['pnl']:.2f} "
                  f"(${'+' if pnl_diff > 0 else ''}{pnl_diff:.2f})")

    # Final recommendation
    big_b = baseline["big"]
    big_m = ml_metrics["big"]
    if big_b["brier"] and big_m["brier"]:
        if big_m["brier"] < big_b["brier"]:
            print(f"\n*** ML MODEL WINS on big movies. Using learned weights. ***")
            print(f"    Model saved to {MODEL_PATH}")
        else:
            print(f"\n*** BASELINE WINS on big movies. ML model not better. ***")
            print(f"    Keeping hand-crafted weights.")
            print(f"    (ML model still saved for reference at {MODEL_PATH})")
    else:
        print(f"\n*** Not enough big-movie data to compare. ***")

    # Save comparison results
    output = {
        "baseline": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                         for kk, vv in v.items()} for k, v in baseline.items()},
        "ml": {k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()} for k, v in ml_metrics.items()},
    }
    output_path = ROOT / "backtest" / "model_comparison.json"
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nFull comparison saved to {output_path}")


if __name__ == "__main__":
    main()
