"""
Backtesting framework for the RT prediction model.

Reconstructs what the model would have predicted for historical movies
at various review count checkpoints, then compares to actual outcomes.

Data source: critic_reviews.json from the scraper, which contains per-critic
review histories with movie EMS IDs, sentiments, scores, and (after Phase 2)
final tomatometers for each movie.

Usage:
    python -m backtest.evaluate
    python -m backtest.evaluate --checkpoint 10  # only test at N=10 reviews
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from backtest.metrics import (
    brier_score,
    calibration_buckets,
    calibration_error,
    mean_absolute_error,
    sharpe_ratio,
    simulated_pnl,
)
from data.critics import CriticDatabase
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
import config

ROOT = Path(__file__).parent.parent
REVIEWS_PATH = ROOT / "critic_reviews.json"
MOVIES_CACHE = Path.home() / ".cache" / "kalshi-rt" / "movie_scores.json"
RESULTS_PATH = ROOT / "backtest" / "results.json"

CHECKPOINTS = [5, 10, 15, 20, 25, 30, 40, 50]


def load_movie_reviews():
    """Build a movie -> [reviews] mapping from the critic review data.

    Each movie gets a list of all critic reviews we have for it,
    with the final tomatometer attached.
    """
    if not REVIEWS_PATH.exists():
        print(f"ERROR: {REVIEWS_PATH} not found. Run the critic scraper first.")
        sys.exit(1)

    with open(REVIEWS_PATH) as f:
        critic_data = json.load(f)

    # Load movie tomatometers
    movie_scores = {}
    if MOVIES_CACHE.exists():
        with open(MOVIES_CACHE) as f:
            raw = json.load(f)
            for ems_id, info in raw.items():
                if info.get("tomatometer") is not None:
                    movie_scores[ems_id] = info["tomatometer"]

    # Collect reviews per movie
    movies = defaultdict(lambda: {"reviews": [], "tomatometer": None, "title": ""})

    for critic_name, cdata in critic_data.items():
        for r in cdata.get("reviews", []):
            ems = r.get("movie_ems_id")
            if not ems:
                continue
            if ems not in movie_scores:
                continue

            movies[ems]["tomatometer"] = movie_scores[ems]
            movies[ems]["title"] = r.get("movie_title", "")
            movies[ems]["reviews"].append({
                "critic_name": critic_name,
                "sentiment": "Fresh" if r["sentiment"] == "POSITIVE" else "Rotten",
                "publication": r.get("publication", ""),
                "top_critic": False,
                "rating_text": r.get("original_score", ""),
                "quote": "",
            })

    # Filter to movies with enough reviews and a known score
    valid = {}
    for ems, data in movies.items():
        if data["tomatometer"] is not None and len(data["reviews"]) >= 5:
            data["total_reviews"] = len(data["reviews"])
            valid[ems] = data

    return valid


def run_backtest(critic_db, movies, checkpoints=None):
    """Run the model at each checkpoint for each movie and collect predictions."""
    if checkpoints is None:
        checkpoints = CHECKPOINTS

    results = {cp: [] for cp in checkpoints}

    for ems_id, movie in movies.items():
        actual = movie["tomatometer"]
        all_reviews = movie["reviews"]
        title = movie["title"]

        for cp in checkpoints:
            if len(all_reviews) < cp:
                continue

            # Take the first N reviews (simulating early access)
            subset = all_reviews[:cp]
            prediction = predict_distribution(subset, critic_db)

            if prediction["n_reviews"] == 0:
                continue

            calibrated = calibrate_thresholds(
                prediction["threshold_probs"],
                prediction["n_reviews"],
            )

            results[cp].append({
                "movie": title,
                "ems_id": ems_id,
                "actual_score": actual,
                "predicted_mean": prediction["model_mean"],
                "predicted_ci": prediction["model_ci"],
                "n_reviews_used": cp,
                "n_known": prediction["n_known"],
                "known_pct": prediction["known_pct"],
                "confidence": prediction["confidence"],
                "naive_pct": prediction["naive_pct"],
                "total_reviews": movie.get("total_reviews", len(all_reviews)),
                "threshold_probs": {str(t): round(p, 4) for t, p in calibrated.items()},
            })

    return results


def evaluate_results(results):
    """Compute metrics for each checkpoint."""
    report = {}

    for cp, preds in results.items():
        if not preds:
            continue

        # 1. Mean score prediction accuracy
        score_pairs = [(p["predicted_mean"], p["actual_score"]) for p in preds]
        mae = mean_absolute_error(score_pairs)
        naive_pairs = [(p["naive_pct"], p["actual_score"]) for p in preds]
        naive_mae = mean_absolute_error(naive_pairs)

        # 2. Threshold probability calibration (across all thresholds)
        threshold_predictions = []
        for p in preds:
            actual = p["actual_score"]
            for t_str, prob in p["threshold_probs"].items():
                t = int(t_str)
                outcome = 1 if actual > t else 0
                threshold_predictions.append((prob, outcome))

        brier = brier_score(threshold_predictions)
        cal_error = calibration_error(threshold_predictions)
        cal_buckets = calibration_buckets(threshold_predictions)

        # 3. Simulated trading (simple: bet on any threshold with >5% edge vs 50%)
        trades = []
        for p in preds:
            actual = p["actual_score"]
            for t_str, prob in p["threshold_probs"].items():
                t = int(t_str)
                if prob > 0.55:
                    trades.append({
                        "direction": "BUY YES",
                        "threshold": t,
                        "cost": 0.50,
                        "contracts": 1.0,
                        "actual_score": actual,
                        "model_prob": prob,
                    })
                elif prob < 0.45:
                    trades.append({
                        "direction": "BUY NO",
                        "threshold": t,
                        "cost": 0.50,
                        "contracts": 1.0,
                        "actual_score": actual,
                        "model_prob": 1 - prob,
                    })

        pnl = simulated_pnl(trades)
        trade_pnls = [t["pnl"] for t in pnl["details"]] if pnl else []
        sharpe = sharpe_ratio(trade_pnls)

        # 4. Model vs naive comparison
        model_correct = 0
        naive_correct = 0
        total_compared = 0
        for p in preds:
            actual = p["actual_score"]
            for t_str, prob in p["threshold_probs"].items():
                t = int(t_str)
                if t < 20 or t > 95:
                    continue
                model_says_above = prob > 0.5
                naive_says_above = p["naive_pct"] > t
                actual_above = actual > t

                if model_says_above == actual_above:
                    model_correct += 1
                if naive_says_above == actual_above:
                    naive_correct += 1
                total_compared += 1

        # 5. Big-movie subset (movies with 80+ total reviews = Kalshi-like)
        big_preds = [p for p in preds if p.get("total_reviews", 0) >= config.BIG_MOVIE_REVIEW_THRESHOLD]
        big_mae = None
        big_naive_mae = None
        if big_preds:
            big_score_pairs = [(p["predicted_mean"], p["actual_score"]) for p in big_preds]
            big_naive_pairs = [(p["naive_pct"], p["actual_score"]) for p in big_preds]
            big_mae = mean_absolute_error(big_score_pairs)
            big_naive_mae = mean_absolute_error(big_naive_pairs)

        report[cp] = {
            "n_movies": len(preds),
            "n_big_movies": len(big_preds),
            "mae_model": round(mae, 2) if mae else None,
            "mae_naive": round(naive_mae, 2) if naive_mae else None,
            "mae_improvement": round(naive_mae - mae, 2) if mae and naive_mae else None,
            "big_movie_mae_model": round(big_mae, 2) if big_mae else None,
            "big_movie_mae_naive": round(big_naive_mae, 2) if big_naive_mae else None,
            "big_movie_mae_improvement": round(big_naive_mae - big_mae, 2) if big_mae and big_naive_mae else None,
            "brier_score": round(brier, 4) if brier else None,
            "calibration_error": round(cal_error, 4) if cal_error else None,
            "calibration_buckets": cal_buckets,
            "model_accuracy": round(model_correct / total_compared * 100, 1) if total_compared else None,
            "naive_accuracy": round(naive_correct / total_compared * 100, 1) if total_compared else None,
            "simulated_pnl": pnl["total_pnl"] if pnl else None,
            "simulated_roi": pnl["roi"] if pnl else None,
            "win_rate": pnl["win_rate"] if pnl else None,
            "total_trades": pnl["total_trades"] if pnl else None,
            "sharpe": sharpe,
        }

    return report


def print_report(report):
    """Print a human-readable summary."""
    print("\n" + "=" * 70)
    print("BACKTEST RESULTS")
    print("=" * 70)

    for cp in sorted(report.keys()):
        r = report[cp]
        print(f"\n--- {cp} reviews ---  ({r['n_movies']} movies, {r.get('n_big_movies', '?')} big)")
        print(f"  Score MAE:     Model {r['mae_model']}%  vs  Naive {r['mae_naive']}%  (improvement: {r['mae_improvement']}%)")
        if r.get("big_movie_mae_model") is not None:
            print(f"  Big Movie MAE: Model {r['big_movie_mae_model']}%  vs  Naive {r['big_movie_mae_naive']}%  (improvement: {r['big_movie_mae_improvement']}%)")
        print(f"  Threshold accuracy: Model {r['model_accuracy']}%  vs  Naive {r['naive_accuracy']}%")
        print(f"  Brier score:   {r['brier_score']}  (lower is better, 0.25 = random)")
        print(f"  Calibration error: {r['calibration_error']}  (lower is better)")

        if r["simulated_pnl"] is not None:
            print(f"  Simulated P&L: ${r['simulated_pnl']}  ({r['total_trades']} trades, {r['win_rate']}% win rate)")
            print(f"  Simulated ROI: {r['simulated_roi']}%")
            if r["sharpe"]:
                print(f"  Sharpe ratio:  {r['sharpe']}")

        if r["calibration_buckets"]:
            print(f"  Calibration by bucket:")
            for b in r["calibration_buckets"]:
                bar_len = int(b["count"] / 5)
                bar = "#" * min(bar_len, 30)
                print(f"    {b['bucket_center']:.0%}: predicted {b['predicted_avg']:.0%} actual {b['actual_rate']:.0%} (n={b['count']}) {bar}")


def main():
    parser = argparse.ArgumentParser(description="Backtest the RT prediction model")
    parser.add_argument("--checkpoint", type=int, help="Only test at this review count")
    parser.add_argument("--csv", type=str, help="Path to critic database CSV")
    args = parser.parse_args()

    print("Loading critic database...", flush=True)
    critic_db = CriticDatabase(args.csv) if args.csv else CriticDatabase()
    print(f"  {critic_db.count} critics loaded", flush=True)

    print("Loading movie review data...", flush=True)
    movies = load_movie_reviews()
    print(f"  {len(movies)} movies with reviews and known tomatometers", flush=True)

    if not movies:
        print("\nNo movies available for backtesting.")
        print("Run the critic scraper first: python -m scraper.rt_critics")
        return

    checkpoints = [args.checkpoint] if args.checkpoint else CHECKPOINTS

    print(f"\nRunning backtest at checkpoints: {checkpoints}...", flush=True)
    results = run_backtest(critic_db, movies, checkpoints)

    total_preds = sum(len(v) for v in results.values())
    print(f"  Generated {total_preds} predictions across {len(movies)} movies", flush=True)

    report = evaluate_results(results)
    print_report(report)

    # Save results
    with open(RESULTS_PATH, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\nResults saved to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
