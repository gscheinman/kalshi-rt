"""
Lightweight snapshot for CI/GitHub Actions.

Same as market_snapshot.take_snapshot() but writes to a repo-local file
(data/snapshots.jsonl) instead of ~/.cache/. This lets GitHub Actions
commit the data back to the repo after each run.

Also merges any locally-collected snapshots from ~/.cache/ so nothing is lost.
"""
import json
import sys
import os
import time
from datetime import datetime, timezone
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market.kalshi_client import KalshiClient
from market.mapper import TickerMapper
from scraper.rt_page import get_movie_summary
from scraper.rt_reviews import scrape_reviews
from data.critics import CriticDatabase
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
import config

REPO_SNAPSHOT_FILE = Path(__file__).parent.parent / "data" / "snapshots.jsonl"


def take_snapshot():
    """Capture current state of all active RT markets + model predictions."""
    critic_db = CriticDatabase()
    kalshi = KalshiClient()
    mapper = TickerMapper()

    events = kalshi.get_rt_events()
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshots = []

    print(f"Snapshotting {len(events)} active RT events at {timestamp[:19]}", flush=True)

    for event in events:
        ticker = event["event_ticker"]
        movie = event["movie_name"]

        markets = kalshi.get_markets(ticker)
        if not markets:
            continue

        market_data = {}
        total_volume = 0
        for m in markets:
            t = m.get("threshold")
            if t is None:
                continue
            market_data[t] = {
                "yes_price": m.get("yes_price"),
                "yes_bid": m.get("yes_bid"),
                "yes_ask": m.get("yes_ask"),
                "volume": m.get("volume", 0),
                "open_interest": m.get("open_interest", 0),
                "ticker": m.get("ticker", ""),
            }
            total_volume += float(m.get("volume", 0) or 0)

        # Try to get model prediction
        model_data = None
        rt_slug = mapper.get_rt_slug(event)
        if rt_slug:
            try:
                summary = get_movie_summary(rt_slug)
                if summary:
                    ems_id = summary.get("ems_id")
                    if ems_id:
                        reviews = scrape_reviews(ems_id, slug=rt_slug, max_pages=5)
                        if reviews:
                            close_time = markets[0].get("close_time") if markets else None
                            pred = predict_distribution(
                                reviews, critic_db,
                                movie_summary=summary, close_time=close_time,
                            )
                            calibrated = calibrate_thresholds(
                                pred["threshold_probs"], pred["n_reviews"]
                            )
                            model_data = {
                                "n_reviews": pred["n_reviews"],
                                "n_known": pred["n_known"],
                                "known_pct": pred["known_pct"],
                                "naive_pct": pred["naive_pct"],
                                "model_mean": pred["model_mean"],
                                "model_ci": pred["model_ci"],
                                "confidence": pred["confidence"],
                                "prior_mean": pred.get("prior_mean"),
                                "review_completion": pred.get("review_completion"),
                                "corr_discount": pred.get("corr_discount"),
                                "threshold_probs": {
                                    str(k): round(v, 4)
                                    for k, v in calibrated.items()
                                },
                            }

                            # Compute edges at each threshold
                            edges = {}
                            for t, prob in calibrated.items():
                                mkt = market_data.get(t, {})
                                yes_ask = mkt.get("yes_ask") or mkt.get("yes_price")
                                if yes_ask:
                                    edges[str(t)] = round((prob - yes_ask) * 100, 1)
                            model_data["edges"] = edges
            except Exception as e:
                print(f"  Error modeling {movie}: {e}", flush=True)

        snapshot = {
            "timestamp": timestamp,
            "event_ticker": ticker,
            "movie": movie,
            "rt_slug": rt_slug,
            "markets": {str(k): v for k, v in market_data.items()},
            "total_volume": round(total_volume, 2),
            "model": model_data,
            "resolved": False,
            "actual_score": None,
        }

        snapshots.append(snapshot)

        status = "no reviews"
        if model_data:
            best_edge = max(model_data.get("edges", {}).values(), default=0, key=abs)
            status = (
                f"{model_data['n_reviews']} reviews, "
                f"mean={model_data['model_mean']}, "
                f"best_edge={best_edge}%"
            )
        print(f"  {movie}: {status}", flush=True)

    # Write to repo-local file
    REPO_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPO_SNAPSHOT_FILE, "a") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")

    print(f"\n{len(snapshots)} snapshots saved to {REPO_SNAPSHOT_FILE}", flush=True)
    return snapshots


if __name__ == "__main__":
    take_snapshot()
