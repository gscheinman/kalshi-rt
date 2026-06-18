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
REPO_DEPTH_FILE = Path(__file__).parent.parent / "data" / "orderbook_depth.jsonl"


def take_snapshot():
    """Capture current state of all active RT markets + model predictions."""
    critic_db = CriticDatabase()
    kalshi = KalshiClient()
    mapper = TickerMapper()

    events = kalshi.get_rt_events()
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshots = []

    # Detect markets we haven't tracked before. Compare current event tickers
    # against everything we've ever snapshotted in the repo-local jsonl.
    # Also build a set of tickers that have ever had reviews tracked so we
    # can notify on the 0-reviews -> first-reviews transition.
    seen_tickers = set()
    ever_had_reviews = set()
    if REPO_SNAPSHOT_FILE.exists():
        with open(REPO_SNAPSHOT_FILE) as f:
            for line in f:
                try:
                    s = json.loads(line)
                    if s.get("event_ticker"):
                        seen_tickers.add(s["event_ticker"])
                        n = ((s.get("model") or {}).get("n_reviews")) or 0
                        if n > 0:
                            ever_had_reviews.add(s["event_ticker"])
                except json.JSONDecodeError:
                    continue
    new_events = [e for e in events if e["event_ticker"] not in seen_tickers]

    print(f"Snapshotting {len(events)} active RT events at {timestamp[:19]}", flush=True)
    if new_events:
        print(f"  NEW markets detected ({len(new_events)}):", flush=True)
        for e in new_events:
            print(f"    + {e['event_ticker']:30} {e['movie_name']}", flush=True)

    # Collected during the loop: markets that just transitioned from
    # 0-reviews-ever to first-reviews-now. The workflow scans these
    # marker lines and creates a GitHub issue per match.
    first_review_markets = []

    # Orderbook depth ladders for tradeable markets this cycle (separate file).
    depth_records = []

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
                        reviews = scrape_reviews(
                            ems_id, slug=rt_slug,
                            expected_count=summary.get("review_count"),
                        )
                        rt_review_count = summary.get("review_count") or 0
                        if not reviews and rt_review_count > 0:
                            # RT page says reviews exist but scraper got nothing
                            # even after retry+cache fallback. Skip this market
                            # so we don't overwrite prior good model data.
                            print(
                                f"  {movie}: SKIP - RT reports {rt_review_count} "
                                f"reviews but scrape returned 0 (transient NAPI failure)",
                                flush=True,
                            )
                            continue
                        if reviews:
                            # Persist reviews to the cumulative per-movie store
                            # (dates + ids) so future model variants can be
                            # replayed against the exact historical inputs.
                            try:
                                from tracker.review_store import upsert_reviews
                                upsert_reviews(ticker, movie, rt_slug, reviews)
                            except Exception as e:
                                print(f"  (review_store skipped for {movie}: {e})", flush=True)

                            close_time = markets[0].get("close_time") if markets else None
                            # Include actual Kalshi thresholds for granular brackets
                            mkt_thresholds = [m["threshold"] for m in markets if m.get("threshold") is not None]
                            pred = predict_distribution(
                                reviews, critic_db,
                                movie_summary=summary, close_time=close_time,
                                extra_thresholds=mkt_thresholds,
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

        # Capture full orderbook depth ONLY for markets that have reviews (the
        # tradeable universe). Top-of-book bid/ask can't reconstruct slippage in
        # thin markets -- depth is the whole game there. This is irreversible:
        # Kalshi doesn't serve historical orderbooks, so any cycle we skip is
        # lost forever. Written to a SEPARATE file so snapshots.jsonl stays lean
        # for the tooling that parses it in full. Gated on model_data so we
        # don't fire ~15 orderbook calls for every zero-review movie.
        if model_data:
            for t, node in market_data.items():
                mt = node.get("ticker")
                if not mt:
                    continue
                try:
                    ob = kalshi.get_orderbook(mt)
                    if ob:
                        depth_records.append({
                            "timestamp": timestamp,
                            "event_ticker": ticker,
                            "threshold": t,
                            "market_ticker": mt,
                            "yes_bids": [[round(p, 4), round(s, 2)] for p, s in ob.get("yes_bids", [])],
                            "no_bids": [[round(p, 4), round(s, 2)] for p, s in ob.get("no_bids", [])],
                        })
                except Exception:
                    pass

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
            # First-reviews transition: this event has reviews now, and
            # had never had reviews in any prior snapshot.
            if ticker not in ever_had_reviews:
                first_review_markets.append({
                    "ticker": ticker,
                    "movie": movie,
                    "n_reviews": model_data["n_reviews"],
                    "naive_pct": model_data.get("naive_pct"),
                    "model_mean": model_data.get("model_mean"),
                    "best_edge": best_edge,
                })
                # Marker line the workflow greps for. Format must stay stable.
                print(
                    f"FIRST_REVIEWS_DETECTED ticker={ticker} "
                    f"movie={movie!r} n_reviews={model_data['n_reviews']} "
                    f"naive={model_data.get('naive_pct')} "
                    f"model_mean={model_data.get('model_mean')} "
                    f"best_edge={best_edge}",
                    flush=True,
                )
        print(f"  {movie}: {status}", flush=True)

    # Write to repo-local file
    REPO_SNAPSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(REPO_SNAPSHOT_FILE, "a") as f:
        for s in snapshots:
            f.write(json.dumps(s) + "\n")

    print(f"\n{len(snapshots)} snapshots saved to {REPO_SNAPSHOT_FILE}", flush=True)

    # Write orderbook depth ladders (separate file, tradeable markets only).
    if depth_records:
        with open(REPO_DEPTH_FILE, "a") as f:
            for r in depth_records:
                f.write(json.dumps(r) + "\n")
        print(f"{len(depth_records)} orderbook depth records saved to {REPO_DEPTH_FILE}", flush=True)

    if first_review_markets:
        print(f"\nFIRST-REVIEWS transitions this run: {len(first_review_markets)}", flush=True)

    # Log any opportunities the sanity guard blocked, so we can score the
    # guard's opportunity cost after settlement. Cheap, dedup'd, append-only.
    try:
        from tracker.shadow_tracker import log_shadow_trades
        log_shadow_trades()
    except Exception as e:
        print(f"  (shadow_tracker logging skipped: {e})", flush=True)

    # Log the naive-anchored strategy in parallel so we can judge it
    # out-of-sample against the model as new movies settle.
    try:
        from tracker.naive_tracker import log_naive_trades
        log_naive_trades()
    except Exception as e:
        print(f"  (naive_tracker logging skipped: {e})", flush=True)

    return snapshots


if __name__ == "__main__":
    take_snapshot()
