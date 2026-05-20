"""
Kalshi market price snapshot collector.

Captures prices, orderbook depth, and model predictions for all active
RT markets at regular intervals. Each snapshot becomes a training example
after settlement: "given these reviews and this market price, what actually
happened?"

This is the foundation for training on real market data instead of
hypothetical fair prices.

Usage:
    python -m tracker.market_snapshot              # single snapshot
    python -m tracker.market_snapshot --loop 3600  # hourly snapshots
    python -m tracker.market_snapshot --status     # check loop health
"""
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

from data.critics import CriticDatabase
from market.kalshi_client import KalshiClient
from market.mapper import TickerMapper
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
from scraper.rt_page import get_movie_summary
from scraper.rt_reviews import scrape_reviews
import config

SNAPSHOT_DIR = Path.home() / ".cache" / "kalshi-rt" / "snapshots"
SNAPSHOT_FILE = SNAPSHOT_DIR / "market_snapshots.jsonl"
HEARTBEAT_FILE = Path.home() / ".cache" / "kalshi-rt" / "snapshot_heartbeat.json"
ERROR_LOG = Path.home() / ".cache" / "kalshi-rt" / "snapshot_errors.log"

# Stop after this many consecutive failures -- something is structurally broken.
MAX_CONSECUTIVE_FAILURES = 5


def take_snapshot(critic_db=None, kalshi=None, mapper=None):
    """Capture current state of all active RT markets + model predictions."""
    if critic_db is None:
        critic_db = CriticDatabase()
    if kalshi is None:
        kalshi = KalshiClient()
    if mapper is None:
        mapper = TickerMapper()

    events = kalshi.get_rt_events()
    timestamp = datetime.now(timezone.utc).isoformat()
    snapshots = []

    print(f"Snapshotting {len(events)} active RT events at {timestamp[:19]}", flush=True)

    for event in events:
        ticker = event["event_ticker"]
        movie = event["movie_name"]

        # Get market prices
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
            summary = get_movie_summary(rt_slug)
            if summary:
                ems_id = summary.get("ems_id")
                if ems_id:
                    reviews = scrape_reviews(ems_id, slug=rt_slug, max_pages=5)
                    if reviews:
                        # Get close_time from first market for completion estimate
                        close_time = None
                        if markets:
                            close_time = markets[0].get("close_time")
                        pred = predict_distribution(reviews, critic_db, movie_summary=summary, close_time=close_time)
                        calibrated = calibrate_thresholds(pred["threshold_probs"], pred["n_reviews"])

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
                            "threshold_probs": {str(k): round(v, 4) for k, v in calibrated.items()},
                        }

                        # Compute edges at each threshold
                        edges = {}
                        for t, prob in calibrated.items():
                            mkt = market_data.get(t, {})
                            yes_ask = mkt.get("yes_ask") or mkt.get("yes_price")
                            if yes_ask:
                                edges[str(t)] = round((prob - yes_ask) * 100, 1)
                        model_data["edges"] = edges

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
        _save_snapshot(snapshot)

        status = "no reviews"
        if model_data:
            best_edge = max(model_data.get("edges", {}).values(), default=0, key=abs)
            status = f"{model_data['n_reviews']} reviews, mean={model_data['model_mean']}, best_edge={best_edge}%"
        print(f"  {movie}: {status}", flush=True)

    print(f"\n{len(snapshots)} snapshots saved", flush=True)
    return snapshots


def _save_snapshot(snapshot):
    """Append a snapshot to the JSONL log."""
    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
    with open(SNAPSHOT_FILE, "a") as f:
        f.write(json.dumps(snapshot) + "\n")


def _write_heartbeat(n_events):
    """Record that a snapshot cycle completed successfully."""
    HEARTBEAT_FILE.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if HEARTBEAT_FILE.exists():
        try:
            with open(HEARTBEAT_FILE) as f:
                data = json.load(f)
        except Exception:
            pass
    data["timestamp"] = datetime.now(timezone.utc).isoformat()
    data["n_events"] = n_events
    data["count"] = data.get("count", 0) + 1
    with open(HEARTBEAT_FILE, "w") as f:
        json.dump(data, f)


def _log_error(exc, consecutive_failures):
    """Append an exception to the error log."""
    ERROR_LOG.parent.mkdir(parents=True, exist_ok=True)
    ts = datetime.now(timezone.utc).isoformat()
    with open(ERROR_LOG, "a") as f:
        f.write(f"\n[{ts}] failure #{consecutive_failures}\n")
        f.write(traceback.format_exc() if not isinstance(exc, str) else exc)
        f.write("\n")


def get_status():
    """Return current snapshot loop health as a dict."""
    if not HEARTBEAT_FILE.exists():
        return {"healthy": False, "reason": "heartbeat file missing -- loop may never have run"}

    with open(HEARTBEAT_FILE) as f:
        hb = json.load(f)

    last_ts = datetime.fromisoformat(hb["timestamp"])
    age_minutes = (datetime.now(timezone.utc) - last_ts).total_seconds() / 60

    # Healthy if last snapshot was within 90 min (1.5x the 60-min interval)
    healthy = age_minutes < 90
    return {
        "healthy": healthy,
        "last_snapshot": hb["timestamp"][:19],
        "age_minutes": round(age_minutes, 1),
        "n_events_last_run": hb.get("n_events", "?"),
        "total_cycles": hb.get("count", "?"),
        "reason": None if healthy else f"last snapshot was {age_minutes:.0f} min ago (>90 min threshold)",
    }


def load_snapshots(event_ticker=None):
    """Load all snapshots, optionally filtered by event."""
    if not SNAPSHOT_FILE.exists():
        return []
    snapshots = []
    with open(SNAPSHOT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if event_ticker and s["event_ticker"] != event_ticker:
                continue
            snapshots.append(s)
    return snapshots


def resolve_snapshots(event_ticker, actual_score):
    """Mark all snapshots for an event with the actual outcome.

    After settlement, this turns each snapshot into a labeled training example:
    - market_price at time T
    - model_prediction at time T
    - actual_outcome (did score exceed threshold?)

    Returns count of resolved snapshots.
    """
    if not SNAPSHOT_FILE.exists():
        return 0

    all_snapshots = []
    resolved_count = 0
    with open(SNAPSHOT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if s["event_ticker"] == event_ticker and not s["resolved"]:
                s["resolved"] = True
                s["actual_score"] = actual_score
                resolved_count += 1
            all_snapshots.append(s)

    with open(SNAPSHOT_FILE, "w") as f:
        for s in all_snapshots:
            f.write(json.dumps(s) + "\n")

    return resolved_count


def compute_edge_performance():
    """Analyze how model edges performed against actual outcomes.

    This is the key metric: when the model said "market is wrong by X%",
    was the model right? Returns performance broken down by edge size,
    confidence level, and review count.
    """
    snapshots = load_snapshots()
    resolved = [s for s in snapshots if s["resolved"] and s["model"]]

    if not resolved:
        return {"error": "No resolved snapshots with model predictions"}

    results = {
        "total_snapshots": len(resolved),
        "by_edge_bucket": {},
        "by_confidence": {},
        "by_review_count": {},
        "all_trades": [],
    }

    for s in resolved:
        actual = s["actual_score"]
        model = s["model"]
        markets = s["markets"]

        for t_str, prob in model.get("threshold_probs", {}).items():
            t = int(t_str)
            mkt = markets.get(t_str, {})
            yes_price = mkt.get("yes_price")
            if not yes_price or prob is None:
                continue

            # What actually happened
            outcome = 1 if actual > t else 0

            # Model's edge
            edge = prob - yes_price

            # Would we have traded?
            if abs(edge) < config.MIN_EDGE:
                continue

            if edge > 0:
                direction = "BUY YES"
                cost = mkt.get("yes_ask") or yes_price
                win = outcome == 1
            else:
                direction = "BUY NO"
                cost = 1.0 - (mkt.get("yes_bid") or yes_price)
                win = outcome == 0

            pnl = (1.0 - cost) * (1 - config.KALSHI_FEE_RATE) if win else -cost

            trade = {
                "movie": s["movie"],
                "threshold": t,
                "direction": direction,
                "edge": round(edge * 100, 1),
                "model_prob": round(prob * 100, 1),
                "market_price": round(yes_price * 100, 1),
                "cost": round(cost * 100, 1),
                "actual_score": actual,
                "outcome": outcome,
                "win": win,
                "pnl": round(pnl * 100, 1),  # cents per contract
                "confidence": model.get("confidence"),
                "n_reviews": model.get("n_reviews"),
                "timestamp": s["timestamp"],
            }
            results["all_trades"].append(trade)

            # Bucket by edge size
            edge_bucket = f"{int(abs(edge * 100) // 5) * 5}-{int(abs(edge * 100) // 5) * 5 + 5}%"
            if edge_bucket not in results["by_edge_bucket"]:
                results["by_edge_bucket"][edge_bucket] = {"wins": 0, "losses": 0, "pnl": 0}
            bucket = results["by_edge_bucket"][edge_bucket]
            if win:
                bucket["wins"] += 1
            else:
                bucket["losses"] += 1
            bucket["pnl"] += pnl

            # Bucket by confidence
            conf = model.get("confidence", "?")
            if conf not in results["by_confidence"]:
                results["by_confidence"][conf] = {"wins": 0, "losses": 0, "pnl": 0}
            cb = results["by_confidence"][conf]
            if win:
                cb["wins"] += 1
            else:
                cb["losses"] += 1
            cb["pnl"] += pnl

    # Compute win rates
    for bucket_type in ["by_edge_bucket", "by_confidence"]:
        for key, data in results[bucket_type].items():
            total = data["wins"] + data["losses"]
            data["win_rate"] = round(data["wins"] / total * 100, 1) if total > 0 else 0
            data["total"] = total
            data["pnl"] = round(data["pnl"], 2)

    total_trades = len(results["all_trades"])
    total_wins = sum(1 for t in results["all_trades"] if t["win"])
    total_pnl = sum(t["pnl"] for t in results["all_trades"])
    results["summary"] = {
        "total_trades": total_trades,
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0,
        "total_pnl_cents": round(total_pnl, 1),
    }

    return results


def print_edge_report():
    """Print a human-readable edge performance report."""
    perf = compute_edge_performance()

    if "error" in perf:
        print(perf["error"])
        return

    print(f"\n{'='*60}")
    print("EDGE PERFORMANCE vs REAL KALSHI MARKETS")
    print(f"{'='*60}")

    s = perf["summary"]
    print(f"\nTotal trades: {s['total_trades']}")
    print(f"Win rate: {s['win_rate']}%")
    print(f"Total P&L: {s['total_pnl_cents']:.1f}c per contract")

    print(f"\nBy edge size:")
    for bucket in sorted(perf["by_edge_bucket"].keys()):
        d = perf["by_edge_bucket"][bucket]
        print(f"  {bucket}: {d['win_rate']}% win ({d['total']} trades, P&L: {d['pnl']:.1f}c)")

    print(f"\nBy confidence:")
    for conf in ("HIGH", "MEDIUM", "LOW"):
        if conf in perf["by_confidence"]:
            d = perf["by_confidence"][conf]
            print(f"  {conf}: {d['win_rate']}% win ({d['total']} trades, P&L: {d['pnl']:.1f}c)")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Snapshot Kalshi RT market prices + model predictions")
    parser.add_argument("--loop", type=int, help="Re-snapshot every N seconds")
    parser.add_argument("--report", action="store_true", help="Print edge performance report")
    parser.add_argument("--status", action="store_true", help="Check snapshot loop health")
    parser.add_argument("--resolve", nargs=2, metavar=("TICKER", "SCORE"),
                       help="Resolve an event with its actual score")
    args = parser.parse_args()

    if args.status:
        s = get_status()
        icon = "OK" if s["healthy"] else "WARN"
        print(f"[{icon}] Snapshot loop health")
        print(f"  Last run:      {s.get('last_snapshot', 'never')}")
        print(f"  Age:           {s.get('age_minutes', '?')} min")
        print(f"  Events (last): {s.get('n_events_last_run', '?')}")
        print(f"  Total cycles:  {s.get('total_cycles', '?')}")
        if not s["healthy"]:
            print(f"  Problem:       {s.get('reason', '')}")
        return

    if args.report:
        print_edge_report()
        return

    if args.resolve:
        ticker, score = args.resolve[0], int(args.resolve[1])
        n = resolve_snapshots(ticker, score)
        print(f"Resolved {n} snapshots for {ticker} with score={score}")
        return

    consecutive_failures = 0
    while True:
        try:
            snapshots = take_snapshot()
            _write_heartbeat(len(snapshots))
            consecutive_failures = 0
        except Exception as exc:
            consecutive_failures += 1
            print(f"[ERROR] Snapshot failed (attempt #{consecutive_failures}): {exc}", flush=True)
            _log_error(exc, consecutive_failures)
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"[FATAL] {MAX_CONSECUTIVE_FAILURES} consecutive failures -- exiting. Check {ERROR_LOG}", flush=True)
                sys.exit(1)

        if not args.loop:
            break

        # Back off to 5 min on failures, resume normal interval on recovery
        sleep_s = 300 if consecutive_failures > 0 else args.loop
        print(f"Sleeping {sleep_s}s...", flush=True)
        time.sleep(sleep_s)


if __name__ == "__main__":
    main()
