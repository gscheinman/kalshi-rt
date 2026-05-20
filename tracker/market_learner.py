"""
Learn from resolved Kalshi market data.

After markets settle, we know:
  - What the market priced at various points in time
  - What our model predicted at those same points
  - What actually happened

This module computes:
  1. Market calibration -- how well-calibrated are Kalshi prices?
  2. Model calibration -- how well-calibrated are our predictions?
  3. Edge reliability -- when model disagreed with market, who was right?
  4. Optimal edge threshold -- what minimum edge actually generates profit?
  5. Market efficiency by threshold -- which brackets are most mispriced?

These learnings feed back into the trading engine.
"""
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import config

SNAPSHOT_FILE = Path.home() / ".cache" / "kalshi-rt" / "snapshots" / "market_snapshots.jsonl"
LEARNINGS_FILE = Path.home() / ".cache" / "kalshi-rt" / "market_learnings.json"


def load_resolved_trades(snapshot_path=None):
    """Extract all resolved model-vs-market comparisons from snapshots."""
    path = Path(snapshot_path) if snapshot_path else SNAPSHOT_FILE
    if not path.exists():
        return []

    trades = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            s = json.loads(line)
            if not s.get("resolved") or not s.get("model"):
                continue

            actual = s["actual_score"]
            model = s["model"]
            markets = s["markets"]

            for t_str, mkt in markets.items():
                t = int(t_str)
                # Use yes_ask (actual entry cost) over yes_price (midpoint/last trade).
                # Kalshi's displayed % is NOT the real cost to enter a position.
                # Fall back to yes_price only if yes_ask isn't available (older snapshots).
                yes_ask = mkt.get("yes_ask") or mkt.get("yes_price")
                model_prob = model.get("threshold_probs", {}).get(t_str)

                if yes_ask is None or model_prob is None:
                    continue

                outcome = 1 if actual > t else 0
                edge = model_prob - yes_ask

                trades.append({
                    "movie": s["movie"],
                    "event_ticker": s["event_ticker"],
                    "timestamp": s["timestamp"],
                    "threshold": t,
                    "market_price": yes_ask,
                    "model_prob": model_prob,
                    "edge": edge,
                    "outcome": outcome,
                    "actual_score": actual,
                    "confidence": model.get("confidence"),
                    "n_reviews": model.get("n_reviews"),
                    "known_pct": model.get("known_pct"),
                })

    return trades


def compute_learnings(snapshot_path=None):
    """Analyze resolved trades and produce actionable learnings."""
    trades = load_resolved_trades(snapshot_path=snapshot_path)
    if not trades:
        return {"error": "No resolved trades yet. Snapshot markets, wait for settlement, then resolve."}

    learnings = {
        "n_trades": len(trades),
        "n_movies": len(set(t["movie"] for t in trades)),
    }

    # 1. Market calibration: when market says X%, how often does it happen?
    market_cal = _calibration_curve([t["market_price"] for t in trades],
                                     [t["outcome"] for t in trades])
    learnings["market_calibration"] = market_cal

    # 2. Model calibration: same for our model
    model_cal = _calibration_curve([t["model_prob"] for t in trades],
                                    [t["outcome"] for t in trades])
    learnings["model_calibration"] = model_cal

    # 3. Edge reliability: when model saw edge X%, what was the actual win rate?
    edge_buckets = defaultdict(list)
    for t in trades:
        if abs(t["edge"]) < 0.03:
            continue
        bucket = int(abs(t["edge"]) * 100 // 5) * 5
        direction = "buy_yes" if t["edge"] > 0 else "buy_no"
        win = (t["outcome"] == 1) if direction == "buy_yes" else (t["outcome"] == 0)
        cost = t["market_price"] if direction == "buy_yes" else (1 - t["market_price"])
        pnl = (1 - cost) * (1 - config.KALSHI_FEE_RATE) if win else -cost
        edge_buckets[bucket].append({"win": win, "pnl": pnl, "edge": abs(t["edge"])})

    edge_performance = {}
    for bucket, entries in sorted(edge_buckets.items()):
        wins = sum(1 for e in entries if e["win"])
        total = len(entries)
        total_pnl = sum(e["pnl"] for e in entries)
        avg_edge = sum(e["edge"] for e in entries) / total
        edge_performance[f"{bucket}-{bucket+5}%"] = {
            "trades": total,
            "win_rate": round(wins / total * 100, 1),
            "total_pnl": round(total_pnl, 4),
            "avg_pnl_per_trade": round(total_pnl / total, 4) if total > 0 else 0,
            "avg_edge": round(avg_edge * 100, 1),
        }
    learnings["edge_performance"] = edge_performance

    # 4. Optimal minimum edge: find the edge threshold that maximizes cumulative P&L
    all_edge_trades = []
    for t in trades:
        if abs(t["edge"]) < 0.01:
            continue
        direction = "buy_yes" if t["edge"] > 0 else "buy_no"
        win = (t["outcome"] == 1) if direction == "buy_yes" else (t["outcome"] == 0)
        cost = t["market_price"] if direction == "buy_yes" else (1 - t["market_price"])
        pnl = (1 - cost) * (1 - config.KALSHI_FEE_RATE) if win else -cost
        all_edge_trades.append({"abs_edge": abs(t["edge"]), "pnl": pnl})

    all_edge_trades.sort(key=lambda x: x["abs_edge"])

    best_min_edge = 0.05
    best_pnl_per = -999
    for min_e in [i * 0.01 for i in range(1, 30)]:
        subset = [t for t in all_edge_trades if t["abs_edge"] >= min_e]
        if len(subset) < 5:
            continue
        avg_pnl = sum(t["pnl"] for t in subset) / len(subset)
        if avg_pnl > best_pnl_per:
            best_pnl_per = avg_pnl
            best_min_edge = min_e

    learnings["optimal_min_edge"] = round(best_min_edge, 2)
    learnings["optimal_min_edge_pnl_per_trade"] = round(best_pnl_per, 4)

    # 5. Market efficiency by threshold: which thresholds are most mispriced?
    threshold_performance = defaultdict(list)
    for t in trades:
        if abs(t["edge"]) < config.MIN_EDGE:
            continue
        direction = "buy_yes" if t["edge"] > 0 else "buy_no"
        win = (t["outcome"] == 1) if direction == "buy_yes" else (t["outcome"] == 0)
        cost = t["market_price"] if direction == "buy_yes" else (1 - t["market_price"])
        pnl = (1 - cost) * (1 - config.KALSHI_FEE_RATE) if win else -cost
        threshold_performance[t["threshold"]].append({"win": win, "pnl": pnl})

    threshold_analysis = {}
    for threshold, entries in sorted(threshold_performance.items()):
        wins = sum(1 for e in entries if e["win"])
        total = len(entries)
        total_pnl = sum(e["pnl"] for e in entries)
        threshold_analysis[threshold] = {
            "trades": total,
            "win_rate": round(wins / total * 100, 1),
            "total_pnl": round(total_pnl, 4),
        }
    learnings["threshold_efficiency"] = threshold_analysis

    # 6. Confidence level performance
    conf_perf = defaultdict(list)
    for t in trades:
        if abs(t["edge"]) < config.MIN_EDGE:
            continue
        direction = "buy_yes" if t["edge"] > 0 else "buy_no"
        win = (t["outcome"] == 1) if direction == "buy_yes" else (t["outcome"] == 0)
        cost = t["market_price"] if direction == "buy_yes" else (1 - t["market_price"])
        pnl = (1 - cost) * (1 - config.KALSHI_FEE_RATE) if win else -cost
        conf_perf[t["confidence"]].append({"win": win, "pnl": pnl})

    confidence_analysis = {}
    for conf, entries in conf_perf.items():
        wins = sum(1 for e in entries if e["win"])
        total = len(entries)
        total_pnl = sum(e["pnl"] for e in entries)
        confidence_analysis[conf] = {
            "trades": total,
            "win_rate": round(wins / total * 100, 1),
            "avg_pnl": round(total_pnl / total, 4) if total > 0 else 0,
        }
    learnings["confidence_performance"] = confidence_analysis

    # Save
    output_path = LEARNINGS_FILE
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(learnings, f, indent=2)

    return learnings


def _calibration_curve(probs, outcomes, n_buckets=10):
    """Compute calibration: predicted probability vs actual frequency."""
    buckets = defaultdict(list)
    for p, o in zip(probs, outcomes):
        bucket = min(int(p * n_buckets), n_buckets - 1)
        buckets[bucket].append(o)

    curve = {}
    for b, outcomes_list in sorted(buckets.items()):
        predicted = (b + 0.5) / n_buckets
        actual = sum(outcomes_list) / len(outcomes_list)
        curve[f"{int(predicted*100)}%"] = {
            "predicted": round(predicted * 100, 1),
            "actual": round(actual * 100, 1),
            "count": len(outcomes_list),
            "error": round(abs(actual - predicted) * 100, 1),
        }
    return curve


def apply_learnings():
    """Read saved learnings and suggest config updates."""
    if not LEARNINGS_FILE.exists():
        print("No learnings file. Run compute_learnings() first after resolving snapshots.")
        return

    with open(LEARNINGS_FILE) as f:
        learnings = json.load(f)

    print(f"\n{'='*60}")
    print("LEARNINGS FROM REAL KALSHI MARKET DATA")
    print(f"{'='*60}")
    print(f"\nBased on {learnings['n_trades']} trades across {learnings['n_movies']} movies")

    # Optimal edge
    opt_edge = learnings.get("optimal_min_edge", 0.05)
    current_edge = config.MIN_EDGE
    print(f"\nOptimal min edge: {opt_edge*100:.0f}% (current: {current_edge*100:.0f}%)")
    if abs(opt_edge - current_edge) > 0.01:
        print(f"  -> Suggest changing MIN_EDGE from {current_edge} to {opt_edge}")

    # Edge performance
    print(f"\nEdge performance:")
    for bucket, data in learnings.get("edge_performance", {}).items():
        print(f"  {bucket}: {data['win_rate']}% win, avg P&L={data['avg_pnl_per_trade']:.1%} ({data['trades']} trades)")

    # Confidence
    print(f"\nConfidence performance:")
    for conf in ("HIGH", "MEDIUM", "LOW"):
        data = learnings.get("confidence_performance", {}).get(conf)
        if data:
            print(f"  {conf}: {data['win_rate']}% win, avg P&L={data['avg_pnl']:.1%} ({data['trades']} trades)")

    # Threshold efficiency
    print(f"\nMost mispriced thresholds:")
    thresh = learnings.get("threshold_efficiency", {})
    sorted_thresh = sorted(thresh.items(), key=lambda x: x[1]["total_pnl"], reverse=True)
    for t, data in sorted_thresh[:5]:
        print(f"  Above {t}%: {data['win_rate']}% win, P&L={data['total_pnl']:.2f} ({data['trades']} trades)")


def _run_test():
    """Validate the learnings pipeline with synthetic resolved snapshots.

    Creates 3 synthetic movies, each with 5 snapshots, resolved with known
    outcomes. Checks that compute_learnings() runs without error and returns
    the expected structure.
    """
    import tempfile, os

    # Build 15 synthetic snapshots across 3 movies
    movies = [
        ("KXRT-TEST1", "Test Movie A", 72),
        ("KXRT-TEST2", "Test Movie B", 45),
        ("KXRT-TEST3", "Test Movie C", 88),
    ]

    snapshots = []
    for ticker, movie, actual_score in movies:
        for i in range(5):
            model_prob_60 = 0.65 if actual_score > 60 else 0.35
            market_price_60 = 0.55 if actual_score > 60 else 0.45
            # yes_ask is what we actually compare against (real entry cost)
            yes_ask_60 = market_price_60 + 0.02
            snap = {
                "timestamp": f"2026-05-19T{10+i:02d}:00:00+00:00",
                "event_ticker": ticker,
                "movie": movie,
                "rt_slug": f"m/{movie.lower().replace(' ', '_')}",
                "markets": {
                    "60": {"yes_price": market_price_60, "yes_bid": market_price_60 - 0.02,
                           "yes_ask": yes_ask_60, "volume": 100},
                    "70": {"yes_price": 0.40, "yes_bid": 0.38, "yes_ask": 0.42, "volume": 80},
                },
                "total_volume": 180,
                "model": {
                    "n_reviews": 20 + i,
                    "n_known": 15,
                    "known_pct": 0.75,
                    "naive_pct": 0.70,
                    "model_mean": actual_score + (i - 2),
                    "confidence": "HIGH" if i > 2 else "MEDIUM",
                    "threshold_probs": {
                        "60": model_prob_60,
                        "70": 0.50 if actual_score > 70 else 0.30,
                    },
                },
                "resolved": True,
                "actual_score": actual_score,
            }
            snapshots.append(snap)

    with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as tmp:
        for s in snapshots:
            tmp.write(json.dumps(s) + "\n")
        tmp_path = tmp.name

    try:
        trades = load_resolved_trades(snapshot_path=tmp_path)
        assert len(trades) > 0, "load_resolved_trades() returned no trades"

        learnings = compute_learnings(snapshot_path=tmp_path)
        assert "error" not in learnings, f"compute_learnings() returned error: {learnings}"
        assert learnings["n_trades"] == len(trades), "trade count mismatch"
        assert learnings["n_movies"] == 3, f"expected 3 movies, got {learnings['n_movies']}"
        assert "edge_performance" in learnings, "missing edge_performance"
        assert "market_calibration" in learnings, "missing market_calibration"
        assert "model_calibration" in learnings, "missing model_calibration"
        assert "optimal_min_edge" in learnings, "missing optimal_min_edge"

        print("market_learner.py validation PASSED")
        print(f"  Synthetic trades: {len(trades)}")
        print(f"  Movies: {learnings['n_movies']}")
        print(f"  Optimal min edge: {learnings['optimal_min_edge']*100:.0f}%")
        print(f"  Edge buckets found: {list(learnings['edge_performance'].keys())}")
        return True
    except AssertionError as e:
        print(f"VALIDATION FAILED: {e}")
        return False
    finally:
        os.unlink(tmp_path)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Learn from resolved Kalshi market data")
    parser.add_argument("--compute", action="store_true", help="Compute learnings from resolved snapshots")
    parser.add_argument("--apply", action="store_true", help="Show suggested config changes from learnings")
    parser.add_argument("--test", action="store_true", help="Validate pipeline with synthetic data")
    args = parser.parse_args()

    if args.test:
        ok = _run_test()
        sys.exit(0 if ok else 1)
    elif args.compute:
        learnings = compute_learnings()
        if "error" in learnings:
            print(learnings["error"])
        else:
            print(f"Computed learnings from {learnings['n_trades']} trades across {learnings['n_movies']} movies")
            print(f"Saved to {LEARNINGS_FILE}")
    elif args.apply:
        apply_learnings()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
