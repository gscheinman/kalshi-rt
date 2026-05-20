"""
Settlement score capture and resolution.

Kalshi RT markets resolve at specific times (usually Sunday/Monday).
The tomatometer at settlement time is what determines market outcomes.
This script:
  1. Captures the RT score at settlement time
  2. Resolves market snapshots with the actual score
  3. Resolves prediction logger records
  4. Runs learnings computation on newly resolved data

Usage:
    python -m tracker.settlement                    # show all events approaching settlement
    python -m tracker.settlement --snapshot         # capture current scores for all active events
    python -m tracker.settlement --resolve TICKER SCORE  # resolve an event with its actual score
    python -m tracker.settlement --resolve-all      # auto-resolve any events past their close time

Best run shortly after market close time.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

from market.kalshi_client import KalshiClient
from scraper.rt_page import get_movie_summary
from market.mapper import TickerMapper
from tracker.market_snapshot import resolve_snapshots, load_snapshots
from tracker.market_learner import compute_learnings

SNAPSHOTS_DIR = Path.home() / ".cache" / "kalshi-rt"
SETTLEMENT_LOG = SNAPSHOTS_DIR / "settlement_log.jsonl"


def show_upcoming():
    """Show all active events with their settlement dates and current scores."""
    client = KalshiClient()
    mapper = TickerMapper()
    events = client.get_rt_events()

    print(f"\n{'='*70}")
    print("ACTIVE KALSHI RT EVENTS")
    print(f"{'='*70}\n")

    now = datetime.now(timezone.utc)

    for event in events:
        ticker = event["event_ticker"]
        movie = event["movie_name"]

        # Get close time from first market
        markets = client.get_markets(ticker)
        close_time = None
        if markets:
            close_str = markets[0].get("close_time", "")
            if close_str:
                try:
                    close_time = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
                except ValueError:
                    pass

        # Get current RT score
        rt_slug = mapper.get_rt_slug(event)
        score = None
        review_count = 0
        if rt_slug:
            summary = get_movie_summary(rt_slug)
            if summary:
                score = summary.get("tomatometer")
                review_count = summary.get("review_count", 0)

        # Count snapshots we have
        snapshots = load_snapshots(event_ticker=ticker)
        unresolved = [s for s in snapshots if not s.get("resolved")]

        # Status
        if close_time:
            delta = close_time - now
            if delta.total_seconds() < 0:
                time_str = f"CLOSED {abs(delta.days)}d ago"
            elif delta.days > 0:
                time_str = f"closes in {delta.days}d"
            else:
                hours = delta.seconds // 3600
                time_str = f"closes in {hours}h"
        else:
            time_str = "close time unknown"

        score_str = f"{score}% ({review_count} reviews)" if score else "no score yet"
        snap_str = f"{len(unresolved)} unresolved snapshots"

        print(f"  {ticker}: {movie}")
        print(f"    RT: {score_str} | {time_str} | {snap_str}")
        print()


def snapshot_scores():
    """Capture current RT scores for all active events."""
    client = KalshiClient()
    mapper = TickerMapper()
    events = client.get_rt_events()
    now = datetime.now(timezone.utc).isoformat()

    SNAPSHOTS_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Capturing settlement scores at {now[:19]}...\n")

    entries = []
    for event in events:
        ticker = event["event_ticker"]
        movie = event["movie_name"]
        rt_slug = mapper.get_rt_slug(event)

        score = None
        review_count = 0
        if rt_slug:
            summary = get_movie_summary(rt_slug)
            if summary:
                score = summary.get("tomatometer")
                review_count = summary.get("review_count", 0)

        entry = {
            "timestamp": now,
            "event_ticker": ticker,
            "movie": movie,
            "rt_slug": rt_slug,
            "tomatometer": score,
            "review_count": review_count,
        }
        entries.append(entry)
        score_str = f"{score}% ({review_count} reviews)" if score else "no score"
        print(f"  {movie}: {score_str}")

    with open(SETTLEMENT_LOG, "a") as f:
        for e in entries:
            f.write(json.dumps(e) + "\n")

    print(f"\n{len(entries)} scores saved to {SETTLEMENT_LOG}")
    return entries


def resolve_event(event_ticker, actual_score):
    """Resolve all snapshots for an event with the actual settlement score.

    This is the key step that turns snapshots into training data:
    each snapshot becomes a labeled example of (market_price, model_prediction, actual_outcome).
    """
    print(f"\nResolving {event_ticker} with score={actual_score}%")

    # Resolve market snapshots
    n = resolve_snapshots(event_ticker, actual_score)
    print(f"  Resolved {n} market snapshots")

    # Also try resolving prediction logger records
    try:
        from tracker.logger import load_predictions, save_predictions
        records = load_predictions()
        resolved_preds = 0
        for r in records:
            if r.get("event_ticker") == event_ticker and not r.get("resolved"):
                r["actual_score"] = actual_score
                r["resolved"] = True
                r["settlement_timestamp"] = datetime.now(timezone.utc).isoformat()
                try:
                    from tracker.resolver import _compute_pnl
                    r["pnl"] = _compute_pnl(r, actual_score)
                except Exception:
                    pass
                resolved_preds += 1
        if resolved_preds > 0:
            save_predictions(records)
            print(f"  Resolved {resolved_preds} prediction records")
    except Exception as e:
        print(f"  (prediction logger: {e})")

    # Recompute learnings with new data
    print(f"\n  Computing updated learnings...")
    learnings = compute_learnings()
    if "error" not in learnings:
        print(f"  Learnings updated: {learnings['n_trades']} trades across {learnings['n_movies']} movies")
    else:
        print(f"  {learnings['error']}")

    return n


def resolve_all_closed():
    """Auto-resolve any events past their close time by fetching current RT scores."""
    client = KalshiClient()
    mapper = TickerMapper()
    events = client.get_rt_events()
    now = datetime.now(timezone.utc)

    resolved_count = 0
    for event in events:
        ticker = event["event_ticker"]
        movie = event["movie_name"]

        # Check if past close time
        markets = client.get_markets(ticker)
        if not markets:
            continue

        close_str = markets[0].get("close_time", "")
        if not close_str:
            continue

        try:
            close_time = datetime.fromisoformat(close_str.replace("Z", "+00:00"))
        except ValueError:
            continue

        if close_time > now:
            continue  # not yet closed

        # Check if we already resolved this
        snapshots = load_snapshots(event_ticker=ticker)
        unresolved = [s for s in snapshots if not s.get("resolved")]
        if not unresolved:
            continue

        # Get current RT score
        rt_slug = mapper.get_rt_slug(event)
        if not rt_slug:
            print(f"  {movie}: no RT slug, skipping")
            continue

        summary = get_movie_summary(rt_slug)
        if not summary or summary.get("tomatometer") is None:
            print(f"  {movie}: no RT score available")
            continue

        score = summary["tomatometer"]
        review_count = summary.get("review_count", 0)

        if review_count < 10:
            print(f"  {movie}: only {review_count} reviews, skipping (need 10+)")
            continue

        print(f"\n  {movie}: closed, RT score = {score}% ({review_count} reviews)")
        n = resolve_event(ticker, score)
        resolved_count += 1

    if resolved_count == 0:
        print("\nNo events ready to resolve.")
    else:
        print(f"\nResolved {resolved_count} events.")


def settle_morning():
    """Full settlement-morning workflow in one command.

    Run this shortly after 10 AM ET on settlement Monday for each batch of movies.
    Steps:
      1. Snapshot current RT scores (timestamped record of what scores were at settlement)
      2. Resolve any events whose close_time has passed (turns snapshots into training data)
      3. Run edge performance report (how did the model's edges perform?)
      4. Show learnings (optimal min edge, confidence breakdown)
    """
    print("=" * 60)
    print("SETTLEMENT MORNING RUN")
    print(f"Started: {datetime.now(timezone.utc).isoformat()[:19]} UTC")
    print("=" * 60)

    print("\n[1/4] Snapshotting current RT scores...")
    snapshot_scores()

    print("\n[2/4] Resolving closed events...")
    resolve_all_closed()

    print("\n[3/4] Edge performance report...")
    from tracker.market_snapshot import print_edge_report
    print_edge_report()

    print("\n[4/4] Market learnings...")
    from tracker.market_learner import apply_learnings, compute_learnings
    learnings = compute_learnings()
    if "error" in learnings:
        print(f"  {learnings['error']}")
    else:
        apply_learnings()

    print("\nDone.")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Settlement score capture and resolution")
    parser.add_argument("--snapshot", action="store_true",
                        help="Capture current RT scores for all active events")
    parser.add_argument("--resolve", nargs=2, metavar=("TICKER", "SCORE"),
                        help="Resolve an event with its actual score")
    parser.add_argument("--resolve-all", action="store_true",
                        help="Auto-resolve any events past their close time")
    parser.add_argument("--settle-morning", action="store_true",
                        help="Full settlement workflow: snapshot + resolve + edge report + learnings")
    args = parser.parse_args()

    if args.settle_morning:
        settle_morning()
    elif args.snapshot:
        snapshot_scores()
    elif args.resolve:
        ticker, score = args.resolve[0], int(args.resolve[1])
        resolve_event(ticker, score)
    elif args.resolve_all:
        resolve_all_closed()
    else:
        show_upcoming()


if __name__ == "__main__":
    main()
