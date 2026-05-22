"""
Paper trading engine for CI/GitHub Actions.

Runs the same portfolio optimizer as paper_trader.py but writes trades to
a repo-local file (data/paper_trades.jsonl) so GitHub Actions can commit
the results back. Also maintains data/positions.json for position tracking.

Deduplication: if a position already exists for the same event+threshold+direction,
it skips re-entry (no duplicate trades within the same market). New trades are only
logged when a genuinely new opportunity appears or an existing position should be
adjusted.

Runs every 4 hours via GitHub Actions cron.
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
from engine.portfolio import optimize_portfolio
import config

REPO_DIR = Path(__file__).parent.parent
TRADE_FILE = REPO_DIR / "data" / "paper_trades.jsonl"
POSITION_FILE = REPO_DIR / "data" / "positions.json"


def load_positions():
    """Load existing positions from repo-local file."""
    if not POSITION_FILE.exists():
        return {}
    try:
        with open(POSITION_FILE) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}


def save_positions(positions):
    """Save positions to repo-local file."""
    POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITION_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def has_existing_position(positions, event_ticker, threshold, direction):
    """Check if we already have a position for this event+threshold+direction."""
    event_positions = positions.get(event_ticker, [])
    for pos in event_positions:
        if pos.get("threshold") == threshold and pos.get("direction") == direction:
            return True
    return False


def record_trade(positions, event_ticker, trade):
    """Add a trade to position tracking."""
    if event_ticker not in positions:
        positions[event_ticker] = []
    positions[event_ticker].append({
        "timestamp": trade["timestamp"],
        "market_ticker": trade["market_ticker"],
        "direction": trade["direction"],
        "threshold": trade["threshold"],
        "model_prob": trade.get("model_prob"),
        "edge": trade.get("edge"),
        "suggested_size": trade.get("suggested_size"),
        "avg_fill": trade.get("avg_fill"),
        "contracts": trade.get("contracts"),
    })


def run_paper_pass():
    """Single paper trading pass across all active RT markets."""
    critic_db = CriticDatabase()
    kalshi = KalshiClient()
    mapper = TickerMapper()
    bankroll = config.DEFAULT_BANKROLL

    events = kalshi.get_rt_events()
    timestamp = datetime.now(timezone.utc).isoformat()
    positions = load_positions()
    new_trades = []

    print(f"Paper trading pass at {timestamp[:19]} -- {len(events)} active events", flush=True)

    for event in events:
        ticker = event["event_ticker"]
        movie = event["movie_name"]
        rt_slug = mapper.get_rt_slug(event)

        if not rt_slug:
            continue

        try:
            summary = get_movie_summary(rt_slug)
            if not summary:
                continue

            ems_id = summary.get("ems_id")
            if not ems_id:
                continue

            reviews = scrape_reviews(
                ems_id, slug=rt_slug,
                expected_count=summary.get("review_count"),
            )
            if not reviews:
                continue

            print(f"  {movie}: {len(reviews)} reviews", flush=True)

            markets = kalshi.get_markets(ticker)
            if not markets:
                continue

            # Get close time for settlement-aware sizing
            close_time = None
            if markets:
                close_time = markets[0].get("close_time")

            # Extract all thresholds from actual Kalshi markets so the model
            # generates probabilities for granular brackets (57%, 58%, 62%, etc.)
            market_thresholds = [m["threshold"] for m in markets if m.get("threshold") is not None]

            prediction = predict_distribution(
                reviews, critic_db, movie_summary=summary, close_time=close_time,
                extra_thresholds=market_thresholds,
            )
            if prediction["n_reviews"] == 0:
                continue

            # Get settlement date for time-based sizing
            settlement_date = None
            close_str = markets[0].get("close_time", "") if markets else ""
            if close_str:
                try:
                    settlement_date = datetime.fromisoformat(
                        close_str.replace("Z", "+00:00")
                    )
                except (ValueError, TypeError):
                    pass

            existing = positions.get(ticker, [])

            portfolio = optimize_portfolio(
                prediction, markets, kalshi_client=kalshi,
                bankroll=bankroll, existing_positions=existing,
                settlement_date=settlement_date,
            )

            if not portfolio["trades"]:
                print(f"    No alpha found", flush=True)
                continue

            pnl = portfolio["portfolio_pnl"]
            if pnl["expected_pnl"] <= 0:
                print(f"    Portfolio E[P&L] negative, skipping", flush=True)
                continue

            print(f"    PORTFOLIO: {portfolio['n_positions']} positions, "
                  f"${portfolio['total_spend']:.2f} spend, "
                  f"E[P&L]=${pnl['expected_pnl']:.2f}", flush=True)

            for pt in portfolio["trades"]:
                if pt["type"] != "single":
                    continue  # Skip spreads in CI for simplicity

                # Skip if we already hold this position
                if has_existing_position(positions, ticker, pt["threshold"], pt["direction"]):
                    print(f"      SKIP {pt['direction']} Above {pt['threshold']}% (already held)", flush=True)
                    continue

                trade = {
                    "timestamp": timestamp,
                    "movie": movie,
                    "event_ticker": ticker,
                    "rt_slug": rt_slug,
                    "market_ticker": pt["ticker"],
                    "direction": pt["direction"],
                    "threshold": pt["threshold"],
                    "model_prob": pt["win_prob"] if pt["direction"] == "BUY YES" else round(100 - pt["win_prob"], 1),
                    "edge": pt["edge"],
                    "suggested_size": pt["size"],
                    "avg_fill": round(pt["cost_per"] * 100, 1),
                    "contracts": pt["contracts"],
                    "expected_profit": round(
                        pnl["expected_pnl"] * pt["size"] / portfolio["total_spend"], 2
                    ) if portfolio["total_spend"] > 0 else 0,
                    "win_prob": pt["win_prob"],
                    "confidence": prediction["confidence"],
                    "n_reviews": prediction["n_reviews"],
                    "n_known": prediction["n_known"],
                    "model_mean": prediction["model_mean"],
                    "naive_pct": prediction["naive_pct"],
                    "ob_simulated": pt.get("ob_simulated", False),
                    "source": "ci",
                    "live": False,
                    "execution": {"status": "paper", "dry_run": True},
                }

                ob_tag = "OB" if pt.get("ob_simulated") else "quoted"
                print(f"      NEW {pt['direction']} Above {pt['threshold']}% "
                      f"@ {pt['cost_per']*100:.1f}c ({ob_tag}), "
                      f"${pt['size']:.2f} ({pt['contracts']} cts), "
                      f"edge={pt['edge']}%", flush=True)

                new_trades.append(trade)
                record_trade(positions, ticker, trade)

        except Exception as e:
            print(f"  Error processing {movie}: {e}", flush=True)

    # Write new trades
    if new_trades:
        TRADE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(TRADE_FILE, "a") as f:
            for t in new_trades:
                f.write(json.dumps(t) + "\n")
        save_positions(positions)
        print(f"\n{len(new_trades)} new paper trades logged", flush=True)
    else:
        print(f"\nNo new trades this pass", flush=True)

    return new_trades


if __name__ == "__main__":
    run_paper_pass()
