"""
Paper trading engine.

Monitors active Kalshi RT markets, runs the model, and "places" simulated
trades using the alpha engine's recommendations. Every trade is logged with
full context so we can measure model performance before risking real money.

Two modes:
  - paper: Simulates trades against the real orderbook but doesn't place orders.
  - live: Places real orders via the authenticated Kalshi client.

Usage:
    python -m engine.paper_trader              # single pass, paper mode
    python -m engine.paper_trader --live       # single pass, live mode (real money!)
    python -m engine.paper_trader --loop 300   # re-check every 5 min
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from data.critics import CriticDatabase
from market.kalshi_client import KalshiClient
from market.mapper import TickerMapper
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
from engine.alpha import find_alpha
from engine.portfolio import optimize_portfolio, load_positions, record_trade
from scraper.rt_page import get_movie_summary
from scraper.rt_reviews import scrape_reviews
import config

TRADE_LOG_DIR = Path.home() / ".cache" / "kalshi-rt"
TRADE_LOG_FILE = TRADE_LOG_DIR / "paper_trades.jsonl"


def _get_event_exposure(event_ticker, trades_this_pass):
    """Sum up dollars already committed to this event in the current pass."""
    return sum(t["suggested_size"] for t in trades_this_pass if t["event_ticker"] == event_ticker)


def _parse_settlement_date(event):
    """Extract settlement date from Kalshi event metadata."""
    close_time = event.get("close_time") or event.get("expected_expiration_time")
    if close_time:
        try:
            if isinstance(close_time, str):
                close_time = close_time.replace("Z", "+00:00")
                return datetime.fromisoformat(close_time)
        except (ValueError, TypeError):
            pass
    return None


def run_pass(critic_db, kalshi, mapper, live=False, bankroll=None):
    """Scan all active RT markets, find alpha, log/execute trades."""
    if bankroll is None:
        bankroll = config.DEFAULT_BANKROLL

    auth_client = None
    if live:
        from market.kalshi_auth import KalshiAuthClient
        auth_client = KalshiAuthClient()
        balance = auth_client.get_balance()
        if balance:
            bankroll = balance["balance"]
            print(f"  Live mode. Balance: ${bankroll:.2f}", flush=True)
        else:
            print("  WARNING: Could not fetch balance. Using default bankroll.", flush=True)

    events = kalshi.get_rt_events()
    print(f"Found {len(events)} active RT events", flush=True)

    trades_placed = []
    total_exposure = sum(t["suggested_size"] for t in trades_placed)

    for event in events:
        if total_exposure >= bankroll * config.MAX_TOTAL_EXPOSURE_PCT:
            print(f"  Portfolio exposure cap reached (${total_exposure:.2f}), stopping", flush=True)
            break

        ticker = event["event_ticker"]
        movie = event["movie_name"]
        rt_slug = mapper.get_rt_slug(event)

        if not rt_slug:
            continue

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

        print(f"\n  {movie}: {len(reviews)} reviews", flush=True)

        markets = kalshi.get_markets(ticker)
        settlement_date = _parse_settlement_date(event)
        # Also try to get close_time from market data if not in event
        close_time_str = None
        if settlement_date:
            close_time_str = settlement_date.isoformat()
        elif markets:
            close_time_str = markets[0].get("close_time")

        # Pass actual Kalshi thresholds so model covers granular brackets
        market_thresholds = [m["threshold"] for m in markets if m.get("threshold") is not None]

        prediction = predict_distribution(
            reviews, critic_db, movie_summary=summary, close_time=close_time_str,
            extra_thresholds=market_thresholds,
        )
        if prediction["n_reviews"] == 0:
            continue

        # Load existing positions for this event (for rebalancing)
        all_positions = load_positions()
        existing_positions = all_positions.get(ticker, [])

        # Portfolio optimization: find optimal set of positions across all thresholds
        portfolio = optimize_portfolio(
            prediction, markets, kalshi_client=kalshi,
            bankroll=bankroll, existing_positions=existing_positions,
            settlement_date=settlement_date,
        )

        if not portfolio["trades"]:
            print(f"    No alpha found", flush=True)
            continue

        pnl = portfolio["portfolio_pnl"]
        if pnl["expected_pnl"] <= 0:
            print(f"    Portfolio E[P&L] negative (${pnl['expected_pnl']:.2f}), skipping", flush=True)
            continue

        print(f"    PORTFOLIO: {portfolio['n_positions']} positions, "
              f"${portfolio['total_spend']:.2f} spend, "
              f"E[P&L]=${pnl['expected_pnl']:.2f}, "
              f"max_loss=${pnl['max_loss']:.2f}", flush=True)

        for pt in portfolio["trades"]:
            if pt["type"] == "single":
                trade = {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
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
                    "expected_profit": round(pnl["expected_pnl"] * pt["size"] / portfolio["total_spend"], 2) if portfolio["total_spend"] > 0 else 0,
                    "win_prob": pt["win_prob"],
                    "confidence": prediction["confidence"],
                    "n_reviews": prediction["n_reviews"],
                    "n_known": prediction["n_known"],
                    "model_mean": prediction["model_mean"],
                    "naive_pct": prediction["naive_pct"],
                    "ob_simulated": pt.get("ob_simulated", False),
                    "portfolio_context": {
                        "total_positions": portfolio["n_positions"],
                        "portfolio_expected_pnl": pnl["expected_pnl"],
                        "portfolio_max_loss": pnl["max_loss"],
                    },
                    "live": live,
                    "execution": None,
                }

                ob_tag = "OB" if pt.get("ob_simulated") else "quoted"
                print(f"      {pt['direction']} Above {pt['threshold']}% "
                      f"@ {pt['cost_per']*100:.1f}c ({ob_tag}), "
                      f"${pt['size']:.2f} ({pt['contracts']} cts), "
                      f"edge={pt['edge']}%", flush=True)

                if live and auth_client:
                    result = auth_client.execute_signal(
                        pt["ticker"], pt["direction"],
                        pt["size"], pt["win_prob"] / 100.0,
                        dry_run=False,
                    )
                    trade["execution"] = result
                    print(f"      EXECUTED: {result.get('status')} "
                          f"(filled {result.get('fill_count', 0)} contracts)", flush=True)
                else:
                    trade["execution"] = {"status": "paper", "dry_run": True}

                trades_placed.append(trade)
                total_exposure += pt["size"]
                _log_trade(trade)
                record_trade(ticker, trade)

            elif pt["type"] == "spread":
                # Log spread as two linked trades
                spread_id = f"spread_{pt['t_low']}_{pt['t_high']}_{datetime.now(timezone.utc).strftime('%H%M%S')}"
                for leg, direction, thresh, tckt, cost in [
                    ("yes_leg", "BUY YES", pt["t_low"], pt.get("ticker_low", ""), pt["yes_cost"]),
                    ("no_leg", "BUY NO", pt["t_high"], pt.get("ticker_high", ""), pt["no_cost"]),
                ]:
                    trade = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "movie": movie,
                        "event_ticker": ticker,
                        "rt_slug": rt_slug,
                        "market_ticker": tckt,
                        "direction": direction,
                        "threshold": thresh,
                        "spread_id": spread_id,
                        "spread_leg": leg,
                        "suggested_size": round(pt["size"] / 2, 2),
                        "avg_fill": round(cost * 100, 1),
                        "contracts": pt.get("n_spreads", 0),
                        "confidence": prediction["confidence"],
                        "n_reviews": prediction["n_reviews"],
                        "model_mean": prediction["model_mean"],
                        "naive_pct": prediction["naive_pct"],
                        "live": live,
                        "execution": {"status": "paper", "dry_run": True},
                    }
                    trades_placed.append(trade)
                    _log_trade(trade)
                    record_trade(ticker, trade)

                total_exposure += pt["size"]
                print(f"      SPREAD: YES>{pt['t_low']}% + NO>{pt['t_high']}% "
                      f"${pt['size']:.2f} ({pt.get('n_spreads',0)} spreads) "
                      f"EV={pt['ev_pct']}%", flush=True)

    return trades_placed


def _log_trade(trade):
    """Append trade to the paper trading log."""
    TRADE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    with open(TRADE_LOG_FILE, "a") as f:
        f.write(json.dumps(trade) + "\n")


def load_trades():
    """Load all paper trades."""
    if not TRADE_LOG_FILE.exists():
        return []
    trades = []
    with open(TRADE_LOG_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                trades.append(json.loads(line))
    return trades


def show_summary():
    """Print paper trading performance summary."""
    trades = load_trades()
    if not trades:
        print("No paper trades logged yet.")
        return

    print(f"\nPaper Trading Summary ({len(trades)} trades)")
    print("=" * 60)

    total_spent = sum(t["suggested_size"] for t in trades)
    total_expected = sum(t["expected_profit"] for t in trades)

    by_confidence = {}
    for t in trades:
        conf = t.get("confidence", "?")
        if conf not in by_confidence:
            by_confidence[conf] = []
        by_confidence[conf].append(t)

    print(f"  Total trades: {len(trades)}")
    print(f"  Total capital deployed: ${total_spent:.2f}")
    print(f"  Total expected profit: ${total_expected:.2f}")

    for conf in ("HIGH", "MEDIUM", "LOW"):
        subset = by_confidence.get(conf, [])
        if subset:
            exp = sum(t["expected_profit"] for t in subset)
            print(f"  {conf}: {len(subset)} trades, E[profit]=${exp:.2f}")

    print(f"\nRecent trades:")
    for t in trades[-10:]:
        status = t.get("execution", {}).get("status", "?")
        sizing = t.get("sizing_mult", 1.0)
        print(f"  {t['timestamp'][:10]} {t['movie']}: {t['direction']} Above {t['threshold']}% "
              f"@ {t['avg_fill']}c ${t['suggested_size']:.2f} "
              f"(sizing:{sizing:.2f}x, {status})")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Paper/live trading for Kalshi RT markets")
    parser.add_argument("--live", action="store_true", help="Place real orders (requires credentials)")
    parser.add_argument("--loop", type=int, help="Re-check every N seconds")
    parser.add_argument("--summary", action="store_true", help="Show paper trading summary")
    parser.add_argument("--bankroll", type=float, help="Override bankroll amount")
    args = parser.parse_args()

    if args.summary:
        show_summary()
        return

    print("Initializing...", flush=True)
    critic_db = CriticDatabase()
    kalshi = KalshiClient()
    mapper = TickerMapper()

    mode = "LIVE" if args.live else "PAPER"
    print(f"Mode: {mode}", flush=True)

    if args.live:
        confirm = input("LIVE MODE: Real orders will be placed. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return

    while True:
        print(f"\n--- Pass at {datetime.now().strftime('%H:%M:%S')} ---", flush=True)
        trades = run_pass(critic_db, kalshi, mapper, live=args.live, bankroll=args.bankroll)
        print(f"\n{len(trades)} trades this pass", flush=True)

        if not args.loop:
            break
        print(f"Sleeping {args.loop}s...", flush=True)
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
