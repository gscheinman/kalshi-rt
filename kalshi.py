#!/usr/bin/env python3
"""
Kalshi RT Trading Signal Generator

Usage:
  python signal.py scan                        Scan all active Kalshi RT markets
  python signal.py movie <slug-or-title>       Full analysis for a specific movie
  python signal.py movie <slug> --manual       Manual review entry mode
  python signal.py movie <slug> --reviews f.json  Load reviews from file
  python signal.py watch [--interval 300]      Background monitor
  python signal.py resolve                     Record outcomes for past predictions
  python signal.py dashboard                   Show performance stats
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from data.critics import CriticDatabase
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
from market.kalshi_client import KalshiClient
from market.mapper import TickerMapper
from scraper.rt_page import get_movie_summary
from scraper.fallback import manual_entry, load_reviews_file
from engine.alpha import find_alpha


def cmd_scan(args):
    """Scan all active Kalshi RT markets for opportunities."""
    print("Scanning Kalshi for active RT markets...\n")
    kalshi = KalshiClient()
    mapper = TickerMapper()
    critic_db = CriticDatabase()

    events = kalshi.get_rt_events()
    if not events:
        print("No active RT markets found.")
        return

    print(f"Found {len(events)} RT events\n")
    print(f"{'Movie':<35s} {'RT Score':<12s} {'Reviews':<10s} {'Kalshi Fcst':<12s} {'Status'}")
    print("-" * 85)

    for event in events:
        markets = kalshi.get_markets(event["event_ticker"])
        rt_slug = mapper.get_rt_slug(event)
        summary = get_movie_summary(rt_slug) if rt_slug else None

        tomatometer = f"{summary['tomatometer']}%" if summary and summary.get("tomatometer") is not None else "--"
        review_count = str(summary.get("review_count", "")) if summary else "--"

        # Compute Kalshi implied forecast from market prices
        kalshi_forecast = _compute_kalshi_forecast(markets)

        movie_name = event.get("movie_name", event.get("title", ""))[:34]
        print(f"{movie_name:<35s} {tomatometer:<12s} {review_count:<10s} {kalshi_forecast:<12s} {event.get('subtitle', '')}")

        # Quick alpha check if we have RT data and market prices
        if summary and summary.get("tomatometer") is not None and markets:
            _quick_alpha_flag(summary, markets)


def cmd_movie(args):
    """Full analysis for a specific movie."""
    slug = args.slug
    if not slug.startswith("m/"):
        slug = f"m/{slug}"

    critic_db = CriticDatabase()
    print(f"Loading critic database ({critic_db.count} critics)...\n")

    # Step 1: Get RT summary
    print(f"Fetching RT data for {slug}...")
    summary = get_movie_summary(slug)
    if summary:
        tomatometer = summary.get("tomatometer", "--")
        review_count = summary.get("review_count", "--")
        title = summary.get("title", slug)
        print(f"  {title}: {tomatometer}% ({review_count} reviews)")
    else:
        title = slug
        print(f"  Could not fetch RT data for {slug}")

    # Step 2: Get reviews
    reviews = _get_reviews(args, slug, critic_db)
    if not reviews:
        print("\nNo reviews available. Use --manual to enter reviews or --reviews to load from file.")
        return

    # Step 3: Run model
    print(f"\nRunning model...")
    result = predict_distribution(reviews, critic_db)
    calibrated_probs = calibrate_thresholds(result["threshold_probs"], result["n_reviews"])

    _print_model_summary(result, title, summary)

    # Step 4: Get Kalshi prices
    kalshi = KalshiClient()
    mapper = TickerMapper()

    # Try to find the Kalshi event for this movie
    market_prices = _find_kalshi_markets(kalshi, mapper, title, slug)

    # Step 5: Find alpha
    if market_prices:
        bankroll = args.bankroll if hasattr(args, "bankroll") else 1000
        opportunities = find_alpha(result, market_prices, bankroll=bankroll)
        _print_alpha_table(calibrated_probs, market_prices, opportunities)
    else:
        print("\nNo matching Kalshi market found. Showing model probabilities only.\n")
        _print_probabilities_only(calibrated_probs)

    # Step 6: Log prediction
    _log_prediction(title, slug, result, calibrated_probs, market_prices)


def cmd_watch(args):
    """Background monitor mode."""
    import time
    interval = args.interval if hasattr(args, "interval") else 300

    print(f"Starting background monitor (polling every {interval}s)")
    print("Press Ctrl+C to stop.\n")

    kalshi = KalshiClient()
    mapper = TickerMapper()
    critic_db = CriticDatabase()
    previous_signals = {}

    try:
        while True:
            print(f"\n--- Scan at {_now()} ---")
            events = kalshi.get_rt_events()

            for event in events:
                markets = kalshi.get_markets(event["event_ticker"])
                if not markets:
                    continue

                rt_slug = mapper.get_rt_slug(event)
                if not rt_slug:
                    continue

                summary = get_movie_summary(rt_slug)
                if not summary or summary.get("review_count", 0) < 5:
                    continue

                # Lightweight check: does tomatometer imply mispricing?
                tomatometer = summary.get("tomatometer")
                if tomatometer is None:
                    continue

                for m in markets:
                    threshold = m.get("threshold")
                    yes_price = m.get("yes_price")
                    if threshold is None or yes_price is None:
                        continue

                    implied = tomatometer / 100.0
                    naive_prob = 1.0 if tomatometer > threshold else 0.0
                    # Rough probability based on distance from threshold
                    distance = tomatometer - threshold
                    if abs(distance) < 15:
                        naive_prob = 0.5 + distance / 30.0
                        naive_prob = max(0.05, min(0.95, naive_prob))

                    edge = naive_prob - yes_price
                    if abs(edge) > 0.10:
                        key = f"{event['event_ticker']}_{threshold}"
                        if key not in previous_signals:
                            movie = event.get("movie_name", "Unknown")
                            direction = "BUY YES" if edge > 0 else "BUY NO"
                            print(f"  ALERT: {movie} | Above {threshold}% | {direction} | "
                                  f"RT={tomatometer}% | Kalshi={yes_price:.0%} | Edge={edge:+.0%}")
                            previous_signals[key] = edge

            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def cmd_resolve(args):
    """Record outcomes for past predictions."""
    from tracker.resolver import resolve_predictions
    resolve_predictions()


def cmd_dashboard(args):
    """Show performance statistics."""
    from tracker.resolver import show_dashboard
    show_dashboard()


# --- Helper functions ---

def _get_reviews(args, slug, critic_db):
    """Get reviews from the appropriate source."""
    if hasattr(args, "reviews_file") and args.reviews_file:
        print(f"\nLoading reviews from {args.reviews_file}...")
        return load_reviews_file(args.reviews_file)

    if hasattr(args, "manual") and args.manual:
        return manual_entry()

    # Try Playwright scraper
    print("\nAttempting to scrape reviews via Playwright...")
    try:
        from scraper.rt_reviews import scrape_reviews_sync
        reviews = scrape_reviews_sync(slug, max_pages=5)
        if reviews:
            print(f"  Scraped {len(reviews)} reviews")
            # Show known/unknown breakdown
            known = sum(1 for r in reviews if critic_db.get_critic(r["critic_name"]))
            print(f"  Known critics: {known}/{len(reviews)}")
            return reviews
        else:
            print("  Playwright returned no reviews.")
    except Exception as e:
        print(f"  Playwright failed: {e}")

    # Fallback to manual
    print("\nFalling back to manual entry.")
    return manual_entry()


def _compute_kalshi_forecast(markets):
    """Estimate the implied RT score by interpolating where P(above) crosses 50%."""
    if not markets:
        return "--"
    priced = [(m["threshold"], m["yes_price"]) for m in markets
              if m.get("threshold") is not None and m.get("yes_price") is not None]
    if not priced:
        return "--"
    priced.sort(key=lambda x: x[0])
    for i in range(len(priced) - 1):
        t1, p1 = priced[i]
        t2, p2 = priced[i + 1]
        if p1 >= 0.50 >= p2:
            if p1 == p2:
                return f"~{round((t1 + t2) / 2)}%"
            frac = (p1 - 0.50) / (p1 - p2)
            return f"~{round(t1 + frac * (t2 - t1))}%"
    if all(p >= 0.50 for _, p in priced):
        return f"~{priced[-1][0]}+%"
    if all(p < 0.50 for _, p in priced):
        return f"<{priced[0][0]}%"
    return "--"


def _find_kalshi_markets(kalshi, mapper, title, slug):
    """Try to find Kalshi markets for this movie."""
    events = kalshi.get_rt_events()
    title_lower = title.lower() if title else ""
    slug_clean = slug.replace("m/", "").replace("_", " ").lower()

    for event in events:
        event_movie = event.get("movie_name", "").lower()
        if (title_lower and title_lower in event_movie) or \
           (event_movie and event_movie in title_lower) or \
           (slug_clean and slug_clean in event_movie):
            markets = kalshi.get_markets(event["event_ticker"])
            if markets:
                print(f"\nFound Kalshi market: {event['title']}")
                return markets

    return []


def _print_model_summary(result, title, summary):
    """Print the model's prediction summary."""
    tomatometer = summary.get("tomatometer", "--") if summary else "--"
    review_count = summary.get("review_count", "--") if summary else "--"

    print(f"\n{'=' * 70}")
    print(f"MOVIE: {title}")
    print(f"RT: {tomatometer}% ({review_count} reviews)  |  "
          f"Model input: {result['n_reviews']} reviews ({result['n_known']} known critics)")
    print(f"{'=' * 70}")
    print(f"\n  Model prediction:  {result['model_mean']}%")
    print(f"  Naive (raw %):     {result['naive_pct']}%")
    print(f"  95% CI:            [{result['model_ci'][0]}%, {result['model_ci'][1]}%]")
    print(f"  Confidence:        {result['confidence']} ({result['known_pct']}% known critics)")

    # Show top bellwethers and contrarians
    details = result.get("critic_details", [])
    known = [d for d in details if d["known"]]
    if known:
        known_sorted = sorted(known, key=lambda x: x["weight"], reverse=True)
        bellwethers = [d for d in known_sorted[:5] if d["tier"] in ("Bellwether", "Reliable")]
        contrarians = [d for d in known_sorted if d["tier"] in ("Contrarian", "Mild Contrarian")]

        if bellwethers:
            fresh = sum(1 for d in bellwethers if d["sentiment"] == "Fresh")
            names = ", ".join(d["critic_name"] for d in bellwethers[:3])
            print(f"\n  Bellwethers: {fresh}/{len(bellwethers)} Fresh ({names})")
        if contrarians:
            fresh = sum(1 for d in contrarians if d["sentiment"] == "Fresh")
            names = ", ".join(d["critic_name"] for d in contrarians[:3])
            print(f"  Contrarians: {fresh}/{len(contrarians)} Fresh ({names})")


def _print_alpha_table(calibrated_probs, market_prices, opportunities):
    """Print the threshold analysis table with alpha signals."""
    print(f"\n{'Threshold':<12s} {'Model P':>10s} {'Kalshi':>10s} {'Edge':>8s} {'Signal':>10s} {'Size':>8s}")
    print("-" * 62)

    # Build market lookup
    market_lookup = {}
    for m in market_prices:
        if m.get("threshold") is not None:
            market_lookup[m["threshold"]] = m

    opp_lookup = {}
    for o in opportunities:
        opp_lookup[o["threshold"]] = o

    best_opp = opportunities[0] if opportunities else None

    for t in sorted(calibrated_probs.keys()):
        model_p = calibrated_probs[t]
        market = market_lookup.get(t)
        opp = opp_lookup.get(t)

        if market and market.get("yes_price") is not None:
            kalshi_str = f"{market['yes_price']:.0%}"
        else:
            kalshi_str = "--"

        if opp:
            edge_str = f"{opp['edge']:+.1f}%"
            signal_str = opp["direction"]
            size_str = f"${opp['suggested_size']:.0f}"
            marker = "  <-- BEST" if best_opp and opp["threshold"] == best_opp["threshold"] else ""
        else:
            edge_str = "--"
            signal_str = "--"
            size_str = "--"
            marker = ""

        print(f"Above {t:<5d}  {model_p:>9.1%}  {kalshi_str:>10s}  {edge_str:>8s}  {signal_str:>10s}  {size_str:>8s}{marker}")

    if best_opp:
        print(f"\nTOP SIGNAL: {best_opp['direction']} \"Above {best_opp['threshold']}%\" "
              f"at {best_opp['market_prob']:.0f}c  "
              f"(model: {best_opp['model_prob']:.1f}%, edge: {best_opp['edge']:+.1f}%)")
        print(f"Suggested size: ${best_opp['suggested_size']:.0f} ({best_opp['confidence']} confidence)")
    else:
        print("\nNo signals above minimum edge threshold.")


def _print_probabilities_only(calibrated_probs):
    """Print model probabilities when no Kalshi market is available."""
    print(f"{'Threshold':<12s} {'Model P(above)':>15s}")
    print("-" * 30)
    for t in sorted(calibrated_probs.keys()):
        print(f"Above {t:<5d}  {calibrated_probs[t]:>14.1%}")


def _quick_alpha_flag(summary, markets):
    """Lightweight alpha check for scan mode."""
    tomatometer = summary.get("tomatometer")
    if tomatometer is None:
        return
    for m in markets:
        threshold = m.get("threshold")
        yes_price = m.get("yes_price")
        if threshold is None or yes_price is None:
            continue
        if tomatometer > threshold + 10 and yes_price < 0.60:
            print(f"    ^ Possible alpha: Above {threshold}% at {yes_price:.0%} but RT is {tomatometer}%")
        elif tomatometer < threshold - 10 and yes_price > 0.40:
            print(f"    ^ Possible alpha: Above {threshold}% at {yes_price:.0%} but RT is {tomatometer}%")


def _log_prediction(title, slug, result, calibrated_probs, market_prices):
    """Log the prediction for future tracking."""
    try:
        from tracker.logger import log_prediction
        market_dict = {}
        for m in (market_prices or []):
            if m.get("threshold") is not None and m.get("yes_price") is not None:
                market_dict[m["threshold"]] = m["yes_price"]
        log_prediction(
            movie=title,
            rt_slug=slug,
            model_result=result,
            calibrated_probs=calibrated_probs,
            kalshi_prices=market_dict,
        )
    except Exception:
        pass  # Don't let logging failures break the tool


def _now():
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def main():
    parser = argparse.ArgumentParser(description="Kalshi RT Trading Signals")
    subparsers = parser.add_subparsers(dest="command")

    # scan
    subparsers.add_parser("scan", help="Scan all active Kalshi RT markets")

    # movie
    movie_parser = subparsers.add_parser("movie", help="Analyze a specific movie")
    movie_parser.add_argument("slug", help="RT movie slug (e.g. m/scary_movie or just scary_movie)")
    movie_parser.add_argument("--manual", action="store_true", help="Manual review entry")
    movie_parser.add_argument("--reviews", dest="reviews_file", help="Load reviews from JSON file")
    movie_parser.add_argument("--bankroll", type=float, default=1000, help="Bankroll for sizing (default: $1000)")

    # watch
    watch_parser = subparsers.add_parser("watch", help="Background monitor")
    watch_parser.add_argument("--interval", type=int, default=300, help="Poll interval in seconds (default: 300)")

    # resolve
    subparsers.add_parser("resolve", help="Record outcomes for past predictions")

    # dashboard
    subparsers.add_parser("dashboard", help="Show performance stats")

    args = parser.parse_args()

    if args.command == "scan":
        cmd_scan(args)
    elif args.command == "movie":
        cmd_movie(args)
    elif args.command == "watch":
        cmd_watch(args)
    elif args.command == "resolve":
        cmd_resolve(args)
    elif args.command == "dashboard":
        cmd_dashboard(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
