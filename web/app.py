import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request, jsonify
import config
from data.critics import CriticDatabase
from model.distribution import predict_distribution
from model.calibration import calibrate_thresholds
from market.kalshi_client import KalshiClient
from market.mapper import TickerMapper
from scraper.rt_page import get_movie_summary
from scraper.rt_reviews import scrape_reviews
from engine.alpha import find_alpha
from engine.paper_trader import load_trades as load_local_trades
from engine.portfolio import load_positions
from tracker.logger import log_prediction, load_predictions

app = Flask(__name__)
critic_db = CriticDatabase()
kalshi = KalshiClient()
mapper = TickerMapper()


@app.route("/")
def dashboard():
    # Disable browser caching so JS changes land immediately on next refresh.
    resp = app.make_response(render_template("dashboard.html"))
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/scan")
def api_scan():
    events = kalshi.get_rt_events()
    results = []
    for event in events:
        markets = kalshi.get_markets(event["event_ticker"])
        rt_slug = mapper.get_rt_slug(event)
        summary = get_movie_summary(rt_slug) if rt_slug else None

        tomatometer = summary.get("tomatometer") if summary else None
        review_count = summary.get("review_count", 0) if summary else 0
        release_date = summary.get("release_date") if summary else None

        close_time = markets[0].get("close_time") if markets else None

        kalshi_forecast = _compute_forecast(markets)
        alpha_flags = _check_alpha(summary, markets) if summary and tomatometer is not None else []

        best_edge = max((abs(a.get("edge", 0)) for a in alpha_flags), default=0) if alpha_flags else 0

        # Total event volume (sum across all thresholds) for tier classification
        total_volume = sum(float(m.get("volume") or 0) for m in markets)
        tier = _classify_tier(review_count, total_volume)

        results.append({
            "event_ticker": event["event_ticker"],
            "movie_name": event["movie_name"],
            "title": event["title"],
            "tomatometer": tomatometer,
            "review_count": review_count,
            "kalshi_forecast": kalshi_forecast,
            "market_count": len(markets),
            "alpha_flags": alpha_flags,
            "has_reviews": review_count > 0,
            "rt_slug": rt_slug,
            "release_date": release_date,
            "close_time": close_time,
            "best_edge": best_edge,
            "total_volume": round(total_volume, 0),
            "tier": tier,
        })

    results.sort(key=lambda x: x["close_time"] or "9999")
    return jsonify(results)


@app.route("/api/movie/<event_ticker>")
def api_movie(event_ticker):
    markets = kalshi.get_markets(event_ticker)
    if not markets:
        return jsonify({"error": "No markets found"}), 404

    events = kalshi.get_rt_events()
    event = next((e for e in events if e["event_ticker"] == event_ticker), None)
    movie_name = event["movie_name"] if event else event_ticker

    event_data = {"event_ticker": event_ticker, "movie_name": movie_name}
    rt_slug = mapper.get_rt_slug(event_data)
    summary = get_movie_summary(rt_slug) if rt_slug else None

    ems_id = summary.get("ems_id") if summary else None

    # Auto-scrape reviews if the movie has an EMS ID
    reviews = []
    if ems_id:
        reviews = scrape_reviews(
            ems_id, slug=rt_slug or "",
            expected_count=summary.get("review_count") if summary else None,
        )

    close_time = markets[0].get("close_time") if markets else None
    release_date = summary.get("release_date") if summary else None

    movie_summary = None
    if summary:
        movie_summary = {
            "title": summary.get("title"),
            "genres": summary.get("genres", []),
            "directors": summary.get("directors", []),
            "content_rating": summary.get("content_rating"),
        }

    return jsonify({
        "event_ticker": event_ticker,
        "movie_name": movie_name,
        "rt_slug": rt_slug,
        "ems_id": ems_id,
        "tomatometer": summary.get("tomatometer") if summary else None,
        "review_count": summary.get("review_count", 0) if summary else 0,
        "release_date": release_date,
        "close_time": close_time,
        "markets": markets,
        "reviews": reviews,
        "movie_summary": movie_summary,
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    data = request.json
    reviews = data.get("reviews", [])
    event_ticker = data.get("event_ticker")
    movie_name = data.get("movie_name", "Unknown")
    rt_slug = data.get("rt_slug", "")
    bankroll = data.get("bankroll", 1000)
    movie_summary = data.get("movie_summary")

    if not reviews:
        return jsonify({"error": "No reviews provided"}), 400

    # Get market data first so we know what thresholds to generate probs for
    market_prices = []
    if event_ticker:
        market_prices = kalshi.get_markets(event_ticker)

    # Pass actual Kalshi thresholds for granular brackets (57%, 58%, 62%, etc.)
    extra_thresholds = [m["threshold"] for m in market_prices if m.get("threshold") is not None]
    result = predict_distribution(reviews, critic_db, movie_summary=movie_summary,
                                  extra_thresholds=extra_thresholds)
    calibrated = calibrate_thresholds(result["threshold_probs"], result["n_reviews"])

    opportunities = []
    if market_prices:
        opportunities = find_alpha(result, market_prices, bankroll=bankroll, kalshi_client=kalshi)

    market_dict = {}
    for m in (market_prices or []):
        if m.get("threshold") is not None and m.get("yes_price") is not None:
            market_dict[m["threshold"]] = m["yes_price"]

    try:
        log_prediction(movie_name, rt_slug, result, calibrated, market_dict)
    except Exception:
        pass

    thresholds = []
    market_lookup = {m["threshold"]: m for m in market_prices if m.get("threshold") is not None}
    opp_lookup = {o["threshold"]: o for o in opportunities}

    for t in sorted(calibrated.keys()):
        market = market_lookup.get(t)
        opp = opp_lookup.get(t)

        yes_ask = market.get("yes_ask") or market.get("yes_price") if market else None

        entry = {
            "threshold": t,
            "model_prob": round(calibrated[t] * 100, 1),
            "market_prob": round(yes_ask * 100, 1) if yes_ask is not None else None,
            "edge": None,
            "direction": None,
            "size": None,
            "buy_price": None,
            "avg_fill": None,
        }

        if opp:
            entry["edge"] = opp.get("effective_edge", opp["edge"])
            entry["direction"] = opp["direction"]
            entry["size"] = opp["suggested_size"]
            entry["buy_price"] = opp.get("quoted_price")
            entry["avg_fill"] = opp.get("avg_fill")
            entry["contracts"] = opp.get("contracts")
            entry["profit_if_win"] = opp.get("profit_if_win")

        thresholds.append(entry)

    best = opportunities[0] if opportunities else None
    best_signal = None
    if best:
        best_signal = {
            "threshold": best["threshold"],
            "direction": best["direction"],
            "edge": best.get("effective_edge", best["edge"]),
            "model_prob": best["model_prob"],
            "market_prob": best["market_prob"],
            "buy_price": best.get("quoted_price"),
            "avg_fill": best.get("avg_fill"),
            "size": best["suggested_size"],
            "contracts": best.get("contracts"),
            "profit_if_win": best.get("profit_if_win"),
            "confidence": best["confidence"],
        }

    return jsonify({
        "model_mean": result["model_mean"],
        "model_ci": result["model_ci"],
        "naive_pct": result["naive_pct"],
        "confidence": result["confidence"],
        "n_reviews": result["n_reviews"],
        "n_known": result["n_known"],
        "known_pct": result["known_pct"],
        "thresholds": thresholds,
        "best_signal": best_signal,
        "critic_details": result.get("critic_details", []),
    })


@app.route("/api/history")
def api_history():
    records = load_predictions()
    records.reverse()
    return jsonify(records[:50])


def _load_all_trades():
    """Load paper trades from the CI repo file (data/paper_trades.jsonl).

    The dashboard intentionally ignores the local ~/.cache trades file because
    it contains legacy/test entries from early development (pre-CI). The CI
    file is the canonical record of trades the production system actually made.
    """
    import json
    from pathlib import Path

    ci_file = Path(__file__).parent.parent / "data" / "paper_trades.jsonl"
    trades = []
    if ci_file.exists():
        with open(ci_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return trades


def _load_settlements():
    """Build a {event_ticker: actual_score} map from resolved snapshots.
    Used to mark paper trades as settled with definitive win/loss."""
    settlements = {}
    snap_file = os.path.join(os.path.dirname(__file__), "..", "data", "snapshots.jsonl")
    if not os.path.exists(snap_file):
        return settlements
    try:
        import json as _json
        with open(snap_file) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    s = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                if s.get("resolved") and s.get("actual_score") is not None:
                    settlements[s["event_ticker"]] = s["actual_score"]
    except IOError:
        pass
    return settlements


@app.route("/api/paper-trades")
def api_paper_trades():
    """Return all paper trades with current RT scores + settlement status for the dashboard."""
    trades = _load_all_trades()
    if not trades:
        return jsonify({"trades": [], "summary": {"realized_pnl": 0, "wins": 0, "losses": 0, "pending": 0}})

    settlements = _load_settlements()

    # Group trades by event for efficient RT lookups (skip settled events)
    events_seen = {}
    for t in trades:
        ticker = t.get("event_ticker", "")
        if ticker and ticker not in events_seen and ticker not in settlements:
            slug = t.get("rt_slug", "")
            summary = None
            if slug:
                try:
                    summary = get_movie_summary(slug)
                except Exception:
                    pass
            events_seen[ticker] = {
                "tomatometer": summary.get("tomatometer") if summary else None,
                "review_count": summary.get("review_count", 0) if summary else 0,
            }

    # Enrich each trade with current RT data + computed fields
    enriched = []
    realized_pnl = 0.0
    wins = losses = pending = 0
    for t in trades:
        ticker = t.get("event_ticker", "")
        rt_data = events_seen.get(ticker, {})
        current_tomato = rt_data.get("tomatometer")
        settled_score = settlements.get(ticker)

        avg_fill = t.get("avg_fill", 0)  # in cents
        cost_per = avg_fill / 100.0 if avg_fill else 0
        contracts = t.get("contracts", 0)
        direction = t.get("direction", "")
        threshold = t.get("threshold", 0)
        size = t.get("suggested_size", 0)

        # Compute payout if correct
        if cost_per > 0 and cost_per < 1 and contracts > 0:
            gross_profit = contracts * (1.0 - cost_per)
            fee = gross_profit * 0.07
            payout_if_correct = round(gross_profit - fee, 2)
        else:
            payout_if_correct = 0

        # Determine status:
        #  - "won" / "lost": market is settled, definitive outcome
        #  - "winning" / "losing": still active, based on live tomatometer
        #  - "pending": active but no current score yet
        status = "pending"
        realized_pnl_trade = 0
        if settled_score is not None:
            if direction == "BUY YES":
                won = settled_score > threshold
            elif direction == "BUY NO":
                won = settled_score <= threshold
            else:
                won = False
            if won:
                status = "won"
                realized_pnl_trade = payout_if_correct
                wins += 1
            else:
                status = "lost"
                realized_pnl_trade = -round(size, 2)
                losses += 1
            realized_pnl += realized_pnl_trade
        elif current_tomato is not None:
            if direction == "BUY YES":
                status = "winning" if current_tomato > threshold else "losing"
            elif direction == "BUY NO":
                status = "winning" if current_tomato <= threshold else "losing"
            pending += 1
        else:
            pending += 1

        enriched.append({
            "timestamp": t.get("timestamp", ""),
            "movie": t.get("movie", ""),
            "event_ticker": ticker,
            "direction": direction,
            "threshold": threshold,
            "avg_fill": avg_fill,
            "contracts": contracts,
            "suggested_size": size,
            "edge": t.get("edge", 0),
            "win_prob": t.get("win_prob", 0),
            "model_mean": t.get("model_mean", 0),
            "naive_pct": t.get("naive_pct", 0),
            "confidence": t.get("confidence", ""),
            "n_reviews": t.get("n_reviews", 0),
            "current_tomatometer": current_tomato,
            "current_review_count": rt_data.get("review_count", 0),
            "payout_if_correct": payout_if_correct,
            "status": status,
            "settled": settled_score is not None,
            "settled_score": settled_score,
            "realized_pnl": round(realized_pnl_trade, 2),
            "live": t.get("live", False),
            "spread_id": t.get("spread_id"),
            "ob_simulated": t.get("ob_simulated", False),
        })

    enriched.reverse()  # newest first
    return jsonify({
        "trades": enriched,
        "summary": {
            "realized_pnl": round(realized_pnl, 2),
            "wins": wins,
            "losses": losses,
            "pending": pending,
            "total_trades": len(enriched),
        },
    })


def _guess_slug(movie_name):
    if not movie_name:
        return None
    slug = movie_name.lower().strip()
    slug = slug.replace(":", "").replace("'", "").replace("'", "")
    slug = slug.replace(" - ", "_").replace(" ", "_")
    slug = "".join(c for c in slug if c.isalnum() or c == "_")
    slug = slug.strip("_")
    return f"m/{slug}"


def _compute_forecast(markets):
    """Estimate implied RT score from market prices.
    Find the threshold where yes_price crosses 0.50 (linear interpolation)."""
    if not markets:
        return None
    priced = [(m["threshold"], m["yes_price"]) for m in markets
              if m.get("threshold") is not None and m.get("yes_price") is not None]
    if not priced:
        return None
    priced.sort(key=lambda x: x[0])

    # Find where probability crosses 50%
    for i in range(len(priced) - 1):
        t1, p1 = priced[i]
        t2, p2 = priced[i + 1]
        if p1 >= 0.50 >= p2:
            if p1 == p2:
                return round((t1 + t2) / 2)
            frac = (p1 - 0.50) / (p1 - p2)
            return round(t1 + frac * (t2 - t1))

    # If all above 50%, score is likely above the highest threshold
    if all(p >= 0.50 for _, p in priced):
        return priced[-1][0]
    # If all below 50%, score is likely below the lowest threshold
    if all(p < 0.50 for _, p in priced):
        return priced[0][0]

    return None


def _classify_tier(review_count, total_volume):
    """Tell the user why a market will or won't generate trades.

    Returns a dict with: label, color hint, min_edge required (as %),
    sanity cap (as %), and a one-line human reason.
    """
    rc = int(review_count or 0)
    vol = float(total_volume or 0)
    min_edge, enabled = config.min_edge_for(rc, vol)
    # Find which sanity cap applies
    sanity_cap = None
    for vol_floor, max_edge in config.SANITY_GRADED:
        if vol >= vol_floor:
            sanity_cap = max_edge
            break

    if not enabled:
        return {
            "label": "Waiting for reviews",
            "status": "blocked",
            "reason": f"Need 5+ reviews to enter trading window (have {rc})",
            "min_edge_pct": None,
            "sanity_cap_pct": None,
        }

    if rc < 40:
        zone = "Prime hunting (5-40 reviews)"
        status = "active"
    elif rc < 80:
        zone = "Secondary (40-80 reviews)"
        status = "active"
    else:
        zone = "Skeptical (80+ reviews, mostly efficient)"
        status = "skeptical"

    vol_note = ""
    if vol >= config.HIGH_VOLUME_THRESHOLD:
        vol_note = f" + ${int(vol/1000)}K vol bump"

    return {
        "label": zone,
        "status": status,
        "reason": f"Trades need >{int(min_edge*100)}% edge{vol_note}",
        "min_edge_pct": round(min_edge * 100, 1),
        "sanity_cap_pct": round(sanity_cap * 100, 1) if sanity_cap else None,
    }


def _check_alpha(summary, markets):
    """Sidebar tomatometer-only check. Flags markets where the current RT
    score is wildly different from what the market implies, AND there's
    actual liquidity on the side we'd buy.

    Returns only entries with positive expected edge and a reachable price
    (< 95c) -- anything at 100c has no possible profit even if it wins.
    """
    flags = []
    tomatometer = summary.get("tomatometer")
    if tomatometer is None:
        return flags
    for m in markets:
        t = m.get("threshold")
        if t is None:
            continue
        yes_ask = m.get("yes_ask")
        yes_bid = m.get("yes_bid")

        # BUY YES: need an ask price, not at the cap, with real edge
        if (yes_ask is not None and yes_ask < 0.95
                and tomatometer > t + 10 and yes_ask < 0.60):
            edge = round(tomatometer - t - (yes_ask * 100), 1)
            if edge > 0:
                flags.append({"threshold": t, "direction": "BUY YES",
                              "price": round(yes_ask * 100), "rt": tomatometer, "edge": edge})

        # BUY NO: cost = 1 - yes_bid. If yes_bid is None or 0 there's no real
        # NO offer (we'd be paying ~100c with no possible profit), so skip.
        if yes_bid is not None and yes_bid > 0.05:
            no_cost = 1.0 - yes_bid
            if (no_cost < 0.95 and tomatometer < t - 10 and no_cost < 0.60):
                edge = round(t - tomatometer - (no_cost * 100), 1)
                if edge > 0:
                    flags.append({"threshold": t, "direction": "BUY NO",
                                  "price": round(no_cost * 100), "rt": tomatometer, "edge": edge})

    flags.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)
    return flags


if __name__ == "__main__":
    print("\n  Kalshi RT Trading Tool")
    print("  Open http://localhost:5001 in your browser\n")
    app.run(debug=True, port=5001)
