import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from flask import Flask, render_template, request, jsonify
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
    return render_template("dashboard.html")


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
        reviews = scrape_reviews(ems_id, slug=rt_slug or "", max_pages=5)

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

    result = predict_distribution(reviews, critic_db, movie_summary=movie_summary)
    calibrated = calibrate_thresholds(result["threshold_probs"], result["n_reviews"])

    market_prices = []
    if event_ticker:
        market_prices = kalshi.get_markets(event_ticker)

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
    """Merge paper trades from local cache and CI repo file, deduped."""
    import json
    from pathlib import Path

    local_trades = load_local_trades()

    # Also load CI trades from repo-local data/paper_trades.jsonl
    ci_file = Path(__file__).parent.parent / "data" / "paper_trades.jsonl"
    ci_trades = []
    if ci_file.exists():
        with open(ci_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ci_trades.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass

    # Dedup by (timestamp, event_ticker, threshold, direction)
    seen = set()
    merged = []
    for t in local_trades + ci_trades:
        key = (t.get("timestamp", ""), t.get("event_ticker", ""),
               t.get("threshold", 0), t.get("direction", ""))
        if key not in seen:
            seen.add(key)
            merged.append(t)

    return merged


@app.route("/api/paper-trades")
def api_paper_trades():
    """Return all paper trades with current RT scores for the dashboard."""
    trades = _load_all_trades()
    if not trades:
        return jsonify([])

    # Group trades by event for efficient RT lookups
    events_seen = {}
    for t in trades:
        ticker = t.get("event_ticker", "")
        if ticker and ticker not in events_seen:
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
    for t in trades:
        ticker = t.get("event_ticker", "")
        rt_data = events_seen.get(ticker, {})
        current_tomato = rt_data.get("tomatometer")

        avg_fill = t.get("avg_fill", 0)  # in cents
        cost_per = avg_fill / 100.0 if avg_fill else 0
        contracts = t.get("contracts", 0)
        direction = t.get("direction", "")
        threshold = t.get("threshold", 0)

        # Compute payout if correct
        if cost_per > 0 and cost_per < 1 and contracts > 0:
            gross_profit = contracts * (1.0 - cost_per)
            fee = gross_profit * 0.07
            payout_if_correct = round(gross_profit - fee, 2)
        else:
            payout_if_correct = 0

        # Determine if currently winning based on live tomatometer
        status = "pending"
        if current_tomato is not None:
            if direction == "BUY YES":
                status = "winning" if current_tomato > threshold else "losing"
            elif direction == "BUY NO":
                status = "winning" if current_tomato <= threshold else "losing"

        enriched.append({
            "timestamp": t.get("timestamp", ""),
            "movie": t.get("movie", ""),
            "event_ticker": ticker,
            "direction": direction,
            "threshold": threshold,
            "avg_fill": avg_fill,
            "contracts": contracts,
            "suggested_size": t.get("suggested_size", 0),
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
            "live": t.get("live", False),
            "spread_id": t.get("spread_id"),
            "ob_simulated": t.get("ob_simulated", False),
        })

    enriched.reverse()  # newest first
    return jsonify(enriched)


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


def _check_alpha(summary, markets):
    flags = []
    tomatometer = summary.get("tomatometer")
    if tomatometer is None:
        return flags
    for m in markets:
        t = m.get("threshold")
        if t is None:
            continue
        yes_ask = m.get("yes_ask") or m.get("yes_price")
        no_ask = (1.0 - m["yes_bid"]) if m.get("yes_bid") is not None else \
                 (1.0 - m["yes_price"]) if m.get("yes_price") is not None else None
        if yes_ask is None or no_ask is None:
            continue
        if tomatometer > t + 10 and yes_ask < 0.60:
            edge = round(tomatometer - t - (yes_ask * 100), 1)
            flags.append({"threshold": t, "direction": "BUY YES",
                          "price": round(yes_ask * 100), "rt": tomatometer, "edge": edge})
        elif tomatometer < t - 10 and (1.0 - yes_ask) > 0.40:
            edge = round(t - tomatometer - (no_ask * 100), 1)
            flags.append({"threshold": t, "direction": "BUY NO",
                          "price": round(no_ask * 100), "rt": tomatometer, "edge": edge})
    flags.sort(key=lambda x: abs(x.get("edge", 0)), reverse=True)
    return flags


if __name__ == "__main__":
    print("\n  Kalshi RT Trading Tool")
    print("  Open http://localhost:5001 in your browser\n")
    app.run(debug=True, port=5001)
