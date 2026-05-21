from datetime import datetime, timezone, timedelta
from model.calibration import calibrate_thresholds
import config


def find_alpha(model_result, market_prices, bankroll=None, kalshi_client=None,
               settlement_date=None, existing_event_exposure=0.0):
    """Compare model probabilities vs Kalshi market prices at each threshold.

    If kalshi_client is provided, uses orderbook data for slippage-aware sizing.
    settlement_date: datetime of Monday 10 AM ET resolution (for time-based sizing).
    existing_event_exposure: dollars already deployed on this event (for per-event caps).
    """
    if bankroll is None:
        bankroll = config.DEFAULT_BANKROLL
    if not model_result or not market_prices:
        return []

    raw_probs = model_result["threshold_probs"]
    n_reviews = model_result["n_reviews"]
    confidence = model_result["confidence"]

    calibrated = calibrate_thresholds(raw_probs, n_reviews)

    confidence_mult = config.CONFIDENCE_KELLY_MULTIPLIER.get(confidence, 0.3)
    time_mult = _time_to_settlement_multiplier(settlement_date)
    sizing_mult = confidence_mult * time_mult

    remaining_event_budget = max(0, bankroll * config.MAX_EVENT_EXPOSURE_PCT - existing_event_exposure)

    opportunities = []
    for m in market_prices:
        threshold = m.get("threshold")
        yes_price = m.get("yes_price")

        if threshold is None or yes_price is None:
            continue
        if threshold not in calibrated:
            continue

        model_prob = calibrated[threshold]
        yes_ask = m.get("yes_ask") or yes_price
        no_ask = (1.0 - m["yes_bid"]) if m.get("yes_bid") is not None else (1.0 - yes_price)

        edge = model_prob - yes_ask

        market_volume = m.get("volume") or 0
        tier_min_edge, tier_enabled = config.min_edge_for(n_reviews, market_volume)
        if not tier_enabled:
            continue

        if abs(edge) < tier_min_edge:
            continue

        if edge > 0:
            direction = "BUY YES"
            quoted_cost = yes_ask
            win_prob = model_prob
        else:
            direction = "BUY NO"
            quoted_cost = no_ask
            win_prob = 1.0 - model_prob
            edge = win_prob - no_ask

        if edge < tier_min_edge:
            continue

        if win_prob < config.MIN_WIN_PROB:
            continue

        # Graded sanity guard: extreme edge is almost certainly model error.
        # Tighter caps on thicker markets where consensus is more informed.
        if config.sanity_blocks(edge, market_volume):
            continue

        ob_result = None
        if kalshi_client and m.get("ticker"):
            ob_result = _optimal_size_from_orderbook(
                kalshi_client, m["ticker"], direction, model_prob, bankroll,
                sizing_mult=sizing_mult,
            )

        if ob_result and ob_result["size"] > 0:
            suggested_size = ob_result["size"]
            avg_fill = ob_result["avg_fill"]
            effective_edge = ob_result["effective_edge"]
            contracts = ob_result["contracts"]
            profit_if_win = ob_result["profit_if_win"]
        else:
            kelly = kelly_fraction(edge, quoted_cost) * sizing_mult
            suggested_size = min(kelly * bankroll, bankroll * config.MAX_POSITION_PCT)
            avg_fill = round(quoted_cost * 100, 1)
            effective_edge = round(edge * 100, 1)
            contracts = round(suggested_size / quoted_cost, 1) if quoted_cost > 0 else 0
            gross_profit = contracts - suggested_size
            profit_if_win = round(gross_profit * (1 - config.KALSHI_FEE_RATE), 2)

        suggested_size = min(suggested_size, remaining_event_budget)
        if suggested_size < 0.50:
            continue

        if contracts > 0 and suggested_size > 0:
            contracts = round(suggested_size / (avg_fill / 100), 1) if avg_fill > 0 else 0
            gross_profit = contracts - suggested_size
            profit_if_win = round(gross_profit * (1 - config.KALSHI_FEE_RATE), 2)

        expected_profit = win_prob * profit_if_win - (1.0 - win_prob) * suggested_size

        opportunities.append({
            "threshold": threshold,
            "model_prob": round(model_prob * 100, 1),
            "market_prob": round(yes_ask * 100, 1),
            "edge": round(edge * 100, 1),
            "effective_edge": effective_edge,
            "direction": direction,
            "win_prob": round(win_prob * 100, 1),
            "suggested_size": round(suggested_size, 2),
            "avg_fill": avg_fill,
            "contracts": contracts,
            "profit_if_win": profit_if_win,
            "expected_profit": round(expected_profit, 2),
            "quoted_price": round(quoted_cost * 100, 1),
            "ticker": m.get("ticker", ""),
            "volume": m.get("volume", 0),
            "confidence": confidence,
            "known_pct": model_result.get("known_pct", 0),
            "sizing_mult": round(sizing_mult, 3),
        })

    opportunities.sort(key=lambda x: x["expected_profit"], reverse=True)
    return opportunities


def _time_to_settlement_multiplier(settlement_date):
    """Scale sizing based on proximity to Monday settlement."""
    if settlement_date is None:
        return 1.0

    now = datetime.now(timezone.utc)
    days_left = max(0, (settlement_date - now).total_seconds() / 86400)

    s = config.SETTLEMENT_SIZING
    if days_left <= s["full_confidence_days"]:
        return s["max_multiplier"]
    if days_left >= s["base_confidence_days"]:
        return s["base_multiplier"]

    t = (s["base_confidence_days"] - days_left) / (s["base_confidence_days"] - s["full_confidence_days"])
    return s["base_multiplier"] + t * (s["max_multiplier"] - s["base_multiplier"])


def _optimal_size_from_orderbook(kalshi_client, ticker, direction, model_prob, bankroll,
                                 sizing_mult=1.0):
    """Walk the orderbook to find the optimal bet size accounting for slippage.

    Only buys contracts where the price is below our model's win probability
    (positive expected value). Caps total spend with Kelly criterion on the
    effective edge after slippage."""
    ob = kalshi_client.get_orderbook(ticker)
    if not ob:
        return None

    if direction == "BUY YES":
        book = sorted(ob["no_bids"], reverse=True)
        win_prob = model_prob
    else:
        book = sorted(ob["yes_bids"], reverse=True)
        win_prob = 1.0 - model_prob

    profitable_levels = []
    for bid_price, size in book:
        cost_per = 1.0 - bid_price
        if cost_per <= 0 or cost_per >= 1:
            continue
        if cost_per >= win_prob:
            break
        profitable_levels.append((cost_per, size))

    if not profitable_levels:
        return None

    total_cost = 0.0
    total_contracts = 0.0
    fills = []

    for cost_per, size in profitable_levels:
        level_contracts = size
        level_cost = size * cost_per

        new_total_cost = total_cost + level_cost
        new_total_contracts = total_contracts + level_contracts
        new_avg_fill = new_total_cost / new_total_contracts

        effective_edge = win_prob - new_avg_fill
        if effective_edge <= 0:
            break
        kelly = config.KELLY_FRACTION * sizing_mult * effective_edge / (1.0 - new_avg_fill)
        kelly_spend = kelly * bankroll

        if new_total_cost <= kelly_spend:
            total_cost = new_total_cost
            total_contracts = new_total_contracts
            fills.append({"price": round(cost_per * 100, 1), "qty": round(size, 1)})
        else:
            remaining = kelly_spend - total_cost
            if remaining > 0.01:
                take = remaining / cost_per
                take = min(take, size)
                total_cost += take * cost_per
                total_contracts += take
                fills.append({"price": round(cost_per * 100, 1), "qty": round(take, 1)})
            break

    if total_contracts == 0 or total_cost == 0:
        return None

    avg_fill_pct = total_cost / total_contracts * 100
    effective_edge = round(win_prob * 100 - avg_fill_pct, 1)
    gross_profit = total_contracts - total_cost
    net_profit = gross_profit * (1 - config.KALSHI_FEE_RATE)

    return {
        "size": round(total_cost, 2),
        "avg_fill": round(avg_fill_pct, 1),
        "effective_edge": effective_edge,
        "contracts": round(total_contracts, 1),
        "profit_if_win": round(net_profit, 2),
        "fills": fills,
    }


def kelly_fraction(edge, cost):
    if cost <= 0 or cost >= 1:
        return 0
    full_kelly = edge / (1 - cost) if (1 - cost) > 0 else 0
    return max(0, config.KELLY_FRACTION * full_kelly)
