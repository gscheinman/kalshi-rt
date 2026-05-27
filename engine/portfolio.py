"""
Multi-threshold portfolio optimizer for Kalshi RT markets.

Instead of picking the single best edge, this module optimizes a
portfolio of positions across ALL thresholds for a movie. Key concepts:

1. SCORE SCENARIOS: The final score determines all outcomes simultaneously.
   A score of 62% means Above 60% = YES, Above 65% = NO. We enumerate
   possible score outcomes and compute total P&L for each.

2. SPREAD POSITIONS: Buy YES on a lower threshold + NO on a higher threshold
   to bet on a score range. E.g., YES "Above 55%" + NO "Above 65%" profits
   if the score lands 55-65%.

3. DYNAMIC REBALANCING: When prices change, re-evaluate all positions.
   Add to winners if edge increased, close losers, open new hedges.

4. RISK-CONSTRAINED OPTIMIZATION: Maximize expected P&L subject to:
   - Max loss in any scenario < X% of bankroll
   - Max total exposure < portfolio limit
   - Min edge threshold per position
"""
import json
from datetime import datetime, timezone
from pathlib import Path
from scipy.stats import beta as beta_dist
from model.calibration import calibrate_thresholds
import config

POSITION_FILE = Path.home() / ".cache" / "kalshi-rt" / "positions.json"


def optimize_portfolio(model_result, markets, kalshi_client=None,
                       bankroll=None, existing_positions=None,
                       settlement_date=None):
    """
    Find the optimal set of positions across all thresholds for one movie.

    Returns a list of recommended trades (new positions + adjustments to existing).

    model_result: output from predict_distribution()
    markets: list of market dicts from kalshi.get_markets()
    existing_positions: list of current positions on this event
    """
    if bankroll is None:
        bankroll = config.DEFAULT_BANKROLL
    if existing_positions is None:
        existing_positions = []
    if not model_result or not markets:
        return {"trades": [], "portfolio_pnl": {}, "summary": {}}

    # Build the model's probability distribution
    alpha = model_result["alpha"]
    beta_param = model_result["beta"]
    dist = beta_dist(alpha, beta_param)
    confidence = model_result["confidence"]
    n_reviews = model_result["n_reviews"]

    # Calibrate
    raw_probs = model_result["threshold_probs"]
    calibrated = calibrate_thresholds(raw_probs, n_reviews)

    # Build market state: threshold -> prices + orderbook
    market_state = {}
    for m in markets:
        t = m.get("threshold")
        if t is None:
            continue
        ticker = m.get("ticker", "")
        yes_bid = m.get("yes_bid") or 0
        yes_ask = m.get("yes_ask") or m.get("yes_price") or 0
        yes_price = m.get("yes_price") or 0
        volume = float(m.get("volume", 0) or 0)

        # Get model probability (use calibrated if available, else raw from distribution)
        if t in calibrated:
            model_p = calibrated[t]
        else:
            model_p = float(1.0 - dist.cdf(t / 100.0))

        # Get orderbook depth
        ob = None
        if kalshi_client and ticker:
            ob = kalshi_client.get_orderbook(ticker)

        market_state[t] = {
            "ticker": ticker,
            "yes_bid": yes_bid,
            "yes_ask": yes_ask,
            "yes_price": yes_price,
            "model_prob": model_p,
            "edge_yes": model_p - yes_ask,  # positive = model says YES is cheap
            "edge_no": (1 - model_p) - (1 - yes_bid) if yes_bid else 0,
            "volume": volume,
            "orderbook": ob,
        }

    # Compute scenario P&L matrix
    # Score scenarios: every 1% from 0 to 100
    scenarios = _build_scenario_matrix(market_state, dist)

    # Confidence + time sizing multiplier
    confidence_mult = config.CONFIDENCE_KELLY_MULTIPLIER.get(confidence, 0.3)
    time_mult = _time_multiplier(settlement_date)
    sizing_mult = confidence_mult * time_mult

    # Find all profitable individual positions (orderbook-aware when possible)
    candidates = _find_candidate_positions(market_state, sizing_mult, bankroll, kalshi_client, n_reviews=n_reviews)

    # Find spread opportunities (paired positions)
    spreads = _find_spreads(market_state, dist, sizing_mult, bankroll, kalshi_client)

    # Combine and optimize the portfolio
    portfolio = _optimize_allocation(
        candidates, spreads, scenarios, market_state, dist,
        bankroll, sizing_mult, existing_positions,
    )

    return portfolio


def _build_scenario_matrix(market_state, dist):
    """
    For each possible final score (0-100 by 1%), compute the probability
    and the outcome (YES/NO) at each threshold.
    """
    scenarios = []
    thresholds = sorted(market_state.keys())

    for score in range(0, 101):
        prob = dist.pdf(score / 100.0) / 100.0  # probability density at this score
        outcomes = {}
        for t in thresholds:
            outcomes[t] = 1 if score > t else 0  # YES wins if score > threshold
        scenarios.append({
            "score": score,
            "probability": prob,
            "outcomes": outcomes,
        })

    return scenarios


def _find_candidate_positions(market_state, sizing_mult, bankroll, kalshi_client=None, n_reviews=0):
    """Find all individual threshold positions with positive expected value.

    When kalshi_client is provided, uses orderbook simulation for realistic
    fill prices and liquidity-constrained sizing. Otherwise falls back to
    quoted bid/ask prices.
    """
    candidates = []
    fee = config.KALSHI_FEE_RATE

    for t, state in sorted(market_state.items()):
        model_p = state["model_prob"]
        yes_ask = state["yes_ask"]
        yes_bid = state["yes_bid"]
        no_cost = 1.0 - yes_bid if yes_bid > 0 else 1.0 - state["yes_price"]
        ticker = state["ticker"]

        market_volume = state.get("volume") or 0
        tier_min_edge, tier_enabled = config.min_edge_for(n_reviews, market_volume)
        if not tier_enabled:
            continue

        # BUY YES: cost = yes_ask, win if outcome = 1
        edge_yes = model_p - yes_ask
        if (edge_yes > tier_min_edge
                and model_p >= config.MIN_WIN_PROB
                and yes_ask > 0
                and not config.sanity_blocks(edge_yes, market_volume)):
            # Try orderbook simulation for realistic sizing
            ob_result = None
            if kalshi_client and ticker:
                ob_result = _simulate_position(
                    kalshi_client, ticker, "BUY YES", model_p, bankroll, sizing_mult
                )

            if ob_result and ob_result["size"] > 0:
                avg_cost = ob_result["avg_cost"]
                size = ob_result["size"]
                contracts = ob_result["contracts"]
                effective_edge = ob_result["effective_edge"]
                ev = model_p * ((1 - avg_cost) * (1 - fee)) - (1 - model_p) * avg_cost
            else:
                kelly = _kelly(edge_yes, yes_ask, sizing_mult)
                size = min(kelly * bankroll, bankroll * config.MAX_POSITION_PCT)
                avg_cost = yes_ask
                contracts = size / yes_ask if yes_ask > 0 else 0
                effective_edge = edge_yes
                ev = model_p * ((1 - yes_ask) * (1 - fee)) - (1 - model_p) * yes_ask

            if size >= 0.50:
                candidates.append({
                    "threshold": t,
                    "direction": "BUY YES",
                    "cost": round(avg_cost, 4),
                    "edge": round(effective_edge * 100, 1),
                    "win_prob": round(model_p * 100, 1),
                    "kelly_size": round(size, 2),
                    "ev_per_dollar": round(ev / avg_cost, 4) if avg_cost > 0 else 0,
                    "contracts": round(contracts, 1),
                    "ticker": ticker,
                    "ob_simulated": ob_result is not None,
                })

        # BUY NO: cost = no_cost, win if outcome = 0
        edge_no = (1 - model_p) - no_cost
        if (edge_no > tier_min_edge
                and (1 - model_p) >= config.MIN_WIN_PROB
                and no_cost > 0 and no_cost < 1
                and not config.sanity_blocks(edge_no, market_volume)):
            ob_result = None
            if kalshi_client and ticker:
                ob_result = _simulate_position(
                    kalshi_client, ticker, "BUY NO", model_p, bankroll, sizing_mult
                )

            if ob_result and ob_result["size"] > 0:
                avg_cost = ob_result["avg_cost"]
                size = ob_result["size"]
                contracts = ob_result["contracts"]
                effective_edge = ob_result["effective_edge"]
                ev = (1 - model_p) * ((1 - avg_cost) * (1 - fee)) - model_p * avg_cost
            else:
                kelly = _kelly(edge_no, no_cost, sizing_mult)
                size = min(kelly * bankroll, bankroll * config.MAX_POSITION_PCT)
                avg_cost = no_cost
                contracts = size / no_cost if no_cost > 0 else 0
                effective_edge = edge_no
                ev = (1 - model_p) * ((1 - no_cost) * (1 - fee)) - model_p * no_cost

            if size >= 0.50:
                candidates.append({
                    "threshold": t,
                    "direction": "BUY NO",
                    "cost": round(avg_cost, 4),
                    "edge": round(effective_edge * 100, 1),
                    "win_prob": round((1 - model_p) * 100, 1),
                    "kelly_size": round(size, 2),
                    "ev_per_dollar": round(ev / avg_cost, 4) if avg_cost > 0 else 0,
                    "contracts": round(contracts, 1),
                    "ticker": ticker,
                    "ob_simulated": ob_result is not None,
                })

    return candidates


def _simulate_position(kalshi_client, ticker, direction, model_prob, bankroll, sizing_mult):
    """Walk the orderbook to find optimal position size with slippage.

    Only fills at price levels where the effective edge remains positive.
    Uses the same approach as alpha.py's _optimal_size_from_orderbook.
    """
    ob = kalshi_client.get_orderbook(ticker)
    if not ob:
        return None

    if direction == "BUY YES":
        book = sorted(ob["no_bids"], reverse=True)  # cheapest YES first
        win_prob = model_prob
    else:
        book = sorted(ob["yes_bids"], reverse=True)  # cheapest NO first
        win_prob = 1.0 - model_prob

    # Find price levels with positive EV
    profitable_levels = []
    for bid_price, size in book:
        cost_per = 1.0 - bid_price
        if cost_per <= 0 or cost_per >= 1:
            continue
        if cost_per >= win_prob:  # no edge at this price
            break
        profitable_levels.append((cost_per, size))

    if not profitable_levels:
        return None

    total_cost = 0.0
    total_contracts = 0.0

    for cost_per, size in profitable_levels:
        new_total_cost = total_cost + size * cost_per
        new_total_contracts = total_contracts + size
        new_avg = new_total_cost / new_total_contracts

        effective_edge = win_prob - new_avg
        if effective_edge <= 0:
            break

        # Kelly at this cumulative fill level
        kelly = config.KELLY_FRACTION * sizing_mult * effective_edge / (1.0 - new_avg)
        kelly_spend = min(kelly * bankroll, bankroll * config.MAX_POSITION_PCT)

        if new_total_cost <= kelly_spend:
            total_cost = new_total_cost
            total_contracts = new_total_contracts
        else:
            remaining = kelly_spend - total_cost
            if remaining > 0.01:
                take = min(remaining / cost_per, size)
                total_cost += take * cost_per
                total_contracts += take
            break

    if total_contracts < 1 or total_cost < 0.50:
        return None

    avg_cost = total_cost / total_contracts
    effective_edge = win_prob - avg_cost

    return {
        "size": round(total_cost, 2),
        "contracts": round(total_contracts, 1),
        "avg_cost": round(avg_cost, 4),
        "effective_edge": round(effective_edge, 4),
    }


def _find_spreads(market_state, dist, sizing_mult, bankroll, kalshi_client=None):
    """
    Find profitable spread positions: pairs of thresholds where betting
    the score lands in a range has positive expected value.

    A spread = BUY YES on lower threshold + BUY NO on higher threshold.
    Profit if score lands between the two thresholds.
    """
    spreads = []
    thresholds = sorted(market_state.keys())

    for i, t_low in enumerate(thresholds):
        for t_high in thresholds[i + 1:]:
            state_low = market_state[t_low]
            state_high = market_state[t_high]

            # Use orderbook simulation for realistic spread pricing
            yes_cost = state_low["yes_ask"]
            no_cost = 1.0 - state_high["yes_bid"] if state_high["yes_bid"] > 0 else None

            if yes_cost <= 0 or no_cost is None or no_cost <= 0 or no_cost >= 1:
                continue

            total_cost = yes_cost + no_cost
            if total_cost >= 1.0:
                continue  # can't profit -- cost exceeds max payout

            # Probability score lands in [t_low, t_high]
            # (wins YES on low, wins NO on high)
            p_in_range = float(dist.cdf(t_high / 100.0) - dist.cdf(t_low / 100.0))

            # Probability score > t_high (wins YES on low, loses NO on high)
            p_above = float(1.0 - dist.cdf(t_high / 100.0))

            # Probability score < t_low (loses YES on low, wins NO on high)
            p_below = float(dist.cdf(t_low / 100.0))

            # P&L per spread (1 YES + 1 NO contract):
            # Kalshi: winning contract nets (1 - cost) * (1 - fee), losing = -cost
            fee = config.KALSHI_FEE_RATE
            pnl_in_range = (2 - total_cost) * (1 - fee)
            pnl_above = (1 - yes_cost) * (1 - fee) - no_cost
            pnl_below = -yes_cost + (1 - no_cost) * (1 - fee)

            expected_pnl = (p_in_range * pnl_in_range +
                            p_above * pnl_above +
                            p_below * pnl_below)

            max_loss = min(pnl_above, pnl_below, pnl_in_range)

            if expected_pnl <= 0:
                continue

            # Only keep spreads where EV is meaningfully positive
            ev_pct = expected_pnl / total_cost
            if ev_pct < 0.05:  # need at least 5% EV
                continue

            # Size the spread
            kelly = max(0, config.KELLY_FRACTION * sizing_mult * ev_pct)
            size = min(kelly * bankroll, bankroll * config.MAX_POSITION_PCT)
            if size < 1.00:
                continue

            # Number of spreads we can buy
            n_spreads = size / total_cost

            spreads.append({
                "type": "spread",
                "t_low": t_low,
                "t_high": t_high,
                "yes_cost": round(yes_cost, 3),
                "no_cost": round(no_cost, 3),
                "total_cost": round(total_cost, 3),
                "p_in_range": round(p_in_range * 100, 1),
                "p_above": round(p_above * 100, 1),
                "p_below": round(p_below * 100, 1),
                "pnl_in_range": round(pnl_in_range, 4),
                "pnl_above": round(pnl_above, 4),
                "pnl_below": round(pnl_below, 4),
                "expected_pnl": round(expected_pnl, 4),
                "ev_pct": round(ev_pct * 100, 1),
                "max_loss": round(max_loss, 4),
                "suggested_size": round(size, 2),
                "n_spreads": round(n_spreads, 1),
                "ticker_low": market_state[t_low]["ticker"],
                "ticker_high": market_state[t_high]["ticker"],
            })

    spreads.sort(key=lambda x: x["expected_pnl"] * x["n_spreads"], reverse=True)
    return spreads


def _optimize_allocation(candidates, spreads, scenarios, market_state, dist,
                         bankroll, sizing_mult, existing_positions):
    """
    Allocate capital across candidates and spreads to maximize expected P&L
    subject to risk constraints.

    Simple greedy approach: rank all opportunities by EV/dollar, fill in order
    until budget is spent or risk limits hit.
    """
    max_event_budget = bankroll * config.MAX_EVENT_EXPOSURE_PCT

    # Account for existing positions
    existing_spend = sum(p.get("cost", 0) for p in existing_positions)
    remaining_budget = max(0, max_event_budget - existing_spend)

    # Combine and rank all opportunities
    all_opps = []

    for c in candidates:
        all_opps.append({
            "type": "single",
            "ev_per_dollar": c["ev_per_dollar"],
            "cost": c["cost"],
            "kelly_size": c["kelly_size"],
            "details": c,
        })

    for s in spreads:
        ev_per_dollar = s["expected_pnl"] / s["total_cost"] if s["total_cost"] > 0 else 0
        all_opps.append({
            "type": "spread",
            "ev_per_dollar": ev_per_dollar,
            "cost": s["total_cost"],
            "kelly_size": s["suggested_size"],
            "details": s,
        })

    # Sort by EV per dollar (best bang for the buck)
    all_opps.sort(key=lambda x: x["ev_per_dollar"], reverse=True)

    # Greedy allocation
    trades = []
    total_spend = 0
    thresholds_used = set()
    yes_thresholds = set()   # thresholds where we hold BUY YES (win if score > T)
    no_thresholds = set()    # thresholds where we hold BUY NO  (win if score <= T)

    for opp in all_opps:
        size = min(opp["kelly_size"], remaining_budget - total_spend)
        if size < 0.50:
            continue

        d = opp["details"]

        if opp["type"] == "single":
            t = d["threshold"]
            direction = d["direction"]

            # Don't double up on same threshold + direction.
            key = (t, direction)
            if key in thresholds_used:
                continue

            # Mutual-exclusion guard. BUY YES Above X wins on score > X;
            # BUY NO Above Y wins on score <= Y. If Y < X the winning regions
            # don't overlap, so taking both guarantees we lose at least one.
            # The optimizer would otherwise rank each on its own EV and never
            # see that they're betting on disjoint outcomes.
            if direction == "BUY YES":
                contradicting = any(no_t < t for no_t in no_thresholds)
                if contradicting:
                    continue
            else:  # BUY NO
                contradicting = any(yes_t > t for yes_t in yes_thresholds)
                if contradicting:
                    continue

            thresholds_used.add(key)
            if direction == "BUY YES":
                yes_thresholds.add(t)
            else:
                no_thresholds.add(t)

            # Use orderbook-simulated contracts if available, else compute
            contracts = d.get("contracts", round(size / d["cost"], 1) if d["cost"] > 0 else 0)
            # Scale contracts if we're allocating less than kelly_size
            if size < d["kelly_size"] and d["kelly_size"] > 0:
                contracts = round(contracts * size / d["kelly_size"], 1)

            trades.append({
                "type": "single",
                "threshold": d["threshold"],
                "direction": d["direction"],
                "ticker": d["ticker"],
                "cost_per": d["cost"],
                "edge": d["edge"],
                "win_prob": d["win_prob"],
                "size": round(size, 2),
                "contracts": contracts,
                "ev_per_dollar": round(opp["ev_per_dollar"] * 100, 1),
                "ob_simulated": d.get("ob_simulated", False),
            })
            total_spend += size

        elif opp["type"] == "spread":
            trades.append({
                "type": "spread",
                "t_low": d["t_low"],
                "t_high": d["t_high"],
                "ticker_low": d["ticker_low"],
                "ticker_high": d["ticker_high"],
                "yes_cost": d["yes_cost"],
                "no_cost": d["no_cost"],
                "total_cost_per": d["total_cost"],
                "p_in_range": d["p_in_range"],
                "ev_pct": d["ev_pct"],
                "pnl_in_range": d["pnl_in_range"],
                "pnl_above": d["pnl_above"],
                "pnl_below": d["pnl_below"],
                "size": round(size, 2),
                "n_spreads": round(size / d["total_cost"], 1) if d["total_cost"] > 0 else 0,
            })
            total_spend += size

    # Compute portfolio-level scenario P&L
    portfolio_pnl = _compute_portfolio_pnl(trades, market_state, dist)

    return {
        "trades": trades,
        "total_spend": round(total_spend, 2),
        "budget_used_pct": round(total_spend / max_event_budget * 100, 1) if max_event_budget > 0 else 0,
        "n_positions": len(trades),
        "n_single": sum(1 for t in trades if t["type"] == "single"),
        "n_spreads": sum(1 for t in trades if t["type"] == "spread"),
        "portfolio_pnl": portfolio_pnl,
    }


def _compute_portfolio_pnl(trades, market_state, dist):
    """
    Compute expected P&L and risk metrics across score scenarios.
    """
    if not trades:
        return {"expected_pnl": 0, "max_loss": 0, "best_case": 0, "worst_score": None}

    fee = config.KALSHI_FEE_RATE
    best_pnl = -999999
    worst_pnl = 999999
    worst_score = None
    expected_pnl = 0

    for score in range(0, 101):
        prob = dist.pdf(score / 100.0) / 100.0
        if prob < 0.0001:
            continue

        total_pnl = 0
        for trade in trades:
            if trade["type"] == "single":
                t = trade["threshold"]
                outcome = 1 if score > t else 0
                cost = trade["cost_per"]
                contracts = trade.get("contracts", 0)

                if trade["direction"] == "BUY YES":
                    if outcome == 1:
                        total_pnl += contracts * (1 - cost) * (1 - fee)
                    else:
                        total_pnl -= contracts * cost
                else:  # BUY NO
                    if outcome == 0:
                        total_pnl += contracts * (1 - cost) * (1 - fee)
                    else:
                        total_pnl -= contracts * cost

            elif trade["type"] == "spread":
                n = trade.get("n_spreads", 0)
                t_low = trade["t_low"]
                t_high = trade["t_high"]
                yes_cost = trade["yes_cost"]
                no_cost = trade["no_cost"]

                yes_wins = score > t_low
                no_wins = score <= t_high

                pnl = 0
                if yes_wins:
                    pnl += (1 - yes_cost) * (1 - fee)
                else:
                    pnl -= yes_cost
                if no_wins:
                    pnl += (1 - no_cost) * (1 - fee)
                else:
                    pnl -= no_cost

                total_pnl += n * pnl

        expected_pnl += prob * total_pnl

        if total_pnl < worst_pnl:
            worst_pnl = total_pnl
            worst_score = score
        if total_pnl > best_pnl:
            best_pnl = total_pnl

    return {
        "expected_pnl": round(expected_pnl, 2),
        "max_loss": round(worst_pnl, 2),
        "best_case": round(best_pnl, 2),
        "worst_score": worst_score,
    }


def evaluate_rebalance(model_result, markets, kalshi_client, existing_positions,
                       bankroll=None, settlement_date=None):
    """
    Given existing positions and current market prices, decide whether to:
    1. Hold (edge still exists, no action)
    2. Add (edge increased, size up)
    3. Reduce (edge decreased below threshold)
    4. Close (edge reversed or disappeared)
    5. Hedge (add offsetting position to limit risk)
    6. Open new (new threshold now has edge that didn't before)

    Returns list of recommended actions.
    """
    if bankroll is None:
        bankroll = config.DEFAULT_BANKROLL

    # Get fresh portfolio optimization
    new_portfolio = optimize_portfolio(
        model_result, markets, kalshi_client,
        bankroll=bankroll, existing_positions=existing_positions,
        settlement_date=settlement_date,
    )

    actions = []

    # Check each existing position
    raw_probs = model_result["threshold_probs"]
    n_reviews = model_result["n_reviews"]
    calibrated = calibrate_thresholds(raw_probs, n_reviews)

    alpha = model_result["alpha"]
    beta_param = model_result["beta"]
    dist = beta_dist(alpha, beta_param)

    for pos in existing_positions:
        t = pos["threshold"]
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        contracts = pos["contracts"]

        # Current market price
        current_market = None
        for m in markets:
            if m.get("threshold") == t:
                current_market = m
                break

        if not current_market:
            continue

        # Model probability
        if t in calibrated:
            model_p = calibrated[t]
        else:
            model_p = float(1.0 - dist.cdf(t / 100.0))

        if direction == "BUY YES":
            current_price = current_market.get("yes_bid", 0) or 0
            current_edge = model_p - current_price
            unrealized_pnl = (current_price - entry_price) * contracts
        else:
            current_price = 1.0 - (current_market.get("yes_ask", 1) or 1)
            current_edge = (1 - model_p) - (1 - current_price)
            unrealized_pnl = (current_price - entry_price) * contracts

        if current_edge < 0:
            # Edge reversed -- recommend closing
            actions.append({
                "action": "CLOSE",
                "threshold": t,
                "direction": direction,
                "reason": f"Edge reversed to {current_edge*100:+.1f}%",
                "unrealized_pnl": round(unrealized_pnl, 2),
                "contracts": contracts,
                "ticker": pos.get("ticker", ""),
            })
        elif current_edge < config.MIN_EDGE * 0.5:
            # Edge shrunk significantly -- recommend reducing
            actions.append({
                "action": "REDUCE",
                "threshold": t,
                "direction": direction,
                "reason": f"Edge shrunk to {current_edge*100:.1f}%",
                "unrealized_pnl": round(unrealized_pnl, 2),
                "reduce_pct": 50,
                "ticker": pos.get("ticker", ""),
            })
        elif current_edge > config.MIN_EDGE * 2:
            # Edge increased -- consider adding
            actions.append({
                "action": "ADD",
                "threshold": t,
                "direction": direction,
                "reason": f"Edge increased to {current_edge*100:.1f}%",
                "unrealized_pnl": round(unrealized_pnl, 2),
                "ticker": pos.get("ticker", ""),
            })
        else:
            actions.append({
                "action": "HOLD",
                "threshold": t,
                "direction": direction,
                "reason": f"Edge stable at {current_edge*100:.1f}%",
                "unrealized_pnl": round(unrealized_pnl, 2),
            })

    # Check for new opportunities not in existing positions
    existing_keys = set((p["threshold"], p["direction"]) for p in existing_positions)
    for trade in new_portfolio["trades"]:
        if trade["type"] == "single":
            key = (trade["threshold"], trade["direction"])
            if key not in existing_keys:
                actions.append({
                    "action": "OPEN",
                    "threshold": trade["threshold"],
                    "direction": trade["direction"],
                    "reason": f"New edge: {trade['edge']}%",
                    "size": trade["size"],
                    "ticker": trade["ticker"],
                })
        elif trade["type"] == "spread":
            actions.append({
                "action": "OPEN_SPREAD",
                "t_low": trade["t_low"],
                "t_high": trade["t_high"],
                "reason": f"Spread EV: {trade['ev_pct']}%",
                "size": trade["size"],
            })

    return actions


def _time_multiplier(settlement_date):
    """Scale sizing based on proximity to settlement."""
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


def _kelly(edge, cost, sizing_mult):
    """Quarter Kelly with sizing multiplier."""
    if cost <= 0 or cost >= 1 or edge <= 0:
        return 0
    full_kelly = edge / (1 - cost)
    return max(0, config.KELLY_FRACTION * sizing_mult * full_kelly)


# --- Position tracking ---

def load_positions():
    """Load current positions from disk."""
    if not POSITION_FILE.exists():
        return {}
    with open(POSITION_FILE) as f:
        return json.load(f)


def save_positions(positions):
    """Save current positions to disk."""
    POSITION_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(POSITION_FILE, "w") as f:
        json.dump(positions, f, indent=2)


def record_trade(event_ticker, trade):
    """Record a new trade in the position tracker."""
    positions = load_positions()
    if event_ticker not in positions:
        positions[event_ticker] = []
    positions[event_ticker].append({
        **trade,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    save_positions(positions)
