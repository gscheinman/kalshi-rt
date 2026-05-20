"""
Scoring functions for evaluating model predictions against actual outcomes.
"""
import math


def brier_score(predictions):
    """Brier score: mean squared error of probability predictions.
    Lower is better. 0 = perfect, 0.25 = coin flip.

    predictions: list of (predicted_prob, actual_outcome) where
        predicted_prob is 0-1 and actual_outcome is 0 or 1.
    """
    if not predictions:
        return None
    total = sum((p - a) ** 2 for p, a in predictions)
    return total / len(predictions)


def calibration_buckets(predictions, n_buckets=10):
    """Group predictions into buckets and compare predicted vs actual rates.

    Returns list of dicts with bucket_center, predicted_avg, actual_rate, count.
    A well-calibrated model has predicted_avg close to actual_rate in each bucket.
    """
    if not predictions:
        return []

    bucket_size = 1.0 / n_buckets
    buckets = {}

    for pred, actual in predictions:
        bucket = min(int(pred / bucket_size), n_buckets - 1)
        if bucket not in buckets:
            buckets[bucket] = {"pred_sum": 0, "actual_sum": 0, "count": 0}
        buckets[bucket]["pred_sum"] += pred
        buckets[bucket]["actual_sum"] += actual
        buckets[bucket]["count"] += 1

    result = []
    for b in sorted(buckets.keys()):
        d = buckets[b]
        result.append({
            "bucket_center": round((b + 0.5) * bucket_size, 2),
            "predicted_avg": round(d["pred_sum"] / d["count"], 3),
            "actual_rate": round(d["actual_sum"] / d["count"], 3),
            "count": d["count"],
            "gap": round(abs(d["pred_sum"] / d["count"] - d["actual_sum"] / d["count"]), 3),
        })
    return result


def calibration_error(predictions, n_buckets=10):
    """Expected calibration error: weighted average of |predicted - actual| per bucket."""
    buckets = calibration_buckets(predictions, n_buckets)
    if not buckets:
        return None
    total_count = sum(b["count"] for b in buckets)
    return sum(b["gap"] * b["count"] / total_count for b in buckets)


def mean_absolute_error(predictions_continuous):
    """MAE for continuous predictions (predicted_score vs actual_score).
    predictions_continuous: list of (predicted, actual) as percentages.
    """
    if not predictions_continuous:
        return None
    return sum(abs(p - a) for p, a in predictions_continuous) / len(predictions_continuous)


def simulated_pnl(trades, fee_rate=0.07):
    """Simulate P&L for a set of trades.

    trades: list of dicts with keys:
        - direction: 'BUY YES' or 'BUY NO'
        - cost: amount spent
        - contracts: number of contracts
        - threshold: the threshold bet on
        - actual_score: the final tomatometer

    Returns dict with total P&L, win rate, avg profit per trade, etc.
    """
    if not trades:
        return None

    total_pnl = 0
    wins = 0
    losses = 0
    total_risked = 0

    details = []
    for t in trades:
        won = _trade_won(t["direction"], t["threshold"], t["actual_score"])
        cost = t["cost"]
        contracts = t["contracts"]

        if won:
            gross_profit = contracts - cost
            net_profit = gross_profit * (1 - fee_rate)
            wins += 1
        else:
            net_profit = -cost
            losses += 1

        total_pnl += net_profit
        total_risked += cost
        details.append({**t, "won": won, "pnl": round(net_profit, 2)})

    total = wins + losses
    return {
        "total_pnl": round(total_pnl, 2),
        "total_risked": round(total_risked, 2),
        "roi": round(total_pnl / total_risked * 100, 1) if total_risked > 0 else 0,
        "win_rate": round(wins / total * 100, 1) if total > 0 else 0,
        "wins": wins,
        "losses": losses,
        "total_trades": total,
        "avg_pnl_per_trade": round(total_pnl / total, 2) if total > 0 else 0,
        "details": details,
    }


def sharpe_ratio(trade_pnls):
    """Sharpe-like ratio: mean P&L / std P&L across trades.
    Higher is better. >1 is good, >2 is excellent."""
    if len(trade_pnls) < 3:
        return None
    mean = sum(trade_pnls) / len(trade_pnls)
    variance = sum((p - mean) ** 2 for p in trade_pnls) / (len(trade_pnls) - 1)
    std = math.sqrt(variance) if variance > 0 else 0
    return round(mean / std, 2) if std > 0 else None


def _trade_won(direction, threshold, actual_score):
    if direction == "BUY YES":
        return actual_score > threshold
    else:
        return actual_score <= threshold
