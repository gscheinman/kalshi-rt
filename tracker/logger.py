import json
from datetime import datetime, timezone
from pathlib import Path

PREDICTIONS_DIR = Path.home() / ".cache" / "kalshi-rt"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions.jsonl"


def log_prediction(movie, rt_slug, model_result, calibrated_probs, kalshi_prices):
    """Append a prediction record to the JSONL log."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)

    signals = []
    for t, model_p in calibrated_probs.items():
        market_p = kalshi_prices.get(t)
        if market_p is None:
            continue
        edge = model_p - market_p
        if abs(edge) >= 0.05:
            signals.append({
                "threshold": t,
                "direction": "BUY YES" if edge > 0 else "BUY NO",
                "edge": round(edge * 100, 1),
                "model_prob": round(model_p * 100, 1),
                "market_prob": round(market_p * 100, 1),
            })

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "movie": movie,
        "rt_slug": rt_slug,
        "n_reviews": model_result.get("n_reviews", 0),
        "n_known": model_result.get("n_known", 0),
        "model_mean": model_result.get("model_mean"),
        "model_ci": model_result.get("model_ci"),
        "confidence": model_result.get("confidence"),
        "threshold_probs": {str(k): round(v, 4) for k, v in calibrated_probs.items()},
        "kalshi_prices": {str(k): round(v, 4) for k, v in kalshi_prices.items()},
        "signals": signals,
        "resolved": False,
        "actual_score": None,
        "pnl": None,
    }

    with open(PREDICTIONS_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

    return record


def load_predictions():
    """Load all prediction records."""
    if not PREDICTIONS_FILE.exists():
        return []
    records = []
    with open(PREDICTIONS_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_predictions(records):
    """Overwrite the predictions file with updated records."""
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    with open(PREDICTIONS_FILE, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
