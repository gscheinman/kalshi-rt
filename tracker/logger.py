"""
DEPRECATED: this module's predictions.jsonl was a partial duplicate of the
canonical store in data/snapshots.jsonl. The dashboard and settlement workflow
now both read from snapshots.jsonl. log_prediction is a no-op kept for backward
compatibility with old scripts that still import it. load_predictions is kept
for any one-off debugging that wants the historical local cache.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

PREDICTIONS_DIR = Path.home() / ".cache" / "kalshi-rt"
PREDICTIONS_FILE = PREDICTIONS_DIR / "predictions.jsonl"


def log_prediction(movie, rt_slug, model_result, calibrated_probs, kalshi_prices):
    """DEPRECATED: no-op. Predictions are now recorded only in
    data/snapshots.jsonl by the snapshot pipeline. Kept so old imports don't
    break."""
    return None


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
