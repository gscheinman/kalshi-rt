from datetime import datetime, timezone
from tracker.logger import load_predictions, save_predictions
from scraper.rt_page import get_movie_summary
import config


def resolve_predictions():
    """Check unresolved predictions and record settlement scores.

    Kalshi RT markets resolve on the Monday after wide release at 10 AM ET.
    The score at that moment is what matters, not the eventual "final" score
    (critics can keep reviewing after Monday). We record the current score
    as the settlement score, and the review count + timestamp so we can
    distinguish Monday-of-settlement snapshots from later lookups.
    """
    records = load_predictions()
    if not records:
        print("No predictions logged yet.")
        return

    unresolved = [r for r in records if not r["resolved"]]
    if not unresolved:
        print("All predictions already resolved.")
        return

    print(f"Found {len(unresolved)} unresolved predictions.\n")

    resolved_count = 0
    for r in records:
        if r["resolved"]:
            continue

        slug = r.get("rt_slug", "")
        movie = r.get("movie", slug)
        print(f"Checking {movie}...")

        summary = get_movie_summary(slug)
        if not summary or summary.get("review_count", 0) < 50:
            print(f"  Not enough reviews yet ({summary.get('review_count', 0) if summary else 0}). Skipping.")
            continue

        actual = summary["tomatometer"]
        if actual is None:
            print(f"  No tomatometer available. Skipping.")
            continue

        r["actual_score"] = actual
        r["settlement_review_count"] = summary.get("review_count", 0)
        r["settlement_timestamp"] = datetime.now(timezone.utc).isoformat()
        r["resolved"] = True

        pnl = _compute_pnl(r, actual)
        r["pnl"] = pnl

        print(f"  Resolved: {movie} = {actual}% ({summary.get('review_count', 0)} reviews at settlement)")
        if r.get("signals"):
            for s in r["signals"]:
                won = _signal_won(s, actual)
                status = "WIN" if won else "LOSS"
                print(f"    {s['direction']} Above {s['threshold']}%: {status} (edge was {s['edge']:+.1f}%)")
        resolved_count += 1

    if resolved_count > 0:
        save_predictions(records)
        print(f"\nResolved {resolved_count} predictions.")
    else:
        print("\nNo predictions ready to resolve (need 50+ reviews).")


def show_dashboard():
    """Print performance summary across all resolved predictions."""
    records = load_predictions()
    if not records:
        print("No predictions logged yet. Run 'signal.py movie <slug>' to start.")
        return

    resolved = [r for r in records if r["resolved"]]
    unresolved = [r for r in records if not r["resolved"]]

    print(f"Predictions: {len(records)} total, {len(resolved)} resolved, {len(unresolved)} pending\n")

    if not resolved:
        print("No resolved predictions yet. Run 'signal.py resolve' after movies release.")
        return

    # Overall accuracy
    all_signals = []
    for r in resolved:
        actual = r.get("actual_score")
        if actual is None:
            continue
        for s in r.get("signals", []):
            won = _signal_won(s, actual)
            all_signals.append({**s, "won": won, "movie": r["movie"], "confidence": r.get("confidence")})

    if all_signals:
        wins = sum(1 for s in all_signals if s["won"])
        total = len(all_signals)
        print(f"Signal Accuracy: {wins}/{total} ({wins/total:.0%})\n")

        # By confidence
        for conf in ("HIGH", "MEDIUM", "LOW"):
            subset = [s for s in all_signals if s.get("confidence") == conf]
            if subset:
                w = sum(1 for s in subset if s["won"])
                print(f"  {conf}: {w}/{len(subset)} ({w/len(subset):.0%})")

        # Total simulated P&L
        total_pnl = sum(r.get("pnl", 0) or 0 for r in resolved)
        print(f"\nSimulated P&L: ${total_pnl:+.2f}")
    else:
        print("No trade signals in resolved predictions.")

    # Model accuracy
    print(f"\n--- Model Accuracy ---")
    errors = []
    for r in resolved:
        actual = r.get("actual_score")
        predicted = r.get("model_mean")
        if actual is not None and predicted is not None:
            errors.append(abs(actual - predicted))
            print(f"  {r['movie']}: predicted {predicted}%, actual {actual}%, error {abs(actual-predicted):.1f}")

    if errors:
        mae = sum(errors) / len(errors)
        print(f"\n  Mean Absolute Error: {mae:.1f}%")


def _signal_won(signal, actual_score):
    """Did this signal win?"""
    threshold = signal["threshold"]
    direction = signal["direction"]
    above = actual_score > threshold
    if direction == "BUY YES":
        return above
    else:
        return not above


def _compute_pnl(record, actual_score):
    """Compute simulated P&L for a prediction's signals."""
    total = 0.0
    for s in record.get("signals", []):
        threshold = s["threshold"]
        market_prob = s.get("market_prob", 50) / 100.0
        won = _signal_won(s, actual_score)

        if s["direction"] == "BUY YES":
            cost = market_prob
        else:
            cost = 1.0 - market_prob

        bet_size = 10.0
        if won:
            gross_profit = bet_size * (1.0 - cost) / cost
            total += gross_profit * (1 - config.KALSHI_FEE_RATE)
        else:
            total -= bet_size

    return round(total, 2)
