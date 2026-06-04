"""
Shadow-trade tracker: measure the opportunity cost of the sanity guard.

The sanity guard blocks trades where the model claims a large edge on a
liquid market, on the theory that the market is usually right. That's a
reasonable prior, but it's untested. This module logs the trades the guard
blocked -- as "shadow" positions -- so we can score them after settlement
and learn whether the guard is saving us from model errors or costing us
real alpha.

A shadow trade is an opportunity that:
  - cleared the review-count tier's min-edge requirement
  - cleared the min win-probability floor
  - would have generated a real trade
  - BUT was blocked solely by the sanity guard (edge too large for the
    market's liquidity)

We log entry prices at detection time so scoring is realistic. After a
market settles, score_shadow_trades() computes win/loss and P&L exactly
as if the trades had been taken.

Usage:
  python -m tracker.shadow_tracker --log              # scan + log current blocked opps
  python -m tracker.shadow_tracker --score TICKER N   # score a settled market
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import config

ROOT = Path(__file__).parent.parent
SNAPSHOT_FILE = ROOT / "data" / "snapshots.jsonl"
SHADOW_FILE = ROOT / "data" / "shadow_trades.jsonl"


def _latest_snapshots():
    """Return the most recent snapshot per event ticker."""
    latest = {}
    if not SNAPSHOT_FILE.exists():
        return latest
    with open(SNAPSHOT_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                s = json.loads(line)
            except json.JSONDecodeError:
                continue
            t = s.get("event_ticker")
            if not t:
                continue
            if t not in latest or s["timestamp"] > latest[t]["timestamp"]:
                latest[t] = s
    return latest


def _existing_keys():
    """Keys already logged, so we don't double-record the same shadow trade.

    Keyed by (event_ticker, threshold, direction) -- we keep only the FIRST
    detection of each so the entry price reflects when the guard first
    blocked it, not a later re-scan.
    """
    keys = set()
    if not SHADOW_FILE.exists():
        return keys
    with open(SHADOW_FILE) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            keys.add((r["event_ticker"], r["threshold"], r["direction"]))
    return keys


def log_shadow_trades():
    """Scan latest snapshots, log opportunities blocked solely by the sanity guard."""
    latest = _latest_snapshots()
    existing = _existing_keys()
    now = datetime.now(timezone.utc).isoformat()

    new_rows = []
    for ticker, snap in latest.items():
        if snap.get("resolved"):
            continue
        model = snap.get("model")
        if not model:
            continue

        n_reviews = model.get("n_reviews", 0)
        volume = snap.get("total_volume", 0)
        tier_min_edge, tier_enabled = config.min_edge_for(n_reviews, volume)
        if not tier_enabled:
            continue

        edges = model.get("edges", {})
        markets = snap.get("markets", {})

        for t_str, edge_pct in edges.items():
            edge = edge_pct / 100.0
            threshold = int(t_str)
            mkt = markets.get(t_str, {})

            # Determine direction + entry price the same way the engines do.
            if edge > 0:
                direction = "BUY YES"
                entry = mkt.get("yes_ask") or mkt.get("yes_price")
                win_prob = (model.get("threshold_probs", {}) or {}).get(t_str)
            else:
                direction = "BUY NO"
                yes_bid = mkt.get("yes_bid")
                entry = (1.0 - yes_bid) if yes_bid is not None else None
                tp = (model.get("threshold_probs", {}) or {}).get(t_str)
                win_prob = (1.0 - tp) if tp is not None else None

            if entry is None or win_prob is None:
                continue

            abs_edge = abs(edge)

            # Only consider opportunities with at least a small edge -- below
            # this we'd be logging pure noise. This floor (5%) is intentionally
            # looser than the tier minimum so we capture trades blocked by the
            # tier requirement too, not just sanity blocks.
            if abs_edge < 0.05:
                continue

            # Evaluate every filter independently so we know exactly why each
            # opportunity didn't trade.
            clears_tier = abs_edge >= tier_min_edge
            clears_winprob = win_prob >= config.MIN_WIN_PROB
            blocked_by_sanity = config.sanity_blocks(abs_edge, volume)

            # If it passes everything, the real paper trader would have taken
            # it -- not a shadow trade.
            if clears_tier and clears_winprob and not blocked_by_sanity:
                continue

            # Record every reason it was blocked (can be more than one).
            reasons = []
            if blocked_by_sanity:
                reasons.append("sanity_guard")
            if not clears_tier:
                reasons.append("below_tier_min")
            if not clears_winprob:
                reasons.append("win_prob_floor")

            key = (ticker, threshold, direction)
            if key in existing:
                continue
            existing.add(key)

            new_rows.append({
                "logged_at": now,
                "event_ticker": ticker,
                "movie": snap.get("movie"),
                "rt_slug": snap.get("rt_slug"),
                "threshold": threshold,
                "direction": direction,
                "entry_price": round(entry, 4),
                "model_prob": round(win_prob, 4),
                "edge_pct": round(abs_edge * 100, 1),
                "tier_min_edge_pct": round(tier_min_edge * 100, 1),
                "n_reviews": n_reviews,
                "total_volume": round(volume, 0),
                "blocked_by": reasons,                 # list -- may be multiple
                "blocked_by_primary": reasons[0],      # for quick grouping
                "model_mean": model.get("model_mean"),
                "naive_pct": model.get("naive_pct"),
                "resolved": False,
                "actual_score": None,
                "won": None,
                "shadow_pnl": None,
            })

    if new_rows:
        SHADOW_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(SHADOW_FILE, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")

    print(f"Logged {len(new_rows)} new shadow trade(s).")
    for r in new_rows:
        print(f"  {r['movie'][:22]:22} {r['direction']} >{r['threshold']}% "
              f"@ {r['entry_price']*100:.0f}c  edge={r['edge_pct']}%  "
              f"vol=${r['total_volume']:,.0f}  n_rev={r['n_reviews']}  "
              f"blocked_by={','.join(r['blocked_by'])}")
    return new_rows


def score_shadow_trades(event_ticker, actual_score, bankroll_size=50.0):
    """Score all shadow trades for a settled market.

    Uses a fixed notional bankroll_size per trade (default $50) since shadow
    trades were never sized by the optimizer. P&L uses Kalshi's 7% fee on
    profit. Rewrites SHADOW_FILE in place with resolution fields filled in.
    """
    if not SHADOW_FILE.exists():
        print("No shadow trades file.")
        return []

    rows = []
    with open(SHADOW_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    scored = []
    for r in rows:
        if r["event_ticker"] != event_ticker or r.get("resolved"):
            continue
        threshold = r["threshold"]
        entry = r["entry_price"]
        if r["direction"] == "BUY YES":
            won = actual_score > threshold
        else:
            won = actual_score <= threshold

        contracts = bankroll_size / entry if entry > 0 else 0
        if won:
            gross = contracts * (1.0 - entry)
            pnl = gross * (1 - config.KALSHI_FEE_RATE)
        else:
            pnl = -bankroll_size

        r["resolved"] = True
        r["actual_score"] = actual_score
        r["won"] = won
        r["shadow_pnl"] = round(pnl, 2)
        scored.append(r)

    # Rewrite file with updates
    with open(SHADOW_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    total = sum(r["shadow_pnl"] for r in scored)
    wins = sum(1 for r in scored if r["won"])
    print(f"\nScored {len(scored)} shadow trades for {event_ticker} (settled at {actual_score}%):")
    for r in scored:
        tag = "WIN " if r["won"] else "LOSS"
        print(f"  {tag} {r['direction']} >{r['threshold']}% @ {r['entry_price']*100:.0f}c "
              f"-> ${r['shadow_pnl']:+.2f}  [{','.join(_reasons(r))}]")
    print(f"\n  Total shadow P&L: ${total:+.2f} ({wins}/{len(scored)} won)")

    # Break down by which filter blocked each trade, so we learn whether each
    # filter is helping or hurting -- independently.
    by_reason = {}
    for r in scored:
        for reason in _reasons(r):
            by_reason.setdefault(reason, []).append(r["shadow_pnl"])
    print(f"  By filter (would-have-been P&L if that filter were removed):")
    for reason, pnls in sorted(by_reason.items()):
        s = sum(pnls)
        verb = "COST us" if s > 0 else ("SAVED us" if s < 0 else "neutral")
        print(f"    {reason:16} {len(pnls):>2} trades  ${s:+.2f}  -> filter {verb} ${abs(s):.2f}")
    return scored


def _reasons(r):
    """Normalize blocked_by to a list (old records stored a string)."""
    b = r.get("blocked_by")
    if isinstance(b, str):
        return [b]
    return b or []


def summary():
    """Print a running tally of all scored shadow trades across markets."""
    if not SHADOW_FILE.exists():
        print("No shadow trades logged yet.")
        return
    rows = []
    with open(SHADOW_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    resolved = [r for r in rows if r.get("resolved")]
    pending = [r for r in rows if not r.get("resolved")]
    total = sum(r["shadow_pnl"] for r in resolved if r.get("shadow_pnl") is not None)
    wins = sum(1 for r in resolved if r.get("won"))
    print(f"Shadow trades: {len(rows)} total, {len(resolved)} scored, {len(pending)} pending")
    if resolved:
        print(f"  Scored shadow P&L: ${total:+.2f} ({wins}/{len(resolved)} won)")
        # Per-filter breakdown: would removing each filter have made money?
        by_reason = {}
        for r in resolved:
            if r.get("shadow_pnl") is None:
                continue
            for reason in _reasons(r):
                by_reason.setdefault(reason, []).append(r["shadow_pnl"])
        print(f"  By filter (P&L of trades each filter blocked):")
        for reason, pnls in sorted(by_reason.items()):
            s = sum(pnls)
            verb = "COST us" if s > 0 else ("SAVED us" if s < 0 else "neutral")
            print(f"    {reason:16} {len(pnls):>2} blocked  ${s:+.2f}  -> {verb} ${abs(s):.2f}")
    if pending:
        from collections import Counter
        by_movie = Counter(r["movie"] for r in pending)
        by_reason_p = Counter(reason for r in pending for reason in _reasons(r))
        print(f"  Pending by movie: {dict(by_movie)}")
        print(f"  Pending by block reason: {dict(by_reason_p)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shadow-trade tracker for sanity-guard opportunity cost")
    parser.add_argument("--log", action="store_true", help="Scan latest snapshots and log blocked opportunities")
    parser.add_argument("--score", nargs=2, metavar=("TICKER", "SCORE"), help="Score a settled market")
    parser.add_argument("--summary", action="store_true", help="Show running tally")
    args = parser.parse_args()

    if args.log:
        log_shadow_trades()
    elif args.score:
        score_shadow_trades(args.score[0], int(args.score[1]))
    elif args.summary:
        summary()
    else:
        parser.print_help()
