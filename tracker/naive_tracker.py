"""
Naive-strategy parallel tracker.

The naive-vs-model bake-off showed, on real settled Kalshi markets, that a
strategy centered on the *visible RT score* beats one centered on the model's
adjusted mean in every cut (model -$431 vs naive +$964 all-settled). That was
in-sample. This module runs the naive rule FORWARD, live, alongside the model's
paper trader, so we can judge it out-of-sample as new movies settle.

The naive rule, held identical to the bake-off:
  - center the score distribution on naive_pct (the plain fresh-rate)
  - same uncertainty (sigma from the model's own CI), same 10pp edge floor,
    same two-sided-quote guard, same top-of-book fill, same 7% fee
  - one entry per (event, threshold, side), first time the signal fires

Logged to data/naive_trades.jsonl, scored at settlement by settlement.py.
This is NOT a real trading strategy yet -- it's a live validation stream.
The honest caveat from the bake-off stands: naive's edge is concentrated in
a couple of events and fills ignore slippage, so the bar is whether it holds
up out-of-sample on thin markets, not the headline dollars.
"""
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import config

ROOT = Path(__file__).parent.parent
SNAPSHOT_FILE = ROOT / "data" / "snapshots.jsonl"
NAIVE_FILE = ROOT / "data" / "naive_trades.jsonl"

STAKE = 50.0
MIN_EDGE = 0.10
MIN_BID, MAX_ASK = 0.05, 0.95
MAX_SPREAD = 0.20


def _ncdf(x, mu, sigma):
    if sigma <= 0:
        sigma = 1e-6
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))


def _p_above(center, thr, sigma):
    return 1.0 - _ncdf(thr, center, sigma)


def _latest_snapshots():
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
    keys = set()
    if not NAIVE_FILE.exists():
        return keys
    with open(NAIVE_FILE) as f:
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


def log_naive_trades():
    """Scan latest snapshots, log first-fire naive-rule signals."""
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
        naive = model.get("naive_pct")
        ci = model.get("model_ci") or []
        if naive is None or len(ci) != 2:
            continue
        sigma = max((ci[1] - ci[0]) / 3.92, 1.0)
        n_reviews = model.get("n_reviews", 0)
        volume = snap.get("total_volume", 0)

        for t_str, node in (snap.get("markets") or {}).items():
            try:
                thr = float(t_str)
            except ValueError:
                continue
            yb, ya, yp = node.get("yes_bid"), node.get("yes_ask"), node.get("yes_price")
            if yb is None or ya is None or yp is None:
                continue
            if yb < MIN_BID or ya > MAX_ASK or (ya - yb) > MAX_SPREAD:
                continue

            p_yes = _p_above(naive, thr, sigma)
            edge_yes = p_yes - ya
            edge_no = (1 - p_yes) - (1 - yb)

            if edge_yes >= MIN_EDGE and edge_yes >= edge_no:
                direction, price, edge = "BUY YES", ya, edge_yes
            elif edge_no >= MIN_EDGE:
                direction, price, edge = "BUY NO", (1 - yb), edge_no
            else:
                continue

            key = (ticker, int(thr), direction)
            if key in existing:
                continue
            existing.add(key)

            new_rows.append({
                "logged_at": now,
                "event_ticker": ticker,
                "movie": snap.get("movie"),
                "rt_slug": snap.get("rt_slug"),
                "threshold": int(thr),
                "direction": direction,
                "entry_price": round(price, 4),
                "edge_pct": round(edge * 100, 1),
                "naive_pct": naive,
                "model_mean": model.get("model_mean"),
                "n_reviews": n_reviews,
                "total_volume": round(volume, 0),
                "resolved": False,
                "actual_score": None,
                "won": None,
                "pnl": None,
            })

    if new_rows:
        NAIVE_FILE.parent.mkdir(parents=True, exist_ok=True)
        with open(NAIVE_FILE, "a") as f:
            for r in new_rows:
                f.write(json.dumps(r) + "\n")

    print(f"Logged {len(new_rows)} new naive-strategy signal(s).")
    for r in new_rows:
        print(f"  {r['movie'][:22]:22} {r['direction']} >{r['threshold']}% "
              f"@ {r['entry_price']*100:.0f}c  edge={r['edge_pct']}%  n_rev={r['n_reviews']}")
    return new_rows


def score_naive_trades(event_ticker, actual_score):
    """Score naive-strategy signals for a settled market. Top-of-book fill,
    7% fee on profit, $50 notional -- same accounting as the bake-off."""
    if not NAIVE_FILE.exists():
        return []
    rows = []
    with open(NAIVE_FILE) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    scored = []
    for r in rows:
        if r["event_ticker"] != event_ticker or r.get("resolved"):
            continue
        thr, entry = r["threshold"], r["entry_price"]
        won = (actual_score > thr) if r["direction"] == "BUY YES" else (actual_score <= thr)
        contracts = STAKE / entry if entry > 0 else 0
        if won:
            gross = contracts * (1.0 - entry)
            pnl = gross * (1 - config.KALSHI_FEE_RATE)
        else:
            pnl = -STAKE
        r["resolved"] = True
        r["actual_score"] = actual_score
        r["won"] = won
        r["pnl"] = round(pnl, 2)
        scored.append(r)
    with open(NAIVE_FILE, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    total = sum(r["pnl"] for r in scored)
    wins = sum(1 for r in scored if r["won"])
    print(f"\nNaive strategy scored {len(scored)} signals for {event_ticker} "
          f"(settled {actual_score}%): ${total:+.2f} ({wins}/{len(scored)} won)")
    return scored


def summary():
    if not NAIVE_FILE.exists():
        print("No naive-strategy signals logged yet.")
        return
    rows = [json.loads(l) for l in open(NAIVE_FILE) if l.strip()]
    resolved = [r for r in rows if r.get("resolved")]
    pending = [r for r in rows if not r.get("resolved")]
    total = sum(r["pnl"] for r in resolved if r.get("pnl") is not None)
    wins = sum(1 for r in resolved if r.get("won"))
    print(f"Naive strategy: {len(rows)} signals, {len(resolved)} scored, {len(pending)} pending")
    if resolved:
        print(f"  Scored naive P&L: ${total:+.2f} ({wins}/{len(resolved)} won)")
        # Per-event so we can see concentration (the bake-off's key caveat).
        by_ev = {}
        for r in resolved:
            by_ev.setdefault(r["event_ticker"], 0.0)
            by_ev[r["event_ticker"]] += r["pnl"]
        for ev, p in sorted(by_ev.items(), key=lambda x: -x[1]):
            print(f"    {ev}: ${p:+.2f}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Naive-strategy parallel tracker")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--score", nargs=2, metavar=("TICKER", "SCORE"))
    parser.add_argument("--summary", action="store_true")
    args = parser.parse_args()
    if args.log:
        log_naive_trades()
    elif args.score:
        score_naive_trades(args.score[0], int(args.score[1]))
    elif args.summary:
        summary()
    else:
        parser.print_help()
