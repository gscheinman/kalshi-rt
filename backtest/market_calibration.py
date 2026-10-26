"""
Model-free test: is the Kalshi RT market itself miscalibrated?

For every settled event and every strike, read the quote at fixed lead times
before close (72h, 24h, 6h). One observation per (event, strike, lead time),
so 15-minute snapshots don't get counted thousands of times.

Movies are the independent unit: all strikes on one movie resolve off one
number. Confidence intervals come from a bootstrap that resamples MOVIES.

Fees: Kalshi taker fee = 0.07 * contracts * P * (1 - P), charged on every fill.
"""
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta

from data.history import iter_all_snapshots

LEADS_H = (72, 24, 6)
FEE_K = 0.07


def kalshi_fee(price):
    return FEE_K * price * (1 - price)


def load():
    """Return {event: {'actual', 'close', 'snaps': [(ts, markets)]}}."""
    ev = defaultdict(lambda: {"snaps": [], "actual": None})
    for s in iter_all_snapshots():
        e = s.get("event_ticker")
        if not e:
            continue
        ts = datetime.fromisoformat(s["timestamp"].replace("Z", "+00:00"))
        ev[e]["snaps"].append((ts, s.get("markets") or {}))
        if s.get("resolved") and s.get("actual_score") is not None:
            ev[e]["actual"] = s["actual_score"]
    out = {}
    for e, d in ev.items():
        if d["actual"] is None or not d["snaps"]:
            continue
        d["snaps"].sort(key=lambda x: x[0])
        d["close"] = d["snaps"][-1][0]
        out[e] = d
    return out


def quote_at(snaps, when, tol_h=6):
    """Latest snapshot at or before `when`, within tol_h hours."""
    best = None
    for ts, mk in snaps:
        if ts <= when:
            best = (ts, mk)
        else:
            break
    if best and (when - best[0]) <= timedelta(hours=tol_h):
        return best[1]
    return None


def observations(events):
    rows = []
    for e, d in events.items():
        for lead in LEADS_H:
            mk = quote_at(d["snaps"], d["close"] - timedelta(hours=lead))
            if not mk:
                continue
            for t_str, node in mk.items():
                try:
                    thr = float(t_str)
                except ValueError:
                    continue
                yb, ya = node.get("yes_bid"), node.get("yes_ask")
                if yb is None or ya is None or ya <= 0 or yb >= 1 or ya < yb:
                    continue
                mid = (yb + ya) / 2
                rows.append({
                    "ev": e, "lead": lead, "thr": thr, "bid": yb, "ask": ya,
                    "mid": mid, "spread": ya - yb, "yes": d["actual"] > thr,
                })
    return rows


def boot_ci(per_movie_values, stat, n=2000, seed=7):
    """Resample movies with replacement; per_movie_values: {ev: [items]}."""
    rng = random.Random(seed)
    keys = list(per_movie_values)
    vals = []
    for _ in range(n):
        sample = [rng.choice(keys) for _ in keys]
        items = [x for k in sample for x in per_movie_values[k]]
        v = stat(items)
        if v is not None:
            vals.append(v)
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def calibration(rows):
    print("\n=== Market calibration (mid price vs how often YES actually hit) ===")
    print("Each row: one quote per (movie, strike, lead time). CI resamples movies.")
    buckets = [(0.02, 0.10), (0.10, 0.25), (0.25, 0.45), (0.45, 0.55),
               (0.55, 0.75), (0.75, 0.90), (0.90, 0.98)]
    print(f"{'price band':>12} {'n':>5} {'movies':>6} {'avg price':>9} {'hit rate':>8} {'gap':>7}   95% CI of gap")
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["mid"] < hi]
        if len(sel) < 10:
            continue
        pm = defaultdict(list)
        for r in sel:
            pm[r["ev"]].append(r)
        gap = lambda it: (sum(r["yes"] for r in it) / len(it) - sum(r["mid"] for r in it) / len(it)) if it else None
        lo_ci, hi_ci = boot_ci(pm, gap)
        hit = sum(r["yes"] for r in sel) / len(sel)
        avg = sum(r["mid"] for r in sel) / len(sel)
        print(f"{lo:>5.2f}-{hi:<5.2f} {len(sel):>5} {len(pm):>6} {avg:>9.3f} {hit:>8.3f} {hit-avg:>+7.3f}   [{lo_ci:+.3f}, {hi_ci:+.3f}]")


def flb_strategy(rows, lead, lo_band=(0.05, 0.20), hi_band=(0.80, 0.95), max_spread=0.10):
    """Model-free extremes: buy NO on cheap YES, buy YES on expensive YES.
    Taker fills (pay the ask), real Kalshi fee, $1 notional per contract."""
    trades = []
    for r in rows:
        if r["lead"] != lead or r["spread"] > max_spread:
            continue
        if lo_band[0] <= r["mid"] <= lo_band[1]:
            price = 1 - r["bid"]          # buy NO
            won = not r["yes"]
        elif hi_band[0] <= r["mid"] <= hi_band[1]:
            price = r["ask"]              # buy YES
            won = r["yes"]
        else:
            continue
        pnl = (1 - price if won else -price) - kalshi_fee(price)
        trades.append({"ev": r["ev"], "pnl": pnl, "cost": price, "won": won})
    return trades


def report(name, trades):
    if not trades:
        print(f"  {name}: no trades")
        return
    pm = defaultdict(list)
    for t in trades:
        pm[t["ev"]].append(t)
    roi = lambda it: sum(t["pnl"] for t in it) / sum(t["cost"] for t in it) if it else None
    lo, hi = boot_ci(pm, roi)
    pos = sum(1 for e in pm if sum(t["pnl"] for t in pm[e]) > 0)
    print(f"  {name}: {len(trades)} contracts on {len(pm)} movies, win {sum(t['won'] for t in trades)/len(trades):.0%}, "
          f"ROI {roi(trades):+.1%}  95% CI [{lo:+.1%}, {hi:+.1%}]  profitable movies {pos}/{len(pm)}")


if __name__ == "__main__":
    events = load()
    rows = observations(events)
    print(f"{len(events)} settled events, {len(rows)} quote observations")
    calibration(rows)
    print("\n=== Model-free extremes strategy (no reviews used) ===")
    for lead in LEADS_H:
        report(f"T-{lead}h", flb_strategy(rows, lead))
