"""
Does the visible RT score carry information the Kalshi price hasn't absorbed?

Uses the RT readings the live system recorded in each snapshot (model.naive_pct,
model.n_reviews). RT put its review API behind a login on ~2026-07-14, so only
movies whose review window fell before that have readings; later movies are
covered by market_calibration.py instead (prices + outcomes only).

Movies are the independent unit throughout; CIs resample movies.

  1. Information test: logistic regression of outcome on the price's log-odds
     plus z = (RT% - strike) / binomial sd. If RT adds nothing beyond the price,
     z's coefficient is ~0.
  2. The June 26 favorite-longshot rule (flb_test.py, designed on the first 11
     movies), scored separately on movies that settled after it was written.
     Real Kalshi fee.
"""
import math
import random
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

from backtest.market_calibration import kalshi_fee
from data.history import iter_all_snapshots

RULE_WRITTEN = "2026-06-26"
LEADS_H = (72, 48, 24, 12, 6)


def parse(ts):
    return datetime.fromisoformat(ts.replace("Z", "+00:00"))


def load():
    ev = defaultdict(lambda: {"snaps": [], "actual": None})
    for s in iter_all_snapshots():
        e = s.get("event_ticker")
        if not e:
            continue
        ev[e]["snaps"].append((parse(s["timestamp"]), s.get("markets") or {}, s.get("model")))
        if s.get("resolved") and s.get("actual_score") is not None:
            ev[e]["actual"] = s["actual_score"]
    out = {}
    for e, d in ev.items():
        if d["actual"] is None:
            continue
        d["snaps"].sort(key=lambda x: x[0])
        d["close"] = d["snaps"][-1][0]
        out[e] = d
    return out


def snap_at(snaps, when, tol_h=6):
    best = None
    for s in snaps:
        if s[0] <= when:
            best = s
        else:
            break
    return best if best and when - best[0] <= timedelta(hours=tol_h) else None


def build_obs(events):
    rows = []
    for e, d in events.items():
        for lead in LEADS_H:
            s = snap_at(d["snaps"], d["close"] - timedelta(hours=lead))
            if not s or not s[2] or s[2].get("naive_pct") is None:
                continue
            n, naive = s[2].get("n_reviews") or 0, s[2]["naive_pct"]
            if n < 5:
                continue
            p = min(max(naive / 100, 0.02), 0.98)
            sd = 100 * math.sqrt(p * (1 - p) / n)
            for t_str, node in s[1].items():
                try:
                    thr = float(t_str)
                except ValueError:
                    continue
                yb, ya = node.get("yes_bid"), node.get("yes_ask")
                if yb is None or ya is None or ya < yb:
                    continue
                mid = min(max((yb + ya) / 2, 0.01), 0.99)
                rows.append({
                    "ev": e, "lead": lead, "thr": thr, "n": n, "naive": naive,
                    "bid": yb, "ask": ya, "mid": mid,
                    "logit": math.log(mid / (1 - mid)),
                    "z": max(min((naive - thr) / sd, 6), -6),
                    "yes": d["actual"] > thr,
                    "settle_date": d["close"].strftime("%Y-%m-%d"),
                })
    return rows


def logit_fit(X, y, iters=50):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-X @ w))
        H = X.T @ (X * (p * (1 - p))[:, None]) + 1e-6 * np.eye(X.shape[1])
        w = w + np.linalg.solve(H, X.T @ (y - p))
    return w


def info_test(rows, label):
    sel = [r for r in rows if 0.03 < r["mid"] < 0.97]
    pm = defaultdict(list)
    for r in sel:
        pm[r["ev"]].append(r)
    if len(pm) < 5:
        print(f"  {label}: too few movies ({len(pm)})")
        return
    fit = lambda it: logit_fit(np.array([[1, r["logit"], r["z"]] for r in it]),
                               np.array([float(r["yes"]) for r in it]))
    w = fit(sel)
    rng, keys, bz = random.Random(11), list(pm), []
    for _ in range(400):
        try:
            bz.append(fit([x for k in (rng.choice(keys) for _ in keys) for x in pm[k]])[2])
        except np.linalg.LinAlgError:
            pass
    bz.sort()
    lo, hi = bz[int(.025 * len(bz))], bz[int(.975 * len(bz))]
    verdict = ("RT adds info beyond price" if lo > 0 else
               "no evidence RT adds info (already priced)" if lo <= 0 <= hi else
               "RT anti-informative given price")
    print(f"  {label}: {len(sel)} quotes / {len(pm)} movies | price coef {w[1]:+.2f} | "
          f"RT coef {w[2]:+.3f} [{lo:+.3f}, {hi:+.3f}] -> {verdict}")


def june_rule(rows, min_reviews=25, conf_margin=5, lo_band=(0.08, 0.22), hi_band=(0.78, 0.92)):
    taken, trades = set(), []
    for r in sorted(rows, key=lambda r: -r["lead"]):
        if r["n"] < min_reviews:
            continue
        side = None
        if hi_band[0] <= r["mid"] <= hi_band[1] and r["naive"] >= r["thr"] + conf_margin and r["ask"] < 0.97:
            side, price, won = "YES", r["ask"], r["yes"]
        elif lo_band[0] <= r["mid"] <= lo_band[1] and r["naive"] <= r["thr"] - conf_margin and r["bid"] > 0.03:
            side, price, won = "NO", 1 - r["bid"], not r["yes"]
        if side is None or (r["ev"], r["thr"], side) in taken:
            continue
        taken.add((r["ev"], r["thr"], side))
        trades.append({"ev": r["ev"], "won": won, "cost": price,
                       "pnl": (1 - price if won else -price) - kalshi_fee(price),
                       "oos": r["settle_date"] >= RULE_WRITTEN})
    return trades


def rule_report(name, trades):
    if not trades:
        print(f"  {name}: no trades")
        return
    pm = defaultdict(list)
    for t in trades:
        pm[t["ev"]].append(t)
    roi = lambda it: sum(t["pnl"] for t in it) / sum(t["cost"] for t in it)
    rng, keys = random.Random(3), list(pm)
    bs = sorted(roi([x for k in (rng.choice(keys) for _ in keys) for x in pm[k]]) for _ in range(2000))
    pos = sum(1 for e in pm if sum(t["pnl"] for t in pm[e]) > 0)
    print(f"  {name}: {len(trades)} trades / {len(pm)} movies, win {sum(t['won'] for t in trades)/len(trades):.0%}, "
          f"ROI {roi(trades):+.1%} [{bs[50]:+.1%}, {bs[1949]:+.1%}], profitable movies {pos}/{len(pm)}")
    for e in sorted(pm):
        print(f"      {e:12} {len(pm[e])} trades  ${sum(t['pnl'] for t in pm[e]):+.2f} per $1 contracts")


if __name__ == "__main__":
    events = load()
    rows = build_obs(events)
    movies = sorted({r["ev"] for r in rows})
    print(f"{len(events)} settled events; {len(movies)} have live RT readings in the final 72h\n")

    print("=== 1. Does the visible RT% predict outcomes beyond the price? ===")
    info_test(rows, "all lead times      ")
    for lead in (72, 24, 6):
        info_test([r for r in rows if r["lead"] == lead], f"T-{lead}h             ")
    info_test([r for r in rows if r["n"] <= 40], "5-40 reviews window ")

    print("\n=== 2. June 26 favorite-longshot rule ===")
    t = june_rule(rows)
    rule_report("IN-SAMPLE (rule was designed on these)", [x for x in t if not x["oos"]])
    rule_report("OUT-OF-SAMPLE (settled after rule was written)", [x for x in t if x["oos"]])
