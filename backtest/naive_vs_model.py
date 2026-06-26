"""
Naive-vs-Model bake-off on real settled Kalshi events.

Isolates ONE variable: the center of the score distribution.
  - MODEL strategy:  center = model_mean  (uses model's threshold_probs directly)
  - NAIVE strategy:  center = naive_pct   (visible RT %), same sigma, same guards

Everything else identical: same per-snapshot uncertainty (from the model's own CI),
same edge threshold, same liquidity guards, same top-of-book fills, same sizing,
same 7% fee. One entry per (event, threshold, side), taken the first time the
signal fires, held to settlement. Settles against the real Kalshi score.
"""
import json, math
from collections import defaultdict

SNAP = "data/snapshots.jsonl"
FEE = 0.07
STAKE = 50.0          # $ notional per position
MIN_EDGE = 0.10       # require 10pp edge to trade (matches engine spirit)
MIN_BID, MAX_ASK = 0.05, 0.95   # require a real two-sided quote
MAX_SPREAD = 0.20

SETTLED = {'KXRT-BAC':89,'KXRT-DIS':80,'KXRT-MAS':66,'KXRT-POW':86,
           'KXRT-PRE':87,'KXRT-SCA':24,'KXRT-STA':62,'KXRT-STO':84,
           # thin titles added 2026-06-26 -- the universe the edge was supposed to live in
           'KXRT-DEA':70,'KXRT-GIR':89,'KXRT-TOY':94}

def ncdf(x, mu, sigma):
    if sigma <= 0: sigma = 1e-6
    return 0.5 * (1 + math.erf((x - mu) / (sigma * math.sqrt(2))))

def p_above(center, thr, sigma):
    # P(final_score > thr)
    return 1.0 - ncdf(thr, center, sigma)

def run(strategy, review_lo=0, review_hi=10**9, max_vol=None):
    """strategy in {'model','naive'}. Returns (trades, wins, pnl, per_event)."""
    taken = set()            # (event, thr, side) dedupe
    trades = []
    # iterate snapshots in time order (file is chronological per the snapshot loop)
    for line in open(SNAP):
        try: d = json.loads(line)
        except: continue
        ev = d.get('event_ticker')
        if ev not in SETTLED: continue
        m = d.get('model') or {}
        nr = m.get('n_reviews')
        if nr is None or not (review_lo <= nr <= review_hi): continue
        if max_vol is not None and (d.get('total_volume') or 0) > max_vol: continue
        ci = m.get('model_ci') or []
        if len(ci) != 2: continue
        sigma = max((ci[1]-ci[0]) / 3.92, 1.0)
        center = m.get('model_mean') if strategy=='model' else m.get('naive_pct')
        if center is None: continue
        actual = SETTLED[ev]
        markets = d.get('markets') or {}
        for thr_s, node in markets.items():
            try: thr = float(thr_s)
            except: continue
            yb, ya = node.get('yes_bid'), node.get('yes_ask')
            yp = node.get('yes_price')
            if yb is None or ya is None or yp is None: continue
            if yb < MIN_BID or ya > MAX_ASK or (ya - yb) > MAX_SPREAD: continue
            p_yes = p_above(center, thr, sigma)
            # YES side: buy at ask, wins if actual > thr
            edge_yes = p_yes - ya
            # NO side: buy at (1-yb), wins if actual <= thr
            edge_no = (1 - p_yes) - (1 - yb)
            if edge_yes >= MIN_EDGE and edge_yes >= edge_no:
                side, price, won = 'YES', ya, (actual > thr)
            elif edge_no >= MIN_EDGE:
                side, price, won = 'NO', (1 - yb), (actual <= thr)
            else:
                continue
            key = (ev, thr, side)
            if key in taken: continue
            taken.add(key)
            contracts = STAKE / price
            payout = contracts if won else 0.0
            cost = contracts * price
            gross = payout - cost
            pnl = gross - (FEE * gross if gross > 0 else 0)
            trades.append({'ev':ev,'thr':thr,'side':side,'price':price,
                           'won':won,'pnl':pnl,'nr':nr,'center':center})
    return trades

def report(name, trades):
    if not trades:
        print(f"  {name}: no trades"); return
    n = len(trades); w = sum(t['won'] for t in trades); pnl = sum(t['pnl'] for t in trades)
    print(f"  {name}: {n} trades, {w}W/{n-w}L ({100*w/n:.0f}%), net ${pnl:.2f}, ${pnl/n:.2f}/trade")
    be = defaultdict(lambda:[0,0,0.0])
    for t in trades:
        be[t['ev']][0]+=1; be[t['ev']][1]+=t['won']; be[t['ev']][2]+=t['pnl']
    for ev in sorted(be):
        c,ww,p = be[ev]
        print(f"      {ev}: {c:2} trades {ww}W  ${p:7.2f}")

for label, lo, hi, mv in [
    ("ALL settled events, all review counts", 0, 10**9, None),
    ("EARLY window (5-40 reviews)", 5, 40, None),
    ("THIN markets (max volume < 250k)", 0, 10**9, 250000),
]:
    print(f"\n=== {label} ===")
    for strat in ('model','naive'):
        report(strat.upper(), run(strat, lo, hi, mv))
