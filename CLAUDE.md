# Kalshi RT Trading Algorithm

This is a research project that aims to capture small, consistent edge in
Rotten Tomatoes prediction markets on Kalshi. Realistic target: a few dollars
of profit per month on a small bankroll, not wealth generation. The plumbing
is built; the model and the universe restrictions still need to demonstrate
positive EV on real Kalshi resolutions before any capital is deployed.

## Honest framing (read this every time)

- **Most Kalshi RT markets are efficient.** Big releases with 50+ reviews and
  >$50K volume reflect serious traders who scrape RT, watch embargo lifts in
  real time, and have more capital than us. We do NOT have edge there. If the
  model claims a huge edge on a thick market, that's calibration error.
- **The plausible edge zone is narrow.** Likely pockets: low-coverage titles
  with thin liquidity, the first ~40 reviews before the market fully prices
  the consensus, granular threshold microstructure, and movements during weekend
  reviewer drift. We have not proven we capture any of these yet.
- **Historical Kaggle "backtests" are not validation.** They fit a distribution
  that doesn't match Kalshi market dynamics. Ignore any ROI/win-rate claims
  derived from them. Only resolved Kalshi markets count.
- **Realistic outcome:** 5-20% annualized on a small bankroll, with months of
  drawdown. Acceptable to lose $50-$100 while learning what doesn't work.
- **Unrealistic outcomes:** scaling to thousands of dollars per month, beating
  the market on big releases, ML on review text producing alpha.

## Mission

Find and exploit the small, real edges that exist in inefficient corners of
Kalshi's RT markets. Be honest when there isn't edge. Don't trade efficient
markets just because the model says we should.

## Non-Negotiable Principles

1. **Every model assumption must be validated against real Kalshi outcomes.**
   No hardcoded parameter is "validated" until it has been tested against
   resolved Kalshi markets in the regime we actually trade. Kaggle backtests
   are useful for sanity checks, not validation.

2. **Never trust the model -- verify it.** Before any real trade, the model
   must demonstrate calibrated positive EV on 20+ resolved Kalshi markets in
   the 5-40 review window with calibration within 5pp at every bucket. If
   model says X% and reality is Y%, fix the model first.

3. **Slippage is real.** These are thin markets. Always simulate through the
   actual order book. Never recommend a trade at the quoted midpoint -- show
   the real fill price and the real profit after Kalshi's 7% fee.

4. **Conservative by default.** Quarter Kelly, 5% per-event cap, tiered MIN_EDGE
   that demands more edge as review count and volume increase. When uncertain,
   size down or skip. Losing less on bad signals beats maximizing good ones.

5. **The market is usually right.** If our model disagrees with a liquid
   market by 15+ points, that is a model error, not alpha. Sanity guards
   block trades like this regardless of what the model claims.

## Architecture

```
kalshi-rt/
  web/app.py                  # Flask dashboard + API
  web/templates/dashboard.html # Single-page dark UI
  model/
    distribution.py           # Beta-binomial score distribution (ML-enhanced when trained)
    learned_weights.py        # LightGBM per-review score predictor + training pipeline
    calibration.py            # Platt scaling calibration
    sentiment.py              # Review intensity scoring from numeric ratings + quotes
    prior.py                  # Genre/franchise-adjusted movie-specific prior (replaces flat 63%)
  data/
    critics.py                # Critic database loader with fuzzy matching
  scraper/
    rt_page.py                # Tomatometer + metadata from RT main page
    rt_reviews.py             # Individual reviews via RT NAPI (no browser needed)
    rt_critics.py             # Full critic directory scraper with calibration data
  market/
    kalshi_client.py          # Kalshi API client (markets, orderbook, fill simulation)
    kalshi_auth.py            # Authenticated client (orders, positions, balance)
    mapper.py                 # Event ticker <-> RT movie slug mapping
  engine/
    alpha.py                  # Slippage-aware opportunity detection + Kelly sizing
    sizing.py                 # Timing-aware position sizing adjustments
    paper_trader.py           # Paper/live trading loop: scan markets, find alpha, log/execute
  tracker/
    logger.py                 # Prediction logging to JSONL
    resolver.py               # Outcome recording + P&L tracking
  backtest/
    evaluate.py               # Run model at checkpoints, compare to actual outcomes
    metrics.py                # Brier score, calibration, MAE, simulated P&L, Sharpe
    optimize.py               # Grid search for optimal Beta-binomial parameters
    apply_params.py           # Write optimized params back to config.py + calibration.py
    train_model.py            # End-to-end: train ML model, compare vs baseline, report winner
    weight_eval.py            # Empirical test of 7 hand-crafted weight functions
  config.py                   # Central config with VALIDATED/UNVALIDATED markers
  critic_database.csv         # Current critic weights (being refreshed)
  critic_reviews.json         # Rich per-critic calibration data
```

## Key APIs

- **Kalshi**: `https://api.elections.kalshi.com/trade-api/v2` -- markets, orderbook (no auth for read-only)
- **RT Reviews**: `https://www.rottentomatoes.com/napi/rtcf/v1/movies/{ems_id}/reviews` -- critic reviews by movie
- **RT Critics**: `https://www.rottentomatoes.com/napi/critics/authors` -- critic directory
- **RT Critic Reviews**: `https://www.rottentomatoes.com/napi/critics/{slug}/movies` -- a critic's review history

## Key Design Decisions

- **Big-movie weighting**: Critics are weighted by their agreement rate on movies with 80+ reviews, not their overall rate. A critic being "right" about a tiny indie tells us nothing about their accuracy on the major releases Kalshi lists.
- **Settlement score vs final score**: Kalshi resolves on Monday after wide release at 10 AM ET. Critics keep reviewing after that, so the "final" RT score may differ from the settlement score. We track the Monday settlement score for model training, not the eventual final score.
- **ML hybrid architecture**: A LightGBM model predicts what each individual review implies about the final tomatometer (using critic stats, sentiment, numeric scores, movie context). These per-review predictions feed into the Beta-binomial framework for uncertainty quantification. ML handles signal extraction; Bayesian inference handles calibrated uncertainty. Falls back to hand-crafted weighting if ML model isn't trained yet.
- **Movie-specific prior**: Instead of a flat 63% population mean, the model uses genre/franchise/rating-adjusted priors (e.g., action sequels ~53%, dramas ~67%, documentaries ~85%). This matters most with few reviews, where the prior has the most influence on the prediction.
- **Confidence-adjusted Kelly sizing**: Position size scales with model confidence (HIGH=1.0x, MEDIUM=0.6x, LOW=0.3x). Combined with time-to-settlement multipliers (1.3x near settlement, 0.7x a week out). A LOW confidence signal 7 days from settlement gets 0.21x the base Kelly size.
- **Portfolio risk limits**: 10% max exposure per event (movie), 30% max total portfolio exposure. Prevents overconcentration even when the model finds multiple signals on the same movie.

## Known Weaknesses (fix these before real money)

- Model predicted 69% for Power Ballad when RT was 88% -- critic database staleness is the primary cause
- Calibration parameters are not yet fitted from backtesting data (framework ready, waiting for scraper)
- Genre base rates in prior.py are approximate, need validation against backtest data
- Agreement rate for unknown critics uses a blunt median default

## Running

```bash
python3 web/app.py                # Dashboard at http://localhost:5001
python3 -m scraper.rt_critics     # Refresh critic database (~30 min full run)
python3 -m backtest.train_model   # Train ML model, compare vs baseline, pick winner
python3 -m backtest.evaluate      # Run backtest (needs critic_reviews.json + movie_scores.json)
python3 -m backtest.optimize      # Grid search for optimal Beta-binomial parameters
python3 -m backtest.apply_params  # Apply optimized params to config.py + calibration.py
python3 -m backtest.weight_eval   # Compare 7 hand-crafted weight functions (if not using ML)
python3 -m tracker.settlement                    # show upcoming events
python3 -m tracker.settlement --settle-morning   # SETTLEMENT DAY: snapshot + resolve + edge report
python3 -m tracker.market_snapshot --status      # check snapshot loop health
python3 -m engine.paper_trader    # Single paper trading pass across all active markets
python3 -m engine.paper_trader --loop 300   # Re-check every 5 min
python3 -m engine.paper_trader --live       # REAL MONEY (requires ~/.config/kalshi-rt/credentials.json)
python3 -m engine.paper_trader --summary    # Show paper trading performance
```

## Before Deploying Real Money

Each gate must be cleared by **resolved Kalshi markets**, not Kaggle backtests.

- [ ] 20+ resolved Kalshi markets in the 5-40 review window, with full
      snapshot history and final settlement scores
- [ ] Calibration within 5pp at every probability bucket (35%, 45%, 55%,
      65%, 75%, 85%, 95%) -- post any refit, validated out of sample
- [ ] Positive simulated P&L on those 20+ resolutions using current config
      (tiered MIN_EDGE, sanity guards, Kelly sizing)
- [ ] Win rate above 55% on HIGH confidence signals across the 20+ resolutions
- [ ] No systematic loss in any review-count bucket
      (see `market_learnings.json:review_count_performance`)
- [ ] Critic database refreshed within the last 60 days
- [ ] Slippage-aware fills within 2% of actual Kalshi executions on a
      manual test trade ($5-$10 max)

## Capital deployment plan (after the gates above are cleared)

- Initial real bankroll: **$200 max**
- Per-position cap: **$25**
- Per-event cap: 5% of bankroll
- Restricted universe: markets in the 5-40 review window first; expand
  to 40+ only after that subset shows positive realized edge
- Scale-up rule: only after 30+ days of positive realized P&L on real money,
  and only by doubling bankroll at most. No "if it works, let it rip."
- Hard kill: if real-money drawdown exceeds 25% of starting bankroll,
  stop trading and revisit calibration before resuming.

## Roadmap (not yet built)

### Continuous Data Freshness
- Critic database and ML model weights must stay current. New critics appear, existing critics' accuracy drifts. Need automated periodic re-scraping of the RT critic directory (weekly or biweekly) and model retraining when new settlement data comes in.
- Every movie the system analyzes should feed back into the training set after settlement. The model should get better with every resolution, not just when we manually re-run the scraper.
- Monitor for data staleness: alert if the critic database is >30 days old or the ML model hasn't been retrained after 5+ new settlements.

### Autonomous Trading
- System must run unattended 24/7: monitor new Kalshi RT markets, scrape reviews as they appear, generate and execute trades without human intervention.
- Requires authenticated Kalshi API integration (order placement, position tracking, balance monitoring).
- Needs robust error handling: RT scraper failures, Kalshi API downtime, network issues, rate limits. Must degrade gracefully (pause trading, don't crash) and resume automatically.
- Health monitoring and alerting: heartbeat checks, P&L tracking, position limit enforcement, circuit breakers if losses exceed thresholds.
- Monday settlement automation: snapshot scores at 10 AM ET, resolve predictions, record P&L, trigger model retrain if enough new data.
- Deployment target: long-running process or scheduled jobs on a server (not Griffin's laptop).
