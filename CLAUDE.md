# Kalshi RT Trading Algorithm

This is a real-money trading system. It will be funded with actual capital and expected to generate returns. Treat every decision with the rigor of production quantitative finance, not a prototype.

## Mission

Build the most effective algorithm for trading Rotten Tomatoes prediction markets on Kalshi. The edge comes from processing early critic reviews faster and more accurately than the market prices them.

## Non-Negotiable Principles

1. **Every model assumption must be validated against data.** No hardcoded parameters without backtesting justification. If a number appears in the model (prior mean, shrinkage rate, minimum edge threshold), there must be evidence it's correct or a TODO to validate it.

2. **Never trust the model -- verify it.** Before any real trade, the model must demonstrate positive expected value on historical data. Track every prediction and compare to outcomes. If the model says X% and reality is consistently Y%, fix the model.

3. **Slippage is real.** These are thin markets. Always simulate through the actual order book. Never recommend a trade at the quoted price -- show the real fill price and the real profit after Kalshi's 7% fee.

4. **Conservative by default.** Quarter Kelly, 5% bankroll cap, minimum 25% win probability. When uncertain, size down, not up. Losing less on bad signals matters more than maximizing good ones.

5. **The critic database is the moat.** The quality of per-critic calibration data directly determines model accuracy. Invest in making this as rich and current as possible.

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
python3 -m tracker.settlement     # Monday morning: snapshot settlement scores
python3 -m engine.paper_trader    # Single paper trading pass across all active markets
python3 -m engine.paper_trader --loop 300   # Re-check every 5 min
python3 -m engine.paper_trader --live       # REAL MONEY (requires ~/.config/kalshi-rt/credentials.json)
python3 -m engine.paper_trader --summary    # Show paper trading performance
```

## Before Deploying Real Money

- [ ] Critic database refreshed with real agreement rates (not approximated)
- [ ] Backtested against 20+ resolved movies with known outcomes
- [ ] Calibration parameters fitted from backtest, not hardcoded
- [ ] Paper trading for 15+ resolutions with positive simulated P&L
- [ ] Model accuracy: predicted probabilities within 10% of actual hit rates
- [ ] Win rate >55% on HIGH confidence signals
- [ ] Slippage-aware sizing validated against actual Kalshi fills

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
