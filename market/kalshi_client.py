import requests
import time
import re

BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
HEADERS = {"Accept": "application/json"}
RATE_LIMIT_DELAY = 0.15


class KalshiClient:
    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_rt_events(self, include_settled=False):
        """Get all active Rotten Tomatoes events on Kalshi.

        include_settled: also pull recently-settled events. Needed by the
        settlement workflow because Kalshi removes settled markets from the
        default "open" query within an hour or so of resolution.
        """
        events = []
        seen_tickers = set()
        statuses = ["open"]
        if include_settled:
            statuses.extend(["closed", "settled"])
        for prefix in ["KXRT"]:
            for status in statuses:
                cursor = None
                while True:
                    params = {"limit": 200, "series_ticker": prefix, "status": status}
                    if cursor:
                        params["cursor"] = cursor
                    resp = self._get("/events", params)
                    if not resp:
                        break
                    batch = resp.get("events", [])
                    if not batch:
                        break
                    for e in batch:
                        title = e.get("title", "")
                        ticker = e.get("event_ticker", "")
                        if "rotten tomatoes" in title.lower() and ticker not in seen_tickers:
                            seen_tickers.add(ticker)
                            parsed = self._parse_event(e)
                            parsed["status"] = status
                            events.append(parsed)
                    cursor = resp.get("cursor")
                    if not cursor:
                        break
        return events

    def get_markets(self, event_ticker):
        """Get all threshold markets for an RT event."""
        resp = self._get("/markets", {"event_ticker": event_ticker, "limit": 50})
        if not resp:
            return []
        markets = []
        for m in resp.get("markets", []):
            parsed = self._parse_market(m)
            if parsed:
                markets.append(parsed)
        return sorted(markets, key=lambda x: x["threshold"] or 0)

    def get_settled_score(self, event_ticker):
        """Derive the canonical settlement score for an event from Kalshi's
        resolved markets. Kalshi is the source of truth -- using RT live
        risks reading a post-settlement drift.

        For an "Above T%" market, YES resolves when final > T, NO when <= T.
        The settlement score is between max(YES thresholds) and min(NO thresholds).
        Returns the score (or None if not derivable yet).
        """
        resp = self._get("/markets", {
            "event_ticker": event_ticker, "status": "settled", "limit": 50,
        })
        if not resp:
            return None
        import re
        yes_thresholds = []
        no_thresholds = []
        for m in resp.get("markets", []):
            ticker = m.get("ticker", "")
            result = m.get("result", "")
            match = re.search(r"-(\d+)$", ticker)
            if not match:
                continue
            t = int(match.group(1))
            if result == "yes":
                yes_thresholds.append(t)
            elif result == "no":
                no_thresholds.append(t)
        if not yes_thresholds or not no_thresholds:
            return None
        # Settlement score sits between max(YES) and min(NO). Pick min(NO)
        # because RT reports integer percentages and "Above 62" resolving NO
        # means score <= 62, so the integer score is min(NO).
        return min(no_thresholds)

    def _is_rt_event(self, ticker, title):
        ticker_upper = ticker.upper()
        title_lower = title.lower()
        if "KXRT" in ticker_upper or ticker_upper.startswith("RT"):
            if "rotten" in title_lower or "tomato" in title_lower or "rt" in title_lower:
                return True
        if "rotten tomatoes" in title_lower:
            return True
        return False

    def _parse_event(self, e):
        title = e.get("title", "")
        movie_name = self._extract_movie_name(title)
        return {
            "event_ticker": e.get("event_ticker", ""),
            "title": title,
            "movie_name": movie_name,
            "subtitle": e.get("sub_title", ""),
            "category": e.get("category", ""),
            "status": e.get("status"),
            "strike_period": e.get("strike_period"),
        }

    def _parse_market(self, m):
        threshold = m.get("floor_strike")
        if threshold is not None:
            threshold = int(threshold)
        else:
            subtitle = m.get("subtitle", "") or m.get("yes_sub_title", "")
            threshold = self._extract_threshold(subtitle)

        yes_price = self._parse_dollar(m.get("last_price_dollars"))
        if yes_price is None:
            bid = self._parse_dollar(m.get("yes_bid_dollars"))
            ask = self._parse_dollar(m.get("yes_ask_dollars"))
            if bid is not None and ask is not None:
                yes_price = (bid + ask) / 2
            else:
                yes_price = bid or ask
        no_price = self._parse_dollar(m.get("last_price_dollars"))
        if no_price is not None:
            no_price = 1.0 - no_price

        return {
            "ticker": m.get("ticker", ""),
            "title": m.get("subtitle", "") or m.get("title", ""),
            "threshold": threshold,
            "yes_price": yes_price,
            "no_price": no_price,
            "volume": m.get("volume_fp", m.get("volume", 0)),
            "open_interest": m.get("open_interest_fp", 0),
            "yes_bid": self._parse_dollar(m.get("yes_bid_dollars")),
            "yes_ask": self._parse_dollar(m.get("yes_ask_dollars")),
            "close_time": m.get("close_time"),
            "rules": m.get("rules_primary", ""),
            "status": m.get("status", ""),
            "result": m.get("result"),
        }

    @staticmethod
    def _parse_dollar(val):
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def _extract_movie_name(self, title):
        match = re.search(r'"([^"]+)"', title)
        if match:
            return match.group(1)
        title = re.sub(r'\s*Rotten\s+Tomatoes\s+score\??.*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s*RT\s+score\??.*', '', title, flags=re.IGNORECASE)
        return title.strip()

    def _extract_threshold(self, title):
        match = re.search(r'(?:above|over|>=?)\s*(\d+)', title, re.IGNORECASE)
        if match:
            return int(match.group(1))
        match = re.search(r'(\d+)\s*(?:or higher|or more|\+|%)', title, re.IGNORECASE)
        if match:
            return int(match.group(1))
        return None

    def get_orderbook(self, market_ticker):
        """Get the full order book for a market."""
        resp = self._get(f"/markets/{market_ticker}/orderbook")
        if not resp:
            return None
        ob = resp.get("orderbook_fp", resp.get("orderbook", {}))
        return {
            "yes_bids": [(float(p), float(s)) for p, s in ob.get("yes_dollars", [])],
            "no_bids": [(float(p), float(s)) for p, s in ob.get("no_dollars", [])],
        }

    def simulate_fill(self, market_ticker, direction, budget):
        """Simulate filling an order and return avg price + contracts filled.
        direction: 'BUY YES' or 'BUY NO'
        budget: dollars to spend
        Returns dict with avg_price, contracts, total_cost, fills list."""
        ob = self.get_orderbook(market_ticker)
        if not ob:
            return None

        if direction == "BUY YES":
            # Match against NO bids (sorted highest first = cheapest YES cost first)
            book = sorted(ob["no_bids"], reverse=True)
            cost_fn = lambda bid_price: 1.0 - bid_price
        else:
            # Match against YES bids (sorted highest first = cheapest NO cost first)
            book = sorted(ob["yes_bids"], reverse=True)
            cost_fn = lambda bid_price: 1.0 - bid_price

        remaining = float(budget)
        total_contracts = 0.0
        total_cost = 0.0
        fills = []

        for bid_price, size in book:
            if remaining <= 0.01:
                break
            cost_per = cost_fn(bid_price)
            if cost_per <= 0 or cost_per >= 1:
                continue
            can_afford = remaining / cost_per
            take = min(size, can_afford)
            cost = take * cost_per
            total_contracts += take
            total_cost += cost
            remaining -= cost
            fills.append({"price": round(cost_per * 100, 1), "contracts": round(take, 1), "cost": round(cost, 2)})

        if total_contracts == 0:
            return None

        return {
            "avg_price": round(total_cost / total_contracts * 100, 1),
            "contracts": round(total_contracts, 1),
            "total_cost": round(total_cost, 2),
            "best_price": fills[0]["price"] if fills else None,
            "fills": fills,
        }

    def _get(self, path, params=None):
        try:
            resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=15)
            if resp.status_code == 429:
                time.sleep(3)
                resp = self.session.get(f"{BASE_URL}{path}", params=params, timeout=15)
            if resp.status_code == 200:
                time.sleep(RATE_LIMIT_DELAY)
                return resp.json()
        except requests.RequestException:
            pass
        return None
