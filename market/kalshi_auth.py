"""
Authenticated Kalshi API client for order placement and portfolio management.

Uses RSA-PSS signing with SHA256 per Kalshi's V2 API spec.
Requires an API key (created at kalshi.com/account/profile) and the
corresponding RSA private key in PEM format.

Credentials are loaded from ~/.config/kalshi-rt/credentials.json:
{
    "api_key_id": "your-key-id",
    "private_key_path": "/path/to/kalshi-key.pem"
}

For demo/paper trading, set "demo": true in credentials.json.
"""
import base64
import json
import time
import uuid
from pathlib import Path
from urllib.parse import urlparse

import requests
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

import config

PROD_BASE = "https://external-api.kalshi.com/trade-api/v2"
DEMO_BASE = "https://external-api.demo.kalshi.co/trade-api/v2"
CREDENTIALS_PATH = Path.home() / ".config" / "kalshi-rt" / "credentials.json"


class KalshiAuthClient:
    """Authenticated Kalshi client for trading operations."""

    def __init__(self, demo=None):
        creds = self._load_credentials()
        self._api_key = creds["api_key_id"]
        self._private_key = self._load_private_key(creds["private_key_path"])

        if demo is None:
            demo = creds.get("demo", False)
        self._base_url = DEMO_BASE if demo else PROD_BASE
        self._demo = demo

        self._session = requests.Session()
        self._session.headers.update({"Accept": "application/json"})

    @property
    def is_demo(self):
        return self._demo

    # --- Account ---

    def get_balance(self):
        """Get account balance in dollars."""
        resp = self._auth_get("/portfolio/balance")
        if not resp:
            return None
        balance_cents = resp.get("balance", 0)
        return {
            "balance": balance_cents / 100.0,
            "payout_available": resp.get("payout_available", 0) / 100.0,
        }

    def get_positions(self, event_ticker=None):
        """Get current positions. Optionally filter by event ticker."""
        params = {"limit": 200}
        if event_ticker:
            params["event_ticker"] = event_ticker
        resp = self._auth_get("/portfolio/positions", params=params)
        if not resp:
            return []
        positions = []
        for p in resp.get("market_positions", []):
            positions.append({
                "ticker": p.get("ticker", ""),
                "yes_count": float(p.get("yes_count", 0)),
                "no_count": float(p.get("no_count", 0)),
                "yes_avg_price": float(p.get("yes_average_price_paid_cents", 0)) / 100.0,
                "no_avg_price": float(p.get("no_average_price_paid_cents", 0)) / 100.0,
                "realized_pnl": float(p.get("realized_pnl_cents", 0)) / 100.0,
            })
        return positions

    # --- Orders ---

    def place_order(self, ticker, side, count, price, time_in_force="good_till_canceled",
                    client_order_id=None):
        """Place an order on Kalshi.

        Args:
            ticker: Market ticker (e.g. "KXRT-POWERBALLAD-T80")
            side: "bid" (buy) or "ask" (sell)
            count: Number of contracts (float, up to 2 decimals)
            price: Price per contract in dollars (float, up to 6 decimals)
            time_in_force: "fill_or_kill", "good_till_canceled", or "immediate_or_cancel"
            client_order_id: Unique ID for idempotency (auto-generated if None)

        Returns dict with order_id, fill_count, remaining_count, avg fill price.
        """
        if client_order_id is None:
            client_order_id = str(uuid.uuid4())

        body = {
            "ticker": ticker,
            "client_order_id": client_order_id,
            "side": side,
            "count": f"{count:.2f}",
            "price": f"{price:.6f}",
            "time_in_force": time_in_force,
            "self_trade_prevention_type": "taker_at_cross",
        }

        resp = self._auth_post("/portfolio/events/orders", body)
        if not resp:
            return None

        return {
            "order_id": resp.get("order_id"),
            "client_order_id": resp.get("client_order_id"),
            "fill_count": float(resp.get("fill_count", "0")),
            "remaining_count": float(resp.get("remaining_count", "0")),
            "avg_fill_price": float(resp.get("average_fill_price", "0")) if resp.get("average_fill_price") else None,
            "avg_fee_paid": float(resp.get("average_fee_paid", "0")) if resp.get("average_fee_paid") else None,
        }

    def buy_yes(self, ticker, count, price, time_in_force="immediate_or_cancel"):
        """Buy YES contracts at a limit price."""
        return self.place_order(ticker, "bid", count, price, time_in_force)

    def buy_no(self, ticker, count, price, time_in_force="immediate_or_cancel"):
        """Buy NO contracts (sell YES) at a limit price."""
        return self.place_order(ticker, "ask", count, 1.0 - price, time_in_force)

    def get_order(self, order_id):
        """Check status of a specific order."""
        return self._auth_get(f"/portfolio/orders/{order_id}")

    def get_orders(self, ticker=None, status=None):
        """List orders. Optionally filter by ticker or status."""
        params = {"limit": 200}
        if ticker:
            params["ticker"] = ticker
        if status:
            params["status"] = status
        resp = self._auth_get("/portfolio/orders", params=params)
        if not resp:
            return []
        return resp.get("orders", [])

    def cancel_order(self, order_id):
        """Cancel an open order."""
        return self._auth_delete(f"/portfolio/orders/{order_id}")

    # --- Execution helpers ---

    def execute_signal(self, ticker, direction, spend, model_prob, dry_run=True):
        """Execute a trade signal from the alpha engine.

        Walks the order book and places an immediate-or-cancel order at the
        best available price up to the spend limit.

        Args:
            ticker: Market ticker
            direction: "BUY YES" or "BUY NO"
            spend: Maximum dollars to spend
            model_prob: Model's win probability (for logging)
            dry_run: If True, simulate only (no real order)

        Returns execution result dict.
        """
        from market.kalshi_client import KalshiClient
        public = KalshiClient()
        ob = public.get_orderbook(ticker)
        if not ob:
            return {"status": "error", "message": "Could not fetch orderbook"}

        if direction == "BUY YES":
            book = sorted(ob["no_bids"], reverse=True)
        else:
            book = sorted(ob["yes_bids"], reverse=True)

        if not book:
            return {"status": "error", "message": "Empty orderbook"}

        best_price = 1.0 - book[0][0]
        win_prob = model_prob if direction == "BUY YES" else 1.0 - model_prob

        if best_price >= win_prob:
            return {"status": "skip", "message": f"Best price {best_price:.2f} >= model prob {win_prob:.2f}"}

        count = int(spend / best_price)
        if count < 1:
            return {"status": "skip", "message": f"Can't afford even 1 contract at {best_price:.2f}"}

        result = {
            "ticker": ticker,
            "direction": direction,
            "price": round(best_price, 4),
            "count": count,
            "spend": round(count * best_price, 2),
            "model_prob": round(model_prob, 4),
            "dry_run": dry_run,
        }

        if dry_run:
            result["status"] = "simulated"
            return result

        if direction == "BUY YES":
            order = self.buy_yes(ticker, count, best_price)
        else:
            order = self.buy_no(ticker, count, best_price)

        if order:
            result["status"] = "filled" if order["fill_count"] > 0 else "placed"
            result["order_id"] = order["order_id"]
            result["fill_count"] = order["fill_count"]
            result["remaining_count"] = order["remaining_count"]
            result["avg_fill_price"] = order["avg_fill_price"]
        else:
            result["status"] = "error"
            result["message"] = "Order placement failed"

        return result

    # --- Auth internals ---

    def _sign(self, timestamp_ms, method, path):
        path_clean = path.split("?")[0]
        full_path = urlparse(self._base_url + path_clean).path
        message = f"{timestamp_ms}{method}{full_path}".encode("utf-8")
        signature = self._private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return base64.b64encode(signature).decode("utf-8")

    def _auth_headers(self, method, path):
        ts = str(int(time.time() * 1000))
        sig = self._sign(ts, method, path)
        return {
            "KALSHI-ACCESS-KEY": self._api_key,
            "KALSHI-ACCESS-TIMESTAMP": ts,
            "KALSHI-ACCESS-SIGNATURE": sig,
        }

    def _auth_get(self, path, params=None):
        headers = self._auth_headers("GET", path)
        try:
            resp = self._session.get(
                f"{self._base_url}{path}", params=params, headers=headers, timeout=15,
            )
            if resp.status_code == 429:
                time.sleep(3)
                headers = self._auth_headers("GET", path)
                resp = self._session.get(
                    f"{self._base_url}{path}", params=params, headers=headers, timeout=15,
                )
            if resp.status_code == 200:
                return resp.json()
            else:
                print(f"Kalshi API error {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            print(f"Kalshi request failed: {e}")
        return None

    def _auth_post(self, path, body):
        headers = self._auth_headers("POST", path)
        headers["Content-Type"] = "application/json"
        try:
            resp = self._session.post(
                f"{self._base_url}{path}", json=body, headers=headers, timeout=15,
            )
            if resp.status_code in (200, 201):
                return resp.json()
            elif resp.status_code == 409:
                print(f"Kalshi: duplicate order (client_order_id already used)")
                return resp.json()
            else:
                print(f"Kalshi API error {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            print(f"Kalshi request failed: {e}")
        return None

    def _auth_delete(self, path):
        headers = self._auth_headers("DELETE", path)
        try:
            resp = self._session.delete(
                f"{self._base_url}{path}", headers=headers, timeout=15,
            )
            if resp.status_code in (200, 204):
                return True
            else:
                print(f"Kalshi API error {resp.status_code}: {resp.text[:200]}")
        except requests.RequestException as e:
            print(f"Kalshi request failed: {e}")
        return False

    @staticmethod
    def _load_credentials():
        if not CREDENTIALS_PATH.exists():
            raise FileNotFoundError(
                f"Kalshi credentials not found at {CREDENTIALS_PATH}\n"
                f"Create this file with:\n"
                f'{{\n'
                f'  "api_key_id": "your-key-id",\n'
                f'  "private_key_path": "/path/to/kalshi-key.pem",\n'
                f'  "demo": true\n'
                f'}}'
            )
        with open(CREDENTIALS_PATH) as f:
            return json.load(f)

    @staticmethod
    def _load_private_key(path):
        with open(path, "rb") as f:
            return serialization.load_pem_private_key(
                f.read(), password=None, backend=default_backend(),
            )
