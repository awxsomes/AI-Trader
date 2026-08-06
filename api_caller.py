"""
API Caller Module — Broker Integration (Simulation-Ready)

This module provides a clean interface for paper/live trading.
Currently supports Alpaca, Interactive Brokers, and custom REST endpoints.

NOTE: This project is simulation-only. The API caller is included for
future paper trading integration, not used in the main training pipeline.
"""

import requests
import json
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, Optional, Any
import warnings
warnings.filterwarnings('ignore')


# =============================================================================
# BASE API INTERFACE
# =============================================================================
class BaseAPI(ABC):
    @abstractmethod
    def connect(self, **kwargs) -> bool:
        pass

    @abstractmethod
    def get_account(self) -> Dict:
        pass

    @abstractmethod
    def get_quote(self, symbol: str) -> Dict:
        pass

    @abstractmethod
    def place_order(self, symbol: str, qty: float, side: str, 
                    order_type: str = "market", **kwargs) -> Dict:
        pass

    @abstractmethod
    def get_positions(self) -> list:
        pass

    @abstractmethod
    def get_order_status(self, order_id: str) -> Dict:
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        pass

    @abstractmethod
    def get_bars(self, symbol: str, timeframe: str = "1Min", 
                 limit: int = 100) -> list:
        pass


# =============================================================================
# ALPACA IMPLEMENTATION
# =============================================================================
class AlpacaAPI(BaseAPI):
    """
    Alpaca Markets API implementation.
    Paper trading: https://paper-api.alpaca.markets
    Live trading: https://api.alpaca.markets
    """

    def __init__(self, paper=True):
        self.paper = paper
        self.base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.headers = {}
        self.connected = False
        self.timeout = 10

    def connect(self, api_key: Optional[str] = None,
                api_secret: Optional[str] = None) -> bool:
        api_key = api_key or os.environ.get("ALPACA_API_KEY")
        api_secret = api_secret or os.environ.get("ALPACA_API_SECRET")
        if not api_key or not api_secret:
            print("Missing credentials: set ALPACA_API_KEY and ALPACA_API_SECRET")
            return False

        self.headers = {
            "APCA-API-KEY-ID": api_key,
            "APCA-API-SECRET-KEY": api_secret
        }
        try:
            response = requests.get(
                f"{self.base_url}/v2/account", 
                headers=self.headers,
                timeout=self.timeout
            )
            if response.status_code == 200:
                self.connected = True
                print(f"Connected to Alpaca {'Paper' if self.paper else 'Live'} Trading")
                return True
            else:
                print(f"Connection failed: {response.status_code}")
                return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def _safe_json(self, response):
        """Safely parse JSON response."""
        try:
            return response.json()
        except Exception:
            return {"error": "Non-JSON response", "status_code": response.status_code, "text": response.text[:200]}

    def get_account(self) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        response = requests.get(
            f"{self.base_url}/v2/account", 
            headers=self.headers,
            timeout=self.timeout
        )
        return self._safe_json(response)

    def get_quote(self, symbol: str) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        response = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/quotes/latest",
            headers=self.headers,
            timeout=self.timeout
        )
        return self._safe_json(response)

    def place_order(self, symbol: str, qty: float, side: str,
                    order_type: str = "market", time_in_force: str = "day",
                    **kwargs) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")

        payload = {
            "symbol": symbol,
            "qty": str(qty),
            "side": side.lower(),
            "type": order_type,
            "time_in_force": time_in_force
        }
        if order_type == "limit" and "limit_price" in kwargs:
            payload["limit_price"] = str(kwargs["limit_price"])
        if "stop_price" in kwargs:
            payload["stop_price"] = str(kwargs["stop_price"])

        response = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=payload,
            timeout=self.timeout
        )
        return self._safe_json(response)

    def get_positions(self) -> list:
        if not self.connected:
            raise ConnectionError("Not connected")
        response = requests.get(
            f"{self.base_url}/v2/positions", 
            headers=self.headers,
            timeout=self.timeout
        )
        data = self._safe_json(response)
        return data if isinstance(data, list) else []

    def get_order_status(self, order_id: str) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        response = requests.get(
            f"{self.base_url}/v2/orders/{order_id}",
            headers=self.headers,
            timeout=self.timeout
        )
        return self._safe_json(response)

    def cancel_order(self, order_id: str) -> bool:
        """FIX: Alpaca cancel returns 204, not 200."""
        if not self.connected:
            raise ConnectionError("Not connected")
        response = requests.delete(
            f"{self.base_url}/v2/orders/{order_id}",
            headers=self.headers,
            timeout=self.timeout
        )
        return response.status_code in [200, 204]

    def get_bars(self, symbol: str, timeframe: str = "1Min", 
             limit: int = 100) -> list:
        if not self.connected:
            raise ConnectionError("Not connected")
        response = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/bars",
            headers=self.headers,
            # sort=desc is required: with no explicit `start`, Alpaca defaults
            # start to the beginning of the current trading day and returns
            # ascending results, so `limit` alone silently caps you at the
            # OLDEST N bars of the day instead of the most recent N — the
            # response looks fine (real data, right shape) but never advances.
            params={"timeframe": timeframe, "limit": limit, "feed": "iex", "sort": "desc"},
            timeout=self.timeout
        )
        data = self._safe_json(response)

        # Don't silently swallow API errors
        if "bars" in data:
            # API returns newest-first when sort=desc; callers expect
            # chronological (oldest-first) order.
            return list(reversed(data["bars"]))
        print(f"[API ERROR] {data}")
        return []


# =============================================================================
# INTERACTIVE BROKERS
# =============================================================================
class IBKRAPI(BaseAPI):
    def __init__(self):
        self.ib = None
        self.connected = False

    def connect(self, host: str = "127.0.0.1", port: int = 7497, 
                client_id: int = 1, **kwargs) -> bool:
        try:
            from ib_insync import IB, Stock
            self.ib = IB()
            self.ib.connect(host, port, clientId=client_id)
            self.connected = self.ib.isConnected()
            if self.connected:
                print(f"Connected to IBKR at {host}:{port}")
            return self.connected
        except ImportError:
            print("ib_insync not installed. Run: pip install ib_insync")
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def get_account(self) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        account_values = self.ib.accountValues()
        return {av.tag: av.value for av in account_values}

    def get_quote(self, symbol: str) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        from ib_insync import Stock
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        ticker = self.ib.reqMktData(contract)
        self.ib.sleep(1)
        return {
            "bid": ticker.bid, "ask": ticker.ask,
            "last": ticker.last, "volume": ticker.volume
        }

    def place_order(self, symbol: str, qty: float, side: str,
                    order_type: str = "MKT", **kwargs) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        from ib_insync import Stock, MarketOrder, LimitOrder
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        action = "BUY" if side.lower() in ["buy", "long"] else "SELL"
        if order_type.upper() == "MKT":
            order = MarketOrder(action, qty)
        elif order_type.upper() == "LMT":
            order = LimitOrder(action, qty, kwargs.get("limit_price", 0))
        else:
            order = MarketOrder(action, qty)
        trade = self.ib.placeOrder(contract, order)
        return {"order_id": trade.order.orderId, "status": trade.orderStatus.status}

    def get_positions(self) -> list:
        if not self.connected:
            raise ConnectionError("Not connected")
        positions = self.ib.positions()
        return [
            {"symbol": p.contract.symbol, "qty": p.position, "avg_cost": p.avgCost}
            for p in positions
        ]

    def get_order_status(self, order_id: str) -> Dict:
        if not self.connected:
            raise ConnectionError("Not connected")
        trades = self.ib.trades()
        for trade in trades:
            if str(trade.order.orderId) == order_id:
                return {"status": trade.orderStatus.status}
        return {"status": "not_found"}

    def cancel_order(self, order_id: str) -> bool:
        if not self.connected:
            raise ConnectionError("Not connected")
        trades = self.ib.trades()
        for trade in trades:
            if str(trade.order.orderId) == order_id:
                self.ib.cancelOrder(trade.order)
                return True
        return False

    def get_bars(self, symbol: str, timeframe: str = "1 min", 
                 limit: int = 100) -> list:
        if not self.connected:
            raise ConnectionError("Not connected")
        from ib_insync import Stock
        contract = Stock(symbol, "SMART", "USD")
        self.ib.qualifyContracts(contract)
        duration_str = f"{limit} D" if limit > 1 else "1 D"
        bar_size = "1 min" if timeframe in ["1Min", "1m"] else timeframe
        bars = self.ib.reqHistoricalData(
            contract, endDateTime="", durationStr=duration_str,
            barSizeSetting=bar_size, whatToShow="TRADES", useRTH=True
        )
        return [
            {"date": bar.date.isoformat(), "open": bar.open, "high": bar.high,
             "low": bar.low, "close": bar.close, "volume": bar.volume}
            for bar in bars
        ]


# =============================================================================
# CUSTOM API TEMPLATE
# =============================================================================
class CustomAPI(BaseAPI):
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.headers = {}
        self.connected = False
        self.timeout = 10

    def connect(self, api_key: str = None, **kwargs) -> bool:
        if api_key:
            self.headers["Authorization"] = f"Bearer {api_key}"
        self.connected = True
        return True

    def get_account(self) -> Dict:
        raise NotImplementedError("Implement your account endpoint")

    def get_quote(self, symbol: str) -> Dict:
        raise NotImplementedError("Implement your quote endpoint")

    def place_order(self, symbol: str, qty: float, side: str,
                    order_type: str = "market", **kwargs) -> Dict:
        raise NotImplementedError("Implement your order endpoint")

    def get_positions(self) -> list:
        raise NotImplementedError("Implement your positions endpoint")

    def get_order_status(self, order_id: str) -> Dict:
        raise NotImplementedError("Implement your order status endpoint")

    def cancel_order(self, order_id: str) -> bool:
        raise NotImplementedError("Implement your cancel endpoint")

    def get_bars(self, symbol: str, timeframe: str = "1Min",
                 limit: int = 100) -> list:
        raise NotImplementedError("Implement your bars endpoint")


# =============================================================================
# WEBHOOK NOTIFIER
# =============================================================================
class WebhookNotifier:
    def __init__(self, webhook_url: str = None):
        self.webhook_url = webhook_url

    def set_webhook(self, url: str):
        self.webhook_url = url

    def send_signal(self, signal: Dict) -> bool:
        if not self.webhook_url:
            return False
        payload = {"timestamp": datetime.now().isoformat(), "signal": signal}
        try:
            response = requests.post(
                self.webhook_url, json=payload,
                headers={"Content-Type": "application/json"},
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            print(f"Webhook error: {e}")
            return False

    def send_discord(self, message: str, webhook_url: str = None) -> bool:
        url = webhook_url or self.webhook_url
        if not url:
            return False
        try:
            response = requests.post(url, json={"content": message}, timeout=5)
            return response.status_code == 204
        except Exception as e:
            print(f"Discord webhook error: {e}")
            return False


# =============================================================================
# MAIN API CALLER CLASS
# =============================================================================
class APICaller:
    PROVIDERS = {
        "alpaca": AlpacaAPI,
        "ibkr": IBKRAPI,
        "custom": CustomAPI
    }

    def __init__(self, provider: str = "alpaca", **kwargs):
        if provider not in self.PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}. Available: {list(self.PROVIDERS.keys())}")
        self.provider_name = provider
        self.api = self.PROVIDERS[provider](**kwargs)
        self.webhook = WebhookNotifier()

    def connect(self, **kwargs) -> bool:
        return self.api.connect(**kwargs)

    def set_webhook(self, url: str):
        self.webhook.set_webhook(url)

    def get_account(self) -> Dict:
        return self.api.get_account()

    def get_quote(self, symbol: str) -> Dict:
        return self.api.get_quote(symbol)

    def place_order(self, symbol: str, qty: float, side: str, 
                    order_type: str = "market", notify: bool = True,
                    **kwargs) -> Dict:
        result = self.api.place_order(symbol, qty, side, order_type, **kwargs)
        if notify and self.webhook.webhook_url:
            self.webhook.send_signal({
                "action": side, "symbol": symbol, "quantity": qty,
                "order_type": order_type, "result": result
            })
        return result

    def get_positions(self) -> list:
        return self.api.get_positions()

    def get_order_status(self, order_id: str) -> Dict:
        return self.api.get_order_status(order_id)

    def cancel_order(self, order_id: str) -> bool:
        return self.api.cancel_order(order_id)

    def get_bars(self, symbol: str, timeframe: str = "1Min", 
                 limit: int = 100) -> list:
        return self.api.get_bars(symbol, timeframe, limit)

    def is_connected(self) -> bool:
        return getattr(self.api, "connected", False)


if __name__ == "__main__":
    print("API Caller Module")
    print("="*60)
    print("Available providers:", list(APICaller.PROVIDERS.keys()))
    print("\nThis module is for future paper trading integration.")
    print("The main simulation pipeline does not require API connectivity.")