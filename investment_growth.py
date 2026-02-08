#!/usr/bin/env python3
"""
Daily growth calculator for the "Investments Spreadsheet.csv" Google Sheet export.

The script supports two account types:
* Savings accounts leverage the APY column to estimate daily interest.
* Investment accounts rely on the Holdings column to identify tickers, fetch
  market data, and estimate the daily change in account value.

Usage:
    python investment_growth.py \
        --sheet "Investments Spreadsheet.csv" \
        --cache price_cache.json \
        [--no-network]

When network access is unavailable, seed the cache file with the most recent
quotes and rerun with --no-network so the script continues to work offline.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import logging
import math
import re
import sys
import warnings
from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP, getcontext
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple
from urllib.error import URLError
from urllib.request import Request, urlopen

try:
    import yfinance as yf
except ImportError:  # pragma: no cover - optional dependency
    yf = None

try:
    from pandas.errors import Pandas4Warning  # type: ignore[attr-defined]
except Exception:  # pragma: no cover - pandas import optional
    class Pandas4Warning(Warning):
        pass

warnings.filterwarnings("ignore", category=Pandas4Warning)

getcontext().prec = 28
TWO_PLACES = Decimal("0.01")
DEFAULT_COLUMNS = [
    "Account Name",
    "Category",
    "Institution",
    "Balance",
    "Holdings",
    "APY",
    "Shares",
    "Share Price",
]


def utc_now_iso() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute daily growth figures from an investments spreadsheet."
    )
    parser.add_argument(
        "--sheet",
        default="Investments Spreadsheet.csv",
        help="Path to the CSV downloaded from Google Sheets.",
    )
    parser.add_argument(
        "--cache",
        default="price_cache.json",
        help="Path to a JSON file used to cache market quotes.",
    )
    parser.add_argument(
        "--balance-cache",
        default="investment_balance_cache.json",
        help="Path to a JSON file storing prior-day balances for investments.",
    )
    parser.add_argument(
        "--no-network",
        action="store_true",
        help="Do not attempt to refresh quotes from the network.",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable verbose logging to trace network requests.",
    )
    return parser.parse_args()


def load_sheet(path: Path) -> Tuple[List[Dict[str, str]], List[str]]:
    if not path.exists():
        return [], list(DEFAULT_COLUMNS)

    with path.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        fieldnames = reader.fieldnames or list(DEFAULT_COLUMNS)
        rows: List[Dict[str, str]] = []
        for row in reader:
            cleaned = {key: (value or "") for key, value in row.items()}
            rows.append(cleaned)
    return rows, fieldnames


def save_sheet(path: Path, rows: List[Dict[str, str]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def coerce_cell(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    if isinstance(value, (Decimal, int, float)):
        return str(value)
    return str(value)


def rows_from_dataframe(df, columns: List[str]) -> List[Dict[str, str]]:
    rows: List[Dict[str, str]] = []
    for _, record in df.iterrows():
        rows.append({col: coerce_cell(record.get(col, "")) for col in columns})
    return rows


def parse_currency(raw: str) -> Decimal:
    if not raw:
        return Decimal("0")
    cleaned = raw.replace("$", "").replace(",", "").strip()
    if not cleaned:
        return Decimal("0")
    return Decimal(cleaned)


def parse_percent(raw: str) -> Optional[Decimal]:
    if not raw:
        return None
    cleaned = raw.strip().replace("%", "")
    if not cleaned:
        return None
    return Decimal(cleaned) / Decimal("100")


def parse_holdings(raw: str) -> List[Tuple[str, Optional[Decimal]]]:
    """
    Parse holdings as a list of (symbol, quantity) tuples.
    Supported formats per token: "VOO", "VOO:10.5", "ETH 2", "VOO=1".
    Tokens are separated by semicolons, vertical bars, or plus signs.
    """
    if not raw:
        return []
    tokens = re.split(r"[;|+]+", raw)
    holdings: List[Tuple[str, Optional[Decimal]]] = []
    for token in tokens:
        token = token.strip()
        if not token:
            continue
        quantity: Optional[Decimal] = None
        symbol = token
        for splitter in (":", "=", " "):
            if splitter in token:
                left, right = token.split(splitter, 1)
                symbol = left.strip()
                right = right.strip()
                if right:
                    quantity = Decimal(right)
                break
        holdings.append((symbol.upper(), quantity))
    return holdings


def parse_shares(raw: str) -> Tuple[Dict[str, Decimal], Optional[Decimal]]:
    """
    Parse the Shares column. Supports either:
    * Symbol-qualified values (e.g., "VOO:3;AAPL:1.5")
    * A single numeric value (used when there is only one holding)
    """
    share_map: Dict[str, Decimal] = {}
    default_quantity: Optional[Decimal] = None
    if not raw:
        return share_map, default_quantity

    raw = raw.strip()
    if not raw:
        return share_map, default_quantity

    if re.search(r"[A-Za-z]", raw):
        for symbol, quantity in parse_holdings(raw):
            if quantity is None:
                continue
            share_map[symbol] = quantity
    else:
        default_quantity = Decimal(raw.replace(",", ""))
    return share_map, default_quantity


def parse_share_prices(raw: str) -> Tuple[Dict[str, Decimal], Optional[Decimal]]:
    """
    Parse the Share Price column. Supports either:
    * Symbol-qualified values (e.g., "VOO:520;ETH-USD:2400")
    * A single numeric value (used when there is only one holding)
    """
    price_map: Dict[str, Decimal] = {}
    default_price: Optional[Decimal] = None
    if not raw:
        return price_map, default_price

    raw = raw.strip()
    if not raw:
        return price_map, default_price

    if re.search(r"[A-Za-z]", raw):
        tokens = re.split(r"[;|+]+", raw)
        for token in tokens:
            token = token.strip()
            if not token:
                continue
            symbol = ""
            value = ""
            for splitter in (":", "=", " "):
                if splitter in token:
                    left, right = token.split(splitter, 1)
                    symbol = left.strip().upper()
                    value = right.strip()
                    break
            if not symbol or not value:
                continue
            cleaned = value.replace("$", "").replace(",", "")
            price_map[symbol] = Decimal(cleaned)
    else:
        cleaned = raw.replace("$", "").replace(",", "")
        default_price = Decimal(cleaned)
    return price_map, default_price


@dataclass
class Quote:
    symbol: str
    currency: str
    current: Decimal
    previous_close: Decimal
    timestamp: str
    source: str


class BalanceCache:
    def __init__(self, path: Path):
        self.path = path
        self.logger = logging.getLogger(f"{__name__}.BalanceCache")
        self.data = self._load()

    def _load(self) -> Dict[str, Dict[str, object]]:
        if not self.path.exists():
            self.logger.debug("Balance cache %s missing; starting fresh", self.path)
            return {}
        try:
            with self.path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
        except json.JSONDecodeError as exc:
            self.logger.warning("Unable to parse balance cache %s: %s", self.path, exc)
            return {}
        if not isinstance(data, dict):
            self.logger.warning("Balance cache %s not a dict; resetting", self.path)
            return {}
        return data

    def get_positions(
        self, account_key: str
    ) -> Optional[Dict[str, Tuple[Decimal, Decimal]]]:
        entry = self.data.get(account_key)
        if not entry or not isinstance(entry, dict):
            return None

        positions = entry.get("positions")
        if isinstance(positions, dict):
            parsed: Dict[str, Tuple[Decimal, Decimal]] = {}
            for symbol, info in positions.items():
                if not isinstance(info, dict):
                    continue
                shares = info.get("shares")
                price = info.get("price")
                if shares is None or price is None:
                    continue
                parsed[symbol] = (Decimal(str(shares)), Decimal(str(price)))
            return parsed or None

        legacy_balance = entry.get("balance")
        if legacy_balance is not None:
            return {"__LEGACY__": (Decimal("1"), Decimal(str(legacy_balance)))}
        return None

    def set_positions(
        self, account_key: str, positions: Dict[str, Tuple[Decimal, Decimal]]
    ) -> None:
        serialized = {}
        for symbol, (shares, price) in positions.items():
            serialized[symbol] = {"shares": float(shares), "price": float(price)}
        self.data[account_key] = {
            "positions": serialized,
            "timestamp": utc_now_iso(),
        }

    def persist(self) -> None:
        self.logger.debug("Persisting balance cache with %d entries", len(self.data))
        self.path.write_text(json.dumps(self.data, indent=2, sort_keys=True), encoding="utf-8")


class PriceFetcher:
    def __init__(self, cache_path: Path, allow_network: bool = True):
        self.cache_path = cache_path
        self.allow_network = allow_network
        self.logger = logging.getLogger(f"{__name__}.PriceFetcher")
        self.cache: Dict[str, Dict[str, object]] = self._load_cache()
        if self.allow_network and yf is None:
            self.logger.warning(
                "yfinance is not installed; install it with 'pip install yfinance' "
                "for more reliable market data."
            )

    def _load_cache(self) -> Dict[str, Dict[str, object]]:
        if not self.cache_path.exists():
            self.logger.debug("Cache file %s missing; starting fresh", self.cache_path)
            return {}
        try:
            with self.cache_path.open("r", encoding="utf-8") as fh:
                return json.load(fh)
        except json.JSONDecodeError as exc:
            self.logger.warning("Unable to parse cache %s: %s", self.cache_path, exc)
            return {}

    def _persist_cache(self) -> None:
        self.logger.debug("Persisting cache with %d symbols", len(self.cache))
        self.cache_path.write_text(
            json.dumps(self.cache, indent=2, sort_keys=True), encoding="utf-8"
        )

    def _cache_aliases(self, symbol: str) -> List[str]:
        symbol = symbol.upper().strip()
        aliases = {symbol}
        if "-" in symbol:
            base, quote = symbol.split("-", 1)
            aliases.add(base)
            aliases.add(f"{base}{quote}")
        elif symbol.endswith("USD"):
            base = symbol[:-3]
            if base:
                aliases.add(base)
                aliases.add(f"{base}-USD")
        else:
            aliases.add(f"{symbol}-USD")
        return [alias for alias in aliases if alias]

    def get_quote(self, symbol: str) -> Quote:
        symbol = symbol.upper().strip()
        quote_data: Optional[Dict[str, object]] = None

        if self.allow_network:
            self.logger.debug("Attempting live quote download for %s", symbol)
            quote_data = self._download_live_quote(symbol)
            if quote_data:
                for alias in self._cache_aliases(symbol):
                    alias_data = dict(quote_data)
                    alias_data["symbol"] = alias
                    self.cache[alias] = alias_data
                self._persist_cache()
            else:
                self.logger.warning(
                    "Live quote request failed for %s; falling back to cache", symbol
                )
        else:
            self.logger.debug("Networking disabled; using cache for %s", symbol)

        if not quote_data:
            for alias in self._cache_aliases(symbol):
                if alias in self.cache:
                    quote_data = dict(self.cache[alias])
                    quote_data["symbol"] = symbol
                    break

        if not quote_data:
            self.logger.error(
                "No price data available for %s (network=%s, cache_keys=%s)",
                symbol,
                self.allow_network,
                list(self.cache.keys()),
            )
            raise RuntimeError(
                f"Missing market data for {symbol}. Refresh cache or enable networking."
            )

        return Quote(
            symbol=quote_data.get("symbol", symbol),
            currency=quote_data.get("currency", "USD"),
            current=Decimal(str(quote_data["current"])),
            previous_close=Decimal(str(quote_data["previous_close"])),
            timestamp=quote_data.get("timestamp", "unknown"),
            source=quote_data.get("source", "cache"),
        )

    def _download_live_quote(self, symbol: str) -> Optional[Dict[str, object]]:
        """
        Try providers in priority order until one returns data.
        """
        providers = [("yfinance", self._download_quote_yfinance)]
        if self._looks_like_crypto(symbol):
            providers.append(("coinbase", self._download_quote_coinbase))
        providers.append(("yahoo-http", self._download_quote_yahoo))
        for name, provider in providers:
            data = provider(symbol)
            if data:
                self.logger.debug("Fetched quote for %s via %s", symbol, name)
                return data
            self.logger.debug("Provider %s failed for %s", name, symbol)
        return None

    def _looks_like_crypto(self, symbol: str) -> bool:
        symbol = symbol.upper()
        return "-" in symbol or symbol.endswith("USD") or symbol.endswith("USDT")

    def _split_crypto_pair(self, symbol: str) -> Tuple[str, str]:
        symbol = symbol.upper()
        if "-" in symbol:
            base, quote = symbol.split("-", 1)
        elif symbol.endswith("USD"):
            base = symbol[:-3]
            quote = "USD"
        elif symbol.endswith("USDT"):
            base = symbol[:-4]
            quote = "USDT"
        else:
            base = symbol
            quote = "USD"
        return base or symbol, quote or "USD"

    def _download_quote_yfinance(self, symbol: str) -> Optional[Dict[str, object]]:
        """
        Fetch quote data using the yfinance library.
        """
        if yf is None:
            self.logger.debug("yfinance unavailable; skipping")
            return None

        try:
            ticker = yf.Ticker(symbol)
        except Exception as exc:  # pragma: no cover - yfinance internals
            self.logger.warning("Unable to init yfinance for %s: %s", symbol, exc)
            return None

        fast_info = None
        try:
            fast_info = getattr(ticker, "fast_info", None)
        except Exception as exc:
            self.logger.warning("yfinance fast_info error for %s: %s", symbol, exc)

        if fast_info and not isinstance(fast_info, dict):
            try:
                fast_info = dict(fast_info)
            except Exception:  # pragma: no cover - depends on yfinance internals
                fast_info = None

        if fast_info:
            quote = self._quote_from_fast_info(symbol, fast_info)
            if quote:
                return quote

        try:
            history = ticker.history(period="2d", interval="1d", auto_adjust=False)
        except Exception as exc:
            self.logger.warning("yfinance history error for %s: %s", symbol, exc)
            return None

        if history is None or history.empty:
            self.logger.warning("yfinance returned empty history for %s", symbol)
            return None

        try:
            close_series = history["Close"]
        except KeyError:
            self.logger.warning("yfinance history missing Close column for %s", symbol)
            return None

        last_close = float(close_series.iloc[-1])
        prev_close = (
            float(close_series.iloc[-2])
            if len(close_series.index) > 1
            else float(close_series.iloc[-1])
        )
        ts_index = history.index[-1]
        if hasattr(ts_index, "to_pydatetime"):
            timestamp = ts_index.to_pydatetime().isoformat()
        else:
            timestamp = str(ts_index)

        currency = "USD"
        if isinstance(fast_info, dict):
            currency = (
                fast_info.get("currency")
                or fast_info.get("currencyCode")
                or fast_info.get("currency_code")
                or currency
            )

        return {
            "symbol": symbol,
            "currency": currency,
            "current": last_close,
            "previous_close": prev_close,
            "timestamp": timestamp,
            "source": "yfinance-history",
        }

    def _quote_from_fast_info(self, symbol: str, fast_info: Dict[str, object]) -> Optional[Dict[str, object]]:
        def first_value(data: Dict[str, object], keys: List[str]) -> Optional[float]:
            for key in keys:
                if key in data and data[key] is not None:
                    try:
                        return float(data[key])
                    except (TypeError, ValueError):
                        continue
            return None

        price = first_value(
            fast_info,
            [
                "last_price",
                "lastPrice",
                "regularMarketPrice",
                "regular_market_price",
                "lastSalePrice",
            ],
        )
        prev_close = first_value(
            fast_info,
            [
                "previous_close",
                "previousClose",
                "regularMarketPreviousClose",
                "regular_market_previous_close",
                "last_close",
                "lastClose",
            ],
        )

        if price is None or prev_close is None:
            return None

        timestamp = fast_info.get("last_trade_time") or fast_info.get("lastTradeTime")
        if not timestamp:
            timestamp = utc_now_iso()

        currency = (
            fast_info.get("currency")
            or fast_info.get("currencyCode")
            or fast_info.get("currency_code")
            or "USD"
        )

        return {
            "symbol": symbol,
            "currency": currency,
            "current": price,
            "previous_close": prev_close,
            "timestamp": timestamp,
            "source": "yfinance-fast",
        }

    def _download_quote_coinbase(self, symbol: str) -> Optional[Dict[str, object]]:
        base, quote = self._split_crypto_pair(symbol)
        pair = f"{base}-{quote}"
        spot_url = f"https://api.coinbase.com/v2/prices/{pair}/spot"
        stats_url = f"https://api.exchange.coinbase.com/products/{pair}/stats"

        try:
            with urlopen(spot_url, timeout=10) as response:
                spot_payload = json.load(response)
            amount = spot_payload["data"]["amount"]
            current_price = float(amount)
        except (KeyError, ValueError, URLError, TimeoutError, json.JSONDecodeError) as exc:
            self.logger.warning("Coinbase spot error for %s: %s", symbol, exc)
            return None

        prev_close = current_price
        try:
            request = Request(
                stats_url,
                headers={"User-Agent": "investment-growth-bot", "Accept": "application/json"},
            )
            with urlopen(request, timeout=10) as response:
                stats_payload = json.load(response)
            prev_candidate = stats_payload.get("open") or stats_payload.get("last")
            if prev_candidate is not None:
                prev_close = float(prev_candidate)
        except (URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            self.logger.warning(
                "Coinbase stats error for %s: %s; using spot price as previous close",
                symbol,
                exc,
            )

        timestamp = utc_now_iso()
        return {
            "symbol": f"{base}-{quote}",
            "currency": quote,
            "current": current_price,
            "previous_close": prev_close,
            "timestamp": timestamp,
            "source": "coinbase",
        }

    def _download_quote_yahoo(self, symbol: str) -> Optional[Dict[str, object]]:
        """
        Fetch quote data from Yahoo Finance. Returns None if unavailable.
        """
        url = (
            "https://query1.finance.yahoo.com/v7/finance/quote?symbols="
            + symbol.replace("/", "-")
        )
        try:
            with urlopen(url, timeout=10) as response:
                payload = json.load(response)
        except (URLError, TimeoutError, ValueError) as exc:
            self.logger.warning("Network error requesting %s: %s", symbol, exc)
            return None

        results = payload.get("quoteResponse", {}).get("result", [])
        if not results:
            self.logger.warning("Quote API returned no result for %s", symbol)
            return None

        info = results[0]
        price = info.get("regularMarketPrice")
        prev_close = info.get("regularMarketPreviousClose")
        if price is None or prev_close is None:
            self.logger.warning("Quote for %s missing pricing data: %s", symbol, info)
            return None

        timestamp = info.get("regularMarketTime")
        if timestamp:
            ts_iso = dt.datetime.fromtimestamp(timestamp, tz=dt.timezone.utc).isoformat()
        else:
            ts_iso = utc_now_iso()

        return {
            "symbol": info.get("symbol", symbol),
            "currency": info.get("currency", "USD"),
            "current": float(price),
            "previous_close": float(prev_close),
            "timestamp": ts_iso,
            "source": "yahoo-http",
        }


def savings_growth(balance: Decimal, apy_raw: str) -> Tuple[Decimal, Decimal, str]:
    percent = parse_percent(apy_raw)
    if percent is None:
        return Decimal("0"), balance, "APY missing; growth assumed 0."

    apy_float = float(percent)
    daily_rate = math.pow(1.0 + apy_float, 1.0 / 365.0) - 1.0
    daily_rate_decimal = Decimal(str(daily_rate))
    growth = (balance * daily_rate_decimal).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)
    ending = balance + growth
    detail = f"APY {percent * Decimal('100'):.2f}% -> daily rate {daily_rate * 100:.4f}%"
    return growth, ending, detail


def investment_growth(
    holdings_raw: str,
    shares_raw: str,
    share_price_raw: str,
    fetcher: PriceFetcher,
    previous_positions: Optional[Dict[str, Tuple[Decimal, Decimal]]],
    account_name: str,
) -> Tuple[Decimal, Decimal, str, Decimal, Dict[str, Tuple[Decimal, Decimal]]]:
    holdings = parse_holdings(holdings_raw)
    if not holdings:
        raise RuntimeError("Holdings missing; cannot determine investment balance.")

    share_map, default_share = parse_shares(shares_raw)
    share_price_map, default_share_price = parse_share_prices(share_price_raw)

    total_value = Decimal("0")
    detail_lines: List[str] = []
    current_positions: Dict[str, Tuple[Decimal, Decimal]] = {}

    for symbol, quantity in holdings:
        if quantity is None:
            if symbol in share_map:
                quantity = share_map[symbol]
            elif default_share is not None and len(holdings) == 1:
                quantity = default_share
            else:
                raise RuntimeError(
                    f"No share count provided for {symbol} in account {account_name}."
                )

        manual_price: Optional[Decimal] = None
        if symbol in share_price_map:
            manual_price = share_price_map[symbol]
        elif default_share_price is not None and len(holdings) == 1:
            manual_price = default_share_price

        if manual_price is not None:
            current_price = manual_price
            price_source = "manual-share-price"
            price_ts = "user-input"
        else:
            quote = fetcher.get_quote(symbol)
            current_price = quote.current
            price_source = quote.source
            price_ts = quote.timestamp

        position_value = (quantity * current_price).quantize(
            TWO_PLACES, rounding=ROUND_HALF_UP
        )
        total_value += position_value
        current_positions[symbol] = (quantity, current_price)
        detail_lines.append(
            f"{symbol}: {quantity:.6f} shares @ {current_price} ({price_source}, {price_ts})"
        )

    ending = total_value
    previous_total = None
    if previous_positions:
        previous_total = Decimal("0")
        for symbol, (shares, price) in previous_positions.items():
            if symbol == "__LEGACY__":
                previous_total += price
            else:
                previous_total += (shares * price).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    if previous_total is None:
        growth = Decimal("0")
        detail_lines.append("No prior positions available; growth shown as 0 for today.")
    else:
        growth = (ending - previous_total).quantize(TWO_PLACES, rounding=ROUND_HALF_UP)

    detail = "; ".join(detail_lines)
    starting = previous_total if previous_total is not None else ending
    return growth, ending, detail, starting, current_positions


def format_currency(amount: Decimal) -> str:
    return f"${amount.quantize(TWO_PLACES, rounding=ROUND_HALF_UP):,.2f}"


@dataclass
class AccountSummary:
    name: str
    category: str
    institution: str
    starting_balance: Decimal
    growth: Decimal
    ending_balance: Decimal
    detail: str
    share_prices: str = ""


def classify_category(raw: str) -> str:
    normalized = (raw or "").strip().lower()
    if "saving" in normalized:
        return "savings"
    if "invest" in normalized or "broker" in normalized or "retire" in normalized:
        return "investments"
    return "other"


def generate_summaries(
    rows: Iterable[Dict[str, str]], fetcher: PriceFetcher, balance_cache: BalanceCache
) -> List[AccountSummary]:
    summaries: List[AccountSummary] = []
    for row in rows:
        name = row.get("Account Name") or row.get("Account") or "Unknown Account"
        category_raw = row.get("Category") or ""
        institution = row.get("Institution") or row.get("Bank") or ""
        balance = parse_currency(row.get("Balance", "0"))
        apy = row.get("APY") or row.get("Rate") or row.get("Interest Rate")
        holdings = row.get("Holdings") or row.get("Ticker") or row.get("Holdings/Ticker") or ""
        shares = row.get("Shares") or row.get("Share Count") or row.get("Quantity") or ""
        share_price = row.get("Share Price") or ""

        if all(not (value and value.strip()) for value in row.values()):
            continue

        category = classify_category(category_raw)
        starting_balance = balance
        share_prices = ""

        try:
            if category == "savings":
                growth, ending, detail = savings_growth(balance, apy or "")
            elif category == "investments":
                prior_positions = balance_cache.get_positions(name)
                growth, ending, detail, starting_balance, positions = investment_growth(
                    holdings, shares, share_price, fetcher, prior_positions, name
                )
                balance_cache.set_positions(name, positions)
                share_prices = "; ".join(
                    format_currency(price)
                    for symbol, (_, price) in positions.items()
                    if symbol != "__LEGACY__"
                )
            else:
                growth = Decimal("0")
                ending = balance
                detail = "Category not supported; growth assumed 0."
        except Exception as exc:  # surface account-specific issues but continue
            detail = f"Error computing growth: {exc}"
            growth = Decimal("0")
            ending = balance

        summaries.append(
            AccountSummary(
                name=name,
                category=category_raw or "Unspecified",
                institution=institution,
                starting_balance=starting_balance,
                growth=growth,
                ending_balance=ending,
                detail=detail,
                share_prices=share_prices,
            )
        )
    return summaries


def print_summary(summaries: List[AccountSummary]) -> None:
    today = dt.date.today().isoformat()
    print(f"Daily Growth Summary ({today})")
    print("=" * 60)

    total_start = Decimal("0")
    total_growth = Decimal("0")
    total_end = Decimal("0")

    for summary in summaries:
        total_start += summary.starting_balance
        total_growth += summary.growth
        total_end += summary.ending_balance

        print(f"{summary.name} [{summary.category}] - {summary.institution}")
        print(f"  Starting balance: {format_currency(summary.starting_balance)}")
        print(f"  Daily growth:    {format_currency(summary.growth)}")
        print(f"  Ending balance:  {format_currency(summary.ending_balance)}")
        if summary.detail:
            print(f"  Details: {summary.detail}")
        print("-" * 60)

    print("Portfolio Totals")
    print(f"  Starting net worth: {format_currency(total_start)}")
    print(f"  Daily growth:       {format_currency(total_growth)}")
    print(f"  Ending net worth:   {format_currency(total_end)}")


def running_in_streamlit() -> bool:
    try:
        import streamlit as st  # type: ignore
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore

        return get_script_run_ctx() is not None and hasattr(st, "sidebar")
    except Exception:
        return False


def run_streamlit_app() -> int:
    try:
        import streamlit as st  # type: ignore
    except Exception as exc:
        print(f"Streamlit is required for the UI: {exc}", file=sys.stderr)
        return 1

    try:
        import pandas as pd  # type: ignore
    except Exception as exc:
        st.error(f"pandas is required for the UI: {exc}")
        return 1

    st.set_page_config(
        page_title="Investment Growth Tracker",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    st.title("Investment Growth Tracker")
    st.caption("Track daily growth, update your spreadsheet, and review balances.")

    with st.sidebar.expander("⚙ Settings", expanded=False):
        sheet_path_text = st.text_input(
            "Spreadsheet path", value="Investments Spreadsheet.csv"
        )
        cache_path_text = st.text_input("Price cache", value="price_cache.json")
        balance_cache_text = st.text_input(
            "Balance cache", value="investment_balance_cache.json"
        )
        show_details = st.checkbox("Show account details", value=False)

    sheet_path = Path(sheet_path_text)
    rows, columns = load_sheet(sheet_path)
    if not rows:
        st.info("No accounts found yet. Add one below or edit the table to begin.")

    df = pd.DataFrame(rows)
    ordered_columns = []
    for col in DEFAULT_COLUMNS:
        if col in df.columns:
            ordered_columns.append(col)
    for col in df.columns:
        if col not in ordered_columns:
            ordered_columns.append(col)
    for col in DEFAULT_COLUMNS:
        if col not in ordered_columns:
            ordered_columns.append(col)

    for col in ordered_columns:
        if col not in df.columns:
            df[col] = ""
    df = df[ordered_columns]

    sheet_state_key = str(sheet_path)
    if (
        "sheet_df" not in st.session_state
        or st.session_state.get("sheet_df_source") != sheet_state_key
    ):
        st.session_state["sheet_df"] = df.copy()
        st.session_state["sheet_df_source"] = sheet_state_key
        st.session_state["pending_delete_account_index"] = None
    if "show_add_account_form" not in st.session_state:
        st.session_state["show_add_account_form"] = False
    if "open_edit_account_index" not in st.session_state:
        st.session_state["open_edit_account_index"] = None
    if "pending_delete_account_index" not in st.session_state:
        st.session_state["pending_delete_account_index"] = None
    if "suppress_next_edit_toggle" not in st.session_state:
        st.session_state["suppress_next_edit_toggle"] = False

    def persist_sheet_state() -> None:
        rows_to_save = rows_from_dataframe(
            st.session_state["sheet_df"], list(st.session_state["sheet_df"].columns)
        )
        save_sheet(
            sheet_path,
            rows_to_save,
            list(st.session_state["sheet_df"].columns),
        )

    def clear_pending_delete() -> None:
        st.session_state["pending_delete_account_index"] = None

    def confirm_pending_delete() -> None:
        pending_idx = st.session_state.get("pending_delete_account_index")
        if pending_idx is None:
            return
        if pending_idx < 0 or pending_idx >= len(st.session_state["sheet_df"]):
            st.session_state["pending_delete_account_index"] = None
            return
        pending_name = str(
            st.session_state["sheet_df"].iloc[pending_idx].get("Account Name", "")
        ).strip() or "Unnamed account"
        st.session_state["sheet_df"] = st.session_state["sheet_df"].drop(
            index=pending_idx
        ).reset_index(drop=True)
        st.session_state["pending_delete_account_index"] = None
        st.session_state["open_edit_account_index"] = None
        persist_sheet_state()
        st.session_state["last_action_message"] = f"Deleted account: {pending_name}"

    compute_now = st.button("Refresh growth")

    summaries = st.session_state.get("last_summaries", [])
    rows_for_calc = rows_from_dataframe(
        st.session_state["sheet_df"], list(st.session_state["sheet_df"].columns)
    )
    calc_signature = json.dumps(rows_for_calc, sort_keys=True)
    should_compute = (
        compute_now
        or not summaries
        or st.session_state.get("last_summary_signature") != calc_signature
    )
    if should_compute:
        fetcher = PriceFetcher(Path(cache_path_text), allow_network=True)
        balance_cache = BalanceCache(Path(balance_cache_text))
        summaries = generate_summaries(rows_for_calc, fetcher, balance_cache)
        balance_cache.persist()
        st.session_state["last_summaries"] = summaries
        st.session_state["last_summary_signature"] = calc_signature

    if (
        summaries
        and not st.session_state["show_add_account_form"]
        and st.session_state["pending_delete_account_index"] is None
    ):
        total_start = sum(s.starting_balance for s in summaries)
        total_growth = sum(s.growth for s in summaries)
        total_end = sum(s.ending_balance for s in summaries)
        top1, top2, top3 = st.columns(3)
        top1.metric("Current Net Worth", format_currency(total_end))
        top2.metric("Daily Growth", format_currency(total_growth))
        top3.metric("Starting Net Worth", format_currency(total_start))

    summary_queues: Dict[str, List[AccountSummary]] = {}
    for summary in summaries:
        summary_key = str(summary.name).strip()
        summary_queues.setdefault(summary_key, []).append(summary)

    header_col, add_col = st.columns([8, 1])
    with header_col:
        st.subheader("Accounts")
    with add_col:
        if st.button("➕", key="add_account_toggle", help="Add account"):
            st.session_state["show_add_account_form"] = not st.session_state["show_add_account_form"]
            st.rerun()

    if st.session_state["show_add_account_form"]:
        with st.form("add_account", clear_on_submit=True):
            st.markdown("**Add account**")
            new_account = st.text_input("Account Name")
            new_category = st.text_input("Category", value="Investments")
            new_institution = st.text_input("Institution")
            new_balance = st.text_input("Balance")
            new_holdings = st.text_input("Holdings")
            new_apy = st.text_input("APY")
            new_shares = st.text_input("Shares")
            submitted = st.form_submit_button("Add")

        if submitted:
            new_row = {column: "" for column in ordered_columns}
            new_row.update(
                {
                    "Account Name": new_account,
                    "Category": new_category,
                    "Institution": new_institution,
                    "Balance": new_balance,
                    "Holdings": new_holdings,
                    "APY": new_apy,
                    "Shares": new_shares,
                }
            )
            updated_df = pd.concat(
                [st.session_state["sheet_df"], pd.DataFrame([new_row])],
                ignore_index=True,
            )
            st.session_state["sheet_df"] = updated_df
            persist_sheet_state()
            st.session_state["show_add_account_form"] = False
            st.session_state["open_edit_account_index"] = None
            st.session_state["pending_delete_account_index"] = None
            st.session_state["last_action_message"] = (
                f"Added account: {new_account.strip() or 'Unnamed account'}"
            )
            st.rerun()

    pending_delete_index = st.session_state.get("pending_delete_account_index")
    if pending_delete_index is not None:
        if pending_delete_index < 0 or pending_delete_index >= len(st.session_state["sheet_df"]):
            st.session_state["pending_delete_account_index"] = None
        else:
            pending_name = str(
                st.session_state["sheet_df"].iloc[pending_delete_index].get("Account Name", "")
            ).strip() or "Unnamed account"
            with st.container(border=True):
                st.error(f"Delete `{pending_name}`? This cannot be undone.")
                confirm_col, cancel_col = st.columns(2)
                with confirm_col:
                    st.button(
                        "Confirm Delete",
                        type="primary",
                        key=f"confirm_delete_{pending_delete_index}",
                        on_click=confirm_pending_delete,
                    )
                with cancel_col:
                    st.button(
                        "Cancel",
                        key=f"cancel_delete_{pending_delete_index}",
                        on_click=clear_pending_delete,
                    )

    if st.session_state["sheet_df"].empty:
        st.info("No accounts yet. Click the plus button to add one.")
    else:
        rows_snapshot = list(st.session_state["sheet_df"].reset_index(drop=True).iterrows())
        for idx, row in rows_snapshot:
            account_name = str(row.get("Account Name", "")).strip() or "Unnamed account"
            category = str(row.get("Category", "")).strip() or "Uncategorized"
            category_kind = classify_category(category)
            institution = str(row.get("Institution", "")).strip() or "-"
            balance_display = coerce_cell(row.get("Balance", "")) or "-"
            holdings_display = coerce_cell(row.get("Holdings", "")) or "-"
            apy_display = coerce_cell(row.get("APY", "")) or "-"
            shares_display = coerce_cell(row.get("Shares", "")) or "-"
            matched_summary = None
            queue_key = str(row.get("Account Name", "")).strip()
            if queue_key in summary_queues and summary_queues[queue_key]:
                matched_summary = summary_queues[queue_key].pop(0)
            if category_kind == "investments" and matched_summary is not None:
                balance_display = format_currency(matched_summary.ending_balance)

            with st.container(border=True):
                left_col, edit_col, delete_col = st.columns([10, 1, 1])
                with left_col:
                    st.markdown(f"**{account_name}**")
                    st.caption(f"{category} | {institution}")
                    detail_parts = [f"Balance: `{balance_display}`"]
                    if category_kind == "investments":
                        detail_parts.append(f"Holdings: `{holdings_display}`")
                        detail_parts.append(f"Shares: `{shares_display}`")
                    elif category_kind == "savings":
                        detail_parts.append(f"APY: `{apy_display}`")
                    else:
                        detail_parts.append(f"Holdings: `{holdings_display}`")
                        detail_parts.append(f"APY: `{apy_display}`")
                    st.markdown("  |  ".join(detail_parts))
                with edit_col:
                    edit_clicked = st.button("✏️", key=f"edit_toggle_{idx}", help="Edit account")
                with delete_col:
                    delete_clicked = st.button("🗑️", key=f"delete_account_{idx}", help="Delete account")

                if edit_clicked:
                    if st.session_state["suppress_next_edit_toggle"]:
                        st.session_state["suppress_next_edit_toggle"] = False
                    else:
                        if st.session_state["open_edit_account_index"] == idx:
                            st.session_state["open_edit_account_index"] = None
                        else:
                            st.session_state["open_edit_account_index"] = idx
                    st.session_state["pending_delete_account_index"] = None
                    st.rerun()

                if delete_clicked:
                    st.session_state["pending_delete_account_index"] = idx
                    st.session_state["open_edit_account_index"] = None
                    st.rerun()

                if st.session_state["open_edit_account_index"] == idx:
                    st.markdown("**Edit account**")
                    with st.form(f"edit_account_form_{idx}", clear_on_submit=False):
                        edited_values = {}
                        for col in ordered_columns:
                            edited_values[col] = st.text_input(
                                col,
                                value=coerce_cell(row.get(col, "")),
                            )
                        save_col, cancel_col = st.columns(2)
                        with save_col:
                            update_clicked = st.form_submit_button("Save Changes", type="primary")
                        with cancel_col:
                            cancel_edit_clicked = st.form_submit_button("Cancel")

                    if cancel_edit_clicked:
                        st.session_state["open_edit_account_index"] = None
                        st.session_state["pending_delete_account_index"] = None
                        st.session_state["suppress_next_edit_toggle"] = True
                        st.rerun()

                    if update_clicked:
                        for col in ordered_columns:
                            st.session_state["sheet_df"].at[idx, col] = edited_values[col]
                        persist_sheet_state()
                        st.session_state["open_edit_account_index"] = None
                        st.session_state["pending_delete_account_index"] = None
                        st.session_state["suppress_next_edit_toggle"] = True
                        st.session_state["last_action_message"] = (
                            f"Updated account: {edited_values.get('Account Name', account_name)}"
                        )
                        st.rerun()

    action_message = st.session_state.pop("last_action_message", "")
    if action_message:
        st.success(action_message)

    if not summaries:
        st.info("No summaries available yet.")
        return 0

    if compute_now and "Share Price" in st.session_state["sheet_df"].columns:
        share_price_map = {summary.name: summary.share_prices for summary in summaries}
        updated_df = st.session_state["sheet_df"].copy()
        for idx, row in updated_df.iterrows():
            account_name = str(row.get("Account Name", "")).strip()
            if account_name and account_name in share_price_map:
                updated_df.at[idx, "Share Price"] = share_price_map[account_name]
        st.session_state["sheet_df"] = updated_df

    summary_data = [
        {
            "Account": summary.name,
            "Category": summary.category,
            "Institution": summary.institution,
            "Starting Balance": float(summary.starting_balance),
            "Daily Growth": float(summary.growth),
            "Ending Balance": float(summary.ending_balance),
            "Details": summary.detail,
            "Share Price": summary.share_prices,
        }
        for summary in summaries
    ]
    summary_df = pd.DataFrame(summary_data)

    st.subheader("Daily Growth by Account")
    chart_df = summary_df[["Account", "Daily Growth"]].set_index("Account")
    chart_type = st.radio(
        "Chart view",
        options=["Line (default)", "Bar", "Area"],
        horizontal=True,
        label_visibility="collapsed",
    )
    if chart_type == "Bar":
        st.bar_chart(chart_df, height=320, use_container_width=True)
    elif chart_type == "Area":
        st.area_chart(chart_df, height=320, use_container_width=True)
    else:
        st.line_chart(chart_df, height=320, use_container_width=True)

    display_columns = [
        "Account",
        "Category",
        "Institution",
        "Starting Balance",
        "Daily Growth",
        "Ending Balance",
    ]
    if show_details:
        display_columns.append("Details")
    st.dataframe(summary_df[display_columns], use_container_width=True, height=360)

    return 0


def main() -> int:
    args = parse_args()
    log_level = logging.DEBUG if args.debug else logging.CRITICAL
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s %(levelname)s %(name)s - %(message)s",
    )

    sheet_path = Path(args.sheet)
    if not sheet_path.exists():
        print(f"Spreadsheet not found at {sheet_path}", file=sys.stderr)
        return 1

    fetcher = PriceFetcher(Path(args.cache), allow_network=not args.no_network)
    balance_cache = BalanceCache(Path(args.balance_cache))

    with sheet_path.open("r", encoding="utf-8-sig", newline="") as csvfile:
        reader = csv.DictReader(csvfile)
        summaries = generate_summaries(reader, fetcher, balance_cache)

    balance_cache.persist()

    if not summaries:
        print("No accounts found in the spreadsheet.", file=sys.stderr)
        return 1

    print_summary(summaries)
    return 0


if __name__ == "__main__":
    if running_in_streamlit():
        raise SystemExit(run_streamlit_app())
    raise SystemExit(main())
