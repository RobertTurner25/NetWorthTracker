import json
import tempfile
import unittest
from functools import wraps
from pathlib import Path

import investment_growth as ig


def report_test(fn):
    @wraps(fn)
    def wrapper(self, *args, **kwargs):
        test_name = f"{self.__class__.__name__}.{fn.__name__}"
        try:
            result = fn(self, *args, **kwargs)
        except Exception as exc:
            print(f"[FAIL] {test_name}: {exc}")
            raise
        else:
            print(f"[PASS] {test_name}")
            return result

    return wrapper


class DummyFetcher:
    def __init__(self, prices):
        self.prices = prices

    def get_quote(self, symbol: str) -> ig.Quote:
        try:
            info = self.prices[symbol]
        except KeyError as exc:  # pragma: no cover - defensive
            raise AssertionError(f"Missing quote for {symbol}") from exc
        return ig.Quote(
            symbol=symbol,
            currency=info.get("currency", "USD"),
            current=ig.Decimal(str(info["current"])),
            previous_close=ig.Decimal(str(info.get("previous_close", info["current"]))),
            timestamp=info.get("timestamp", "2026-01-01T00:00:00Z"),
            source=info.get("source", "test"),
        )


class InvestmentGrowthTests(unittest.TestCase):
    @report_test
    def test_parse_helpers(self):
        self.assertEqual(ig.parse_currency("$1,234.56"), ig.Decimal("1234.56"))
        self.assertIsNone(ig.parse_percent(""))
        self.assertEqual(ig.parse_percent("2.5%"), ig.Decimal("0.025"))

        holdings = ig.parse_holdings("VOO:2;eth-usd=1.5| msft 4")
        expected = [("VOO", ig.Decimal("2")), ("ETH-USD", ig.Decimal("1.5")), ("MSFT", ig.Decimal("4"))]
        self.assertEqual(holdings, expected)

        share_map, default_qty = ig.parse_shares("VOO:3;ETH-USD:1.25")
        self.assertEqual(share_map["VOO"], ig.Decimal("3"))
        self.assertEqual(share_map["ETH-USD"], ig.Decimal("1.25"))
        self.assertIsNone(default_qty)

        share_map, default_qty = ig.parse_shares("5.5")
        self.assertEqual(default_qty, ig.Decimal("5.5"))
        self.assertFalse(share_map)

    @report_test
    def test_savings_growth_calculation(self):
        growth, ending, detail = ig.savings_growth(ig.Decimal("1000"), "3.65%")
        self.assertEqual(ending, ig.Decimal("1000.10"), f"Unexpected savings ending: {ending} ({detail})")
        self.assertEqual(growth, ig.Decimal("0.10"))

    @report_test
    def test_investment_growth_with_shares_and_prior_balance(self):
        fetcher = DummyFetcher(
            {
                "VOO": {"current": "100", "previous_close": "95"},
                "ETH-USD": {"current": "2000", "previous_close": "1900"},
            }
        )
        holdings = "VOO;ETH-USD"
        shares = "VOO:2;ETH-USD:0.5"
        prior_positions = {
            "VOO": (ig.Decimal("2"), ig.Decimal("95")),
            "ETH-USD": (ig.Decimal("0.5"), ig.Decimal("1900")),
        }
        growth, ending, detail, starting, _positions = ig.investment_growth(
            holdings, shares, "", fetcher, prior_positions, "My Account"
        )
        self.assertEqual(ending, ig.Decimal("1200.00"), f"Detail: {detail}")
        self.assertEqual(growth, ig.Decimal("60.00"))
        self.assertEqual(starting, ig.Decimal("1140.00"))

    @report_test
    def test_investment_growth_raises_when_shares_missing(self):
        fetcher = DummyFetcher({"VOO": {"current": "100"}})
        with self.assertRaises(RuntimeError):
            ig.investment_growth("VOO", "", "", fetcher, None, "Test")

    @report_test
    def test_investment_growth_uses_manual_share_price(self):
        fetcher = DummyFetcher({"VOO": {"current": "100"}})
        growth, ending, _detail, _starting, _positions = ig.investment_growth(
            "VOO", "2", "125", fetcher, None, "Manual Price Account"
        )
        self.assertEqual(ending, ig.Decimal("250.00"))
        self.assertEqual(growth, ig.Decimal("0"))

    @report_test
    def test_balance_cache_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "balance.json"
            cache = ig.BalanceCache(cache_path)
            self.assertIsNone(cache.get_positions("Account A"))
            cache.set_positions("Account A", {"VOO": (ig.Decimal("2"), ig.Decimal("123.45"))})
            cache.persist()

            reloaded = ig.BalanceCache(cache_path)
            positions = reloaded.get_positions("Account A")
            self.assertIsNotNone(positions)
            self.assertIn("VOO", positions)
            self.assertEqual(
                positions["VOO"],
                (ig.Decimal("2"), ig.Decimal("123.45")),
                "Positions cache did not persist numeric values",
            )

    @report_test
    def test_generate_summaries_with_savings_and_investment(self):
        fetcher = DummyFetcher({"VOO": {"current": "150"}})
        rows = [
            {
                "Account Name": "Saver",
                "Category": "Savings",
                "Institution": "Capitol One",
                "Balance": "$10,000.00",
                "APY": "3.0%",
            },
            {
                "Account Name": "Brokerage",
                "Category": "Investments",
                "Institution": "Vanguard",
                "Holdings": "VOO",
                "Shares": "3",
            },
        ]
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "balance.json"
            cache_path.write_text(
                json.dumps(
                    {
                        "Brokerage": {
                            "positions": {"VOO": {"shares": 2, "price": 220.0}}
                        }
                    }
                )
            )
            balance_cache = ig.BalanceCache(cache_path)
            summaries = ig.generate_summaries(rows, fetcher, balance_cache)

        self.assertEqual(len(summaries), 2)
        saver = next(s for s in summaries if s.name == "Saver")
        broker = next(s for s in summaries if s.name == "Brokerage")

        self.assertEqual(saver.ending_balance, ig.Decimal("10000.81"))
        self.assertEqual(broker.ending_balance, ig.Decimal("450.00"))
        self.assertEqual(broker.growth, ig.Decimal("10.00"), "Investment growth should be current minus cached value")


if __name__ == "__main__":
    unittest.main()
