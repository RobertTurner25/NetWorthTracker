# Investment Growth Tracker

A Python tool that pulls live market data for the holdings in `Investments Spreadsheet.csv`, estimates daily growth for savings and investment accounts, and stores a rolling cache of prices and prior-day balances so each run compares today's value to yesterday's.

## Files
- `investment_growth.py` – main script with CLI options for sheet path, price cache, and balance cache.
- `Investments Spreadsheet.csv` – Google Sheets export containing account metadata (see format below).
- `price_cache.json` – optional storage for market quotes so throttled requests can fall back to recent values.
- `investment_balance_cache.json` – stores yesterday's ending balance for every investment account so growth is computed as `today - yesterday`.

## Spreadsheet format
| Column | Description |
| --- | --- |
| `Account Name` | Friendly display name. Used as the key in the balance cache. |
| `Category` | Determines the growth model (`Savings` uses APY; `Investments` uses holdings + shares). |
| `Institution` | Bank/broker name, only used for display. |
| `Balance` | Current amount for savings accounts. Ignored for investments. |
| `Holdings` | Semicolon/pipe/plus-separated list of tickers (e.g., `VOO`, `ETH-USD`). |
| `APY` | Percentage for savings accounts. |
| `Shares` | Quantity per holding (`VOO:3`) or a single number when the account only has one holding. |

## Requirements
- Python 3.10+
- Packages: `streamlit`, `yfinance`, `pandas`, `numpy`

Install dependencies from the repository root:
```bash
python3 -m pip install -r requirements.txt
```

Windows note:
- This repo includes `python3.cmd` so `python3 ...` works in Command Prompt from this folder by forwarding to `.venv\Scripts\python.exe`.
- If `.venv` is missing, create it first (`python -m venv .venv`) and reinstall requirements.

## Usage
```bash
cd investment_growth_suite
python3 investment_growth.py \
  --sheet "Investments Spreadsheet.csv" \
  --cache price_cache.json \
  --balance-cache investment_balance_cache.json
```
Add `--debug` for verbose logging when troubleshooting network requests or cache lookups.

## Streamlit UI
Run the app locally with:
```bash
cd investment_growth_suite
python3 -m streamlit run investment_growth.py
```
The UI lets you edit the spreadsheet, add new accounts, compute daily growth, and view charts.

## Free Hosting (Streamlit Community Cloud)
1. Push this folder to a GitHub repository.
2. Ensure `requirements.txt` is present with the packages listed above.
3. In Streamlit Community Cloud, create a new app and set the main file to `investment_growth.py`.
4. Deploy and open the app URL from your phone.

## Testing
Comprehensive unit tests live under `tests/` and can be executed with:
```bash
cd investment_growth_suite
python3 -m unittest discover tests
```
The suite covers parsing helpers, savings math, investment growth calculations (including share handling and prior-day comparisons), and the balance cache persistence logic. Assertions surface detailed context on failure to make diagnosing issues straightforward.
