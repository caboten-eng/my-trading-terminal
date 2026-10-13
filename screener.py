
import yfinance as yf
import pandas as pd
from pathlib import Path

TICKERS = [
    "NVDA", "MU", "SNDK", "NBIS",
    "PLTR", "SOFI", "ARM", "MRVL"
]

results = []

for ticker in TICKERS:
    try:
        df = yf.download(
            ticker,
            period="2y",
            auto_adjust=True,
            progress=False
        )

        if df.empty or len(df) < 221:
            print(f"{ticker}: データ不足")
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()

        if len(close) < 221:
            continue

        price = float(close.iloc[-1])
        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        high52 = float(close.tail(252).max())
        low52 = float(close.tail(252).min())

        checks = {
            "Price > MA50/150/200": (
                price > ma50.iloc[-1]
                and price > ma150.iloc[-1]
                and price > ma200.iloc[-1]
            ),
            "MA50 > MA150 > MA200": (
                ma50.iloc[-1] > ma150.iloc[-1]
                > ma200.iloc[-1]
            ),
            "MA200 rising": (
                ma200.iloc[-1] > ma200.iloc[-22]
            ),
            "Within 25% of high": price >= high52 * 0.75,
            "30% above low": price >= low52 * 1.30
        }

        passed = all(checks.values())

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "MA50": round(float(ma50.iloc[-1]), 2),
            "MA150": round(float(ma150.iloc[-1]), 2),
            "MA200": round(float(ma200.iloc[-1]), 2),
            "PASS": passed
        })

        print(f"{ticker}: {'PASS' if passed else 'FAIL'}")

    except Exception as e:
        print(f"{ticker}: ERROR - {e}")

if not results:
    raise RuntimeError("有効な株価データを取得できませんでした")

output = pd.DataFrame(results)
Path("results").mkdir(exist_ok=True)
output.to_csv("results/screening.csv", index=False)

print("\nスクリーニング完了")
print(output)
