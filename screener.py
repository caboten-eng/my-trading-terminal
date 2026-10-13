
import yfinance as yf
import pandas as pd
from pathlib import Path


TICKERS = [
    # 半導体・AI
    "NVDA", "AMD", "AVGO", "MU", "SNDK",
    "ARM", "MRVL", "QCOM", "TXN", "ADI",
    "MCHP", "MPWR", "ON", "AMAT", "LRCX",
    "KLAC", "ASML", "TSM", "INTC", "COHR",
    "LITE", "SITM", "ALAB", "CRDO", "CAMT",

    # クラウド・ソフトウェア
    "MSFT", "GOOGL", "AMZN", "META", "ORCL",
    "CRM", "NOW", "PLTR", "SNOW", "DDOG",
    "NET", "MDB", "HUBS", "IOT", "CRWD",
    "PANW", "ZS", "FTNT", "SHOP", "APP",

    # AIインフラ・電力
    "NBIS", "CRWV", "IREN", "BE", "VRT",
    "ETN", "GEV", "CEG", "VST", "PWR",

    # 宇宙・防衛
    "RKLB", "ASTS", "LMT", "RTX", "NOC",
    "GD", "BA", "HII", "KTOS", "PL",

    # 金融・フィンテック
    "SOFI", "HOOD", "COIN", "PYPL", "SQ",
    "V", "MA", "JPM", "GS", "MS",

    # 小売・消費
    "WMT", "COST", "AMZN", "TGT", "DG",
    "DLTR", "HD", "LOW", "NKE", "SBUX",

    # その他
    "AAPL", "TSLA", "NFLX", "UBER", "ABNB",
    "DE", "CAT", "UPS", "PLD", "ALB"
]

# 重複銘柄を除外
TICKERS = list(dict.fromkeys(TICKERS))

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


import json
from datetime import datetime, timezone

output = pd.DataFrame(results)
Path("results").mkdir(exist_ok=True)

# CSV保存
output.to_csv("results/screening.csv", index=False)

# JSON保存
data = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "total": len(output),
    "passed": int(output["PASS"].sum()),
    "stocks": output.to_dict(orient="records")
}

with open("results/screening.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\nスクリーニング完了")
print(f"対象銘柄: {data['total']}")
print(f"合格銘柄: {data['passed']}")
print(output)
