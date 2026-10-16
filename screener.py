
import yfinance as yf
import pandas as pd
from pathlib import Path

# S&P 500構成銘柄を自動取得
def get_sp500_tickers():
    from io import StringIO
    from urllib.request import Request, urlopen

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"

    req = Request(
        url,
        headers={
            "User-Agent": "Mozilla/5.0 (compatible; StockScreener/1.0)"
        }
    )

    with urlopen(req, timeout=30) as response:
        html = response.read().decode("utf-8")

    tables = pd.read_html(StringIO(html))
    tickers = tables[0]["Symbol"].tolist()

    return [ticker.replace(".", "-") for ticker in tickers]


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

# 既存の監視銘柄にS&P 500構成銘柄を追加
try:
    sp500_tickers = get_sp500_tickers()
    print(f"S&P 500構成銘柄: {len(sp500_tickers)}銘柄取得")
    TICKERS = list(dict.fromkeys(TICKERS + sp500_tickers))
except Exception as e:
    print(f"S&P 500取得失敗: {e}")
    print("既存の監視銘柄のみで実行します")
    TICKERS = list(dict.fromkeys(TICKERS))

print(f"スクリーニング対象: {len(TICKERS)}銘柄")


import json
from datetime import datetime, timezone

results = []

for ticker in TICKERS:
    try:
        df = yf.download(
            ticker,
            period="2y",
            auto_adjust=True,
            progress=False
        )

        if df.empty:
            print(f"{ticker}: データ不足")
            continue

        close = df["Close"]
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        close = close.dropna()

        # 252営業日 + MA200の過去比較に必要な履歴
        if len(close) < 253:
            print(f"{ticker}: 履歴不足")
            continue

        price = float(close.iloc[-1])

        ma50 = close.rolling(50).mean()
        ma150 = close.rolling(150).mean()
        ma200 = close.rolling(200).mean()

        high52 = float(close.tail(252).max())
        low52 = float(close.tail(252).min())

        # 3・6・9・12か月の各区間の騰落率
        p0 = float(close.iloc[-1])
        p3 = float(close.iloc[-64])
        p6 = float(close.iloc[-127])
        p9 = float(close.iloc[-190])
        p12 = float(close.iloc[-253])

        r1 = p0 / p3 - 1
        r2 = p3 / p6 - 1
        r3 = p6 / p9 - 1
        r4 = p9 / p12 - 1

        rs_raw = (
            r1 * 0.4 +
            r2 * 0.2 +
            r3 * 0.2 +
            r4 * 0.2
        )

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

        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "MA50": round(float(ma50.iloc[-1]), 2),
            "MA150": round(float(ma150.iloc[-1]), 2),
            "MA200": round(float(ma200.iloc[-1]), 2),
            "Trend_PASS": bool(all(checks.values())),
            "RS_raw": rs_raw
        })

        print(f"{ticker}: 取得成功")

    except Exception as e:
        print(f"{ticker}: ERROR - {e}")

if not results:
    raise RuntimeError("有効な株価データを取得できませんでした")

output = pd.DataFrame(results)

# 対象銘柄内でRSを相対評価
output["RS"] = (
    output["RS_raw"].rank(pct=True) * 99
).round().clip(1, 99).astype(int)

# RS上位30%を判定
output["RS_PASS"] = output["RS"] >= 70

# トレンド条件とRS条件を統合
output["PASS"] = (
    output["Trend_PASS"] & output["RS_PASS"]
)

# RSの高い順に並べる
output = output.sort_values(
    "RS", ascending=False
)

output["RS_raw"] = output["RS_raw"].round(4)

Path("results").mkdir(exist_ok=True)

output.to_csv("results/screening.csv", index=False)

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
