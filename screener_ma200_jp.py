"""Independent JPX Prime 200-day moving-average recovery screener.

Run: python screener_ma200_jp.py
Output: results/ma200_jp.json and results/ma200_jp.csv
Existing US/JP screeners and their result files are not modified.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, time, timezone
from io import BytesIO
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

JPX_URL = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
LOOKBACK_CROSS = 20          # last 20 completed trading days
MAX_DISTANCE_PCT = 15.0     # price less than 15% above SMA200
BATCH_SIZE = 25


def get_prime_names():
    req = Request(JPX_URL, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        listing = pd.read_excel(BytesIO(response.read()), dtype=str, engine="openpyxl")
    required = {"コード", "銘柄名", "市場・商品区分"}
    if not required.issubset(listing.columns):
        raise RuntimeError("JPXの銘柄一覧の列形式が変更されました")
    prime = listing[listing["市場・商品区分"].astype(str).str.strip() == "プライム（内国株式）"]
    names = {}
    for _, row in prime.iterrows():
        code = str(row["コード"]).strip()
        if re.fullmatch(r"[0-9]{4}|[0-9]{3}[A-Z]", code):
            names[code] = str(row["銘柄名"]).strip() if pd.notna(row["銘柄名"]) else ""
    if len(names) < 1000:
        raise RuntimeError("JPXのプライム銘柄が1000未満です。取得結果を確認してください")
    return names


def completed_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    for field in ("Close", "Volume"):
        if field not in df:
            return pd.DataFrame()
    df = df.dropna(subset=["Close"])
    if df.empty:
        return df
    now = datetime.now(ZoneInfo("Asia/Tokyo"))
    last_day = pd.Timestamp(df.index[-1]).date()
    if last_day == now.date() and now.time() < time(16, 0):
        df = df.iloc[:-1].copy()
    return df


def detect(df):
    """Return earliest useful latest crossover, using only completed daily closes.

    SMA200 declining at the historical -60 -> -40 day segment; it may be
    flat or rising today. Detect most recent close crossing over SMA200
    within the past 20 days and still above it now. No future bars.
    """
    close = pd.to_numeric(df["Close"], errors="coerce")
    if len(close) < 265 or close.isna().any() or (close <= 0).any():
        return None
    ma = close.rolling(200, min_periods=200).mean()
    old_60 = float(ma.iloc[-61]); old_40 = float(ma.iloc[-41]); last_ma = float(ma.iloc[-1])
    if not all(np.isfinite(x) and x > 0 for x in (old_60, old_40, last_ma)):
        return None
    if not old_40 < old_60:
        return None
    last_close = float(close.iloc[-1])
    distance = (last_close / last_ma - 1) * 100
    if not (0 <= distance <= MAX_DISTANCE_PCT):
        return None
    crossings = []
    for i in range(len(close) - LOOKBACK_CROSS, len(close)):
        if float(close.iloc[i-1]) <= float(ma.iloc[i-1]) and float(close.iloc[i]) > float(ma.iloc[i]):
            crossings.append(i)
    if not crossings:
        return None
    cross_i = crossings[-1]
    days_ago = len(close) - 1 - cross_i
    return {
        "BAR_DATE": str(pd.Timestamp(df.index[-1]).date()),
        "Price": round(last_close, 2),
        "MA200": round(last_ma, 2),
        "Distance_Pct": round(distance, 2),
        "Cross_Days_Ago": int(days_ago),
        "Cross_Date": str(pd.Timestamp(df.index[cross_i]).date()),
        "Past_MA200_Change_Pct": round((old_40 / old_60 - 1) * 100, 2),
        "Stage": "上抜け直後" if days_ago <= 5 else "転換確認",
    }


def chart_history(df, bars=90):
    """Recent completed adjusted daily closes and matching rolling SMA200."""
    close = pd.to_numeric(df["Close"], errors="coerce")
    sma = close.rolling(200, min_periods=200).mean()
    tail = pd.DataFrame({"close": close, "ma200": sma}).dropna().tail(bars)
    return [{"date": str(pd.Timestamp(day).date()),
             "close": round(float(row.close), 2),
             "ma200": round(float(row.ma200), 2)}
            for day, row in tail.iterrows()]


def extract_symbol(downloaded, ticker):
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        # default group_by='ticker' puts ticker at level zero
        if ticker in downloaded.columns.get_level_values(0):
            return downloaded[ticker].copy()
        # defensive handling if column order was changed by yfinance
        if ticker in downloaded.columns.get_level_values(1):
            return downloaded.xs(ticker, axis=1, level=1).copy()
        return pd.DataFrame()
    return downloaded.copy()


def main():
    names = get_prime_names()
    tickers = [code + ".T" for code in names]
    results = []
    analyzed = 0
    errors = []
    for offset in range(0, len(tickers), BATCH_SIZE):
        group = tickers[offset:offset + BATCH_SIZE]
        print(f"Batch {offset // BATCH_SIZE + 1}: {offset + 1}-{offset + len(group)}/{len(tickers)}", flush=True)
        try:
            batch = yf.download(
                group, period="2y", interval="1d", auto_adjust=True,
                group_by="ticker", threads=4, progress=False,
                timeout=35,
            )
        except Exception as exc:
            errors.append(f"batch {offset}: {exc}")
            continue
        for ticker in group:
            try:
                df = completed_bars(extract_symbol(batch, ticker))
                if len(df) < 265:
                    continue
                analyzed += 1
                found = detect(df)
                if found:
                    code = ticker.removesuffix(".T")
                    results.append({"Ticker": code, "Name": names[code], "YahooSymbol": ticker, **found, "Chart": chart_history(df)})
            except Exception as exc:
                errors.append(f"{ticker}: {exc}")
    if analyzed < 500:
        raise RuntimeError(f"分析可能な銘柄が{analyzed}件だけです。データ取得失敗の可能性があるため、旧結果を維持します。")
    results.sort(key=lambda s: (s["Cross_Days_Ago"], s["Distance_Pct"]))
    output = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "market": "JPX Prime",
        "universe": len(tickers), "analyzed": analyzed, "total": len(results),
        "settings": {"cross_lookback_days": LOOKBACK_CROSS, "max_distance_pct": MAX_DISTANCE_PCT},
        "stocks": results,
    }
    folder = Path("results"); folder.mkdir(exist_ok=True)
    with (folder / "ma200_jp.json").open("w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    pd.DataFrame(results, columns=["Ticker", "Name", "BAR_DATE", "Price", "MA200", "Distance_Pct", "Cross_Days_Ago", "Cross_Date", "Past_MA200_Change_Pct", "Stage", "YahooSymbol"]).to_csv(folder / "ma200_jp.csv", index=False, encoding="utf-8-sig")
    print(f"Finished: prime={len(tickers)}, analyzed={analyzed}, matches={len(results)}, errors={len(errors)}")
    for err in errors[:10]:
        print("Warning:", err)


if __name__ == "__main__":
    main()
