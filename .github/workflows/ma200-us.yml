"""Independent US 200-day moving-average recovery screener.

Usage: python screener_ma200_us.py
Output: results/ma200_us.json and results/ma200_us.csv
Shares the detection rules with the JP scanner, but uses US market hours.
"""
from __future__ import annotations

import json
from datetime import datetime, time, timezone
from io import StringIO
from pathlib import Path
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import yfinance as yf

LOOKBACK_CROSS = 20
MAX_DISTANCE_PCT = 15.0
BATCH_SIZE = 25
SP500_URL = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
# Supplement S&P 500 with growth/AI names already commonly monitored.
EXTRA_TICKERS = '''ARM MRVL LITE COHR ALAB CRDO CAMT SNDK NBIS CRWV IREN BE RKLB ASTS SOFI HOOD COIN SNOW IOT SHOP APP PLTR DDOG NET MDB HUBS CRWD PANW ZS MSTR SMCI CELH CAVA RDDT HIMS DUOL UBER ABNB IONQ RGTI OKLO'''.split()


def get_us_names():
    req = Request(SP500_URL, headers={'User-Agent': 'Mozilla/5.0 (compatible; TradingTerminal/1.0)'})
    with urlopen(req, timeout=40) as response:
        tables = pd.read_html(StringIO(response.read().decode('utf-8')))
    source = next((t for t in tables if 'Symbol' in t.columns and 'Security' in t.columns), None)
    if source is None:
        raise RuntimeError('S&P500構成銘柄の取得に失敗しました')
    names = {}
    for _, r in source.iterrows():
        symbol = str(r['Symbol']).strip().replace('.', '-')
        if symbol and symbol != 'nan':
            names[symbol] = str(r['Security']).strip()
    if len(names) < 450:
        raise RuntimeError(f'S&P500一覧が少なすぎます: {len(names)}')
    for ticker in EXTRA_TICKERS:
        names.setdefault(ticker, ticker)  # names may be updated by Yahoo metadata later
    return names


def completed_bars(df):
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    if 'Close' not in df:
        return pd.DataFrame()
    df = df.dropna(subset=['Close'])
    if df.empty:
        return df
    now = datetime.now(ZoneInfo('America/New_York'))
    last_day = pd.Timestamp(df.index[-1]).date()
    # Daily US bar may be incomplete until shortly after regular session close.
    if last_day == now.date() and now.time() < time(16, 30):
        df = df.iloc[:-1].copy()
    return df


def extract_symbol(downloaded, ticker):
    if downloaded is None or downloaded.empty:
        return pd.DataFrame()
    if isinstance(downloaded.columns, pd.MultiIndex):
        if ticker in downloaded.columns.get_level_values(0):
            return downloaded[ticker].copy()
        if ticker in downloaded.columns.get_level_values(1):
            return downloaded.xs(ticker, axis=1, level=1).copy()
        return pd.DataFrame()
    return downloaded.copy()


def detect(df):
    # Same screening logic as screener_ma200_jp.py (2026-11-09).
    close = pd.to_numeric(df['Close'], errors='coerce')
    if len(close) < 265 or close.isna().any() or (close <= 0).any():
        return None
    ma = close.rolling(200, min_periods=200).mean()
    old_60 = float(ma.iloc[-61])
    old_40 = float(ma.iloc[-41])
    last_ma = float(ma.iloc[-1])
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
        if float(close.iloc[i - 1]) <= float(ma.iloc[i - 1]) and float(close.iloc[i]) > float(ma.iloc[i]):
            crossings.append(i)
    if not crossings:
        return None
    cross_i = crossings[-1]
    days_ago = len(close) - 1 - cross_i
    return {
        'BAR_DATE': str(pd.Timestamp(df.index[-1]).date()),
        'Price': round(last_close, 2),
        'MA200': round(last_ma, 2),
        'Distance_Pct': round(distance, 2),
        'Cross_Days_Ago': int(days_ago),
        'Cross_Date': str(pd.Timestamp(df.index[cross_i]).date()),
        'Past_MA200_Change_Pct': round((old_40 / old_60 - 1) * 100, 2),
        'Stage': '上抜け直後' if days_ago <= 5 else '転換確認',
    }


def main():
    names = get_us_names()
    tickers = list(names)
    results, errors = [], []
    analyzed = 0
    for offset in range(0, len(tickers), BATCH_SIZE):
        group = tickers[offset:offset + BATCH_SIZE]
        print(f'Batch {offset // BATCH_SIZE + 1}: {offset + 1}-{offset + len(group)}/{len(tickers)}', flush=True)
        try:
            batch = yf.download(group, period='2y', interval='1d', auto_adjust=True,
                                group_by='ticker', threads=4, progress=False, timeout=35)
        except Exception as exc:
            errors.append(f'batch {offset}: {exc}')
            continue
        for ticker in group:
            try:
                df = completed_bars(extract_symbol(batch, ticker))
                if len(df) < 265:
                    continue
                analyzed += 1
                found = detect(df)
                if found:
                    results.append({'Ticker': ticker, 'Name': names[ticker], 'YahooSymbol': ticker, **found})
            except Exception as exc:
                errors.append(f'{ticker}: {exc}')
    if analyzed < 350:
        raise RuntimeError(f'分析可能な銘柄が{analyzed}件です。取得失敗の可能性があるため旧結果を保持します')
    results.sort(key=lambda s: (s['Cross_Days_Ago'], s['Distance_Pct']))
    output = {
        'updated_at': datetime.now(timezone.utc).isoformat(),
        'market': 'US S&P500 + Watchlist',
        'universe': len(tickers), 'analyzed': analyzed, 'total': len(results),
        'settings': {'cross_lookback_days': LOOKBACK_CROSS, 'max_distance_pct': MAX_DISTANCE_PCT},
        'stocks': results,
    }
    folder = Path('results'); folder.mkdir(exist_ok=True)
    with (folder / 'ma200_us.json').open('w', encoding='utf-8') as f:
        json.dump(output, f, ensure_ascii=False, indent=2, allow_nan=False)
    cols = ['Ticker', 'Name', 'BAR_DATE', 'Price', 'MA200', 'Distance_Pct',
            'Cross_Days_Ago', 'Cross_Date', 'Past_MA200_Change_Pct', 'Stage', 'YahooSymbol']
    pd.DataFrame(results, columns=cols).to_csv(folder / 'ma200_us.csv', index=False, encoding='utf-8-sig')
    print(f'Finished: universe={len(tickers)}, analyzed={analyzed}, matches={len(results)}, errors={len(errors)}')
    for err in errors[:10]:
        print('Warning:', err)


if __name__ == '__main__':
    main()
