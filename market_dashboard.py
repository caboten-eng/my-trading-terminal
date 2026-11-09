#!/usr/bin/env python3
"""US market dashboard data: breadth, 52-week highs/lows, sector ETFs, RS ranking.
No changes to screening criteria. This uses current screener universe, not the entire US market.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parent
SCREEN = ROOT / 'results' / 'screening.json'
OUT = ROOT / 'results' / 'market_dashboard.json'
SECTORS = {
    'SMH': '半導体', 'XLK': '情報技術', 'XLF': '金融', 'XLV': 'ヘルスケア',
    'XLY': '一般消費財', 'XLP': '生活必需品', 'XLI': '資本財', 'XLE': 'エネルギー',
    'XLU': '公益', 'XLC': '通信サービス', 'XLRE': '不動産', 'XLB': '素材',
}

def bars_for(symbols, period='18mo'):
    # yfinance may return a MultiIndex with either price or symbol first.
    frames = {}
    for start in range(0, len(symbols), 35):
        group = symbols[start:start+35]
        try:
            df = yf.download(group, period=period, interval='1d', group_by='ticker',
                             auto_adjust=True, progress=False, threads=False, timeout=25)
            if df is None or df.empty:
                continue
            for symbol in group:
                try:
                    if isinstance(df.columns, pd.MultiIndex):
                        if symbol in df.columns.get_level_values(0):
                            sub = df[symbol]
                        elif symbol in df.columns.get_level_values(1):
                            sub = df.xs(symbol, axis=1, level=1)
                        else:
                            continue
                    elif len(group) == 1:
                        sub = df
                    else:
                        continue
                    cols = {str(c).lower(): c for c in sub.columns}
                    if not {'close','high','low'}.issubset(cols):
                        continue
                    v = sub[[cols['close'], cols['high'], cols['low']]].copy()
                    v.columns = ['Close','High','Low']
                    v = v.apply(pd.to_numeric, errors='coerce').dropna()
                    if len(v) >= 5:
                        frames[symbol] = v
                except (KeyError, ValueError, TypeError):
                    continue
        except Exception as exc:
            print('Download failed:', group[:3], str(exc)[:160])
        time.sleep(.2)
        print('Downloaded', len(frames), '/', len(symbols), flush=True)
    return frames

def perf(series, n):
    if len(series) <= n or series.iloc[-(n+1)] <= 0: return None
    return round(100 * (float(series.iloc[-1])/float(series.iloc[-(n+1)])-1), 2)

def main():
    if not SCREEN.exists():
        raise SystemExit('Run US Stock Screener first: results/screening.json is missing')
    old = json.loads(SCREEN.read_text(encoding='utf-8'))
    stocks = old.get('stocks') or []
    symbols = list(dict.fromkeys(str(s.get('Ticker') or '').strip().upper() for s in stocks))
    symbols = [s for s in symbols if s and not s.endswith('.T')]
    if not symbols:
        raise SystemExit('US screening JSON has no tickers')
    prices = bars_for(symbols, '18mo')
    if len(prices) < max(50, len(symbols)*.4):
        raise SystemExit('Too few valid US histories; refusing to replace prior market_dashboard.json')
    up=down=flat=high52=low52=above50=eligible52=0
    dates=[]
    for symbol, df in prices.items():
        c=df.Close; prev=float(c.iloc[-2]); now=float(c.iloc[-1]);
        if now>prev: up+=1
        elif now<prev: down+=1
        else: flat+=1
        if len(c)>=50 and now>float(c.tail(50).mean()): above50+=1
        if len(df)>=252:
            eligible52+=1
            window=df.tail(252)
            if now>=float(window.High.max())*.999: high52+=1
            if now<=float(window.Low.min())*1.001: low52+=1
        dates.append(str(df.index[-1].date()))
    sectors = bars_for(list(SECTORS), '6mo')
    sector_rows=[]
    for symbol, label in SECTORS.items():
        if symbol not in sectors: continue
        c=sectors[symbol].Close
        sector_rows.append({'ticker':symbol,'name':label,'week_pct':perf(c,5),'month_pct':perf(c,21)})
    sector_rows.sort(key=lambda x: x['week_pct'] if x['week_pct'] is not None else -999, reverse=True)
    rank=sorted((s for s in stocks if isinstance(s.get('RS'), (int,float))),
                key=lambda x: float(x['RS']), reverse=True)
    rs_rows=[{'ticker':s.get('Ticker'), 'name':s.get('Name') or s.get('Company') or s.get('Company_Name') or '',
              'rs':round(float(s['RS'])), 'price':s.get('Price'),
              'trend_pass':bool(s.get('Trend_PASS'))}
             for s in rank[:20]]
    output={
        'updated_at':datetime.now(timezone.utc).isoformat(),
        'price_date':max(dates) if dates else None,
        'universe_label':'現在の米国株スクリーナー対象銘柄',
        'requested_count':len(symbols),'priced_count':len(prices),
        'breadth':{'advancing':up,'declining':down,'unchanged':flat,
                   'above_sma50':above50,'above_sma50_pct':round(100*above50/len(prices),1)},
        'high_low':{'high_52w':high52,'low_52w':low52,'eligible':eligible52},
        'sectors':sector_rows,
        'rs_rankings':rs_rows,
        'market_regime':old.get('market_regime') or old.get('market') or old.get('market_environment'),
        'methodology': 'Adjusted daily prices; last 252 available market sessions. 52-week figures use close near extrema of intraday adjusted high/low. Sector returns from ETFs, 5 and 21 sessions. RS from the current US screening universe.'
    }
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(output, ensure_ascii=False, indent=2, default=str), encoding='utf-8')
    print('Wrote',OUT, 'stocks',len(prices), 'sector ETFs',len(sector_rows))

if __name__ == '__main__': main()
