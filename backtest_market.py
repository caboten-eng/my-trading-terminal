#!/usr/bin/env python3
"""米国市場の独自4点地合いモデルの過去検証（Minervini公式モデルではない）。

2020年以降の判定を、各日の確定日足と当日までの情報のみで再計算する。
注意: 銘柄母集団は現在のS&P500構成銘柄＋既存ウォッチリストなので生存者バイアスあり。
"""
from __future__ import annotations

import ast
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd
import yfinance as yf

START = "2018-10-01"  # 2020年の判定に必要な過去252日以上の余裕
EVAL_START = "2020-01-01"
BATCH_SIZE = 35
OUT = Path("results/market_backtest")
OUT.mkdir(parents=True, exist_ok=True)


def ticker_universe():
    # 現行スクリーナー内の監視銘柄を変更せず取得（screener.py は実行しない）
    extras = []
    file = Path("screener.py")
    if file.exists():
        tree = ast.parse(file.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(
                isinstance(target, ast.Name) and target.id == "TICKERS" for target in node.targets
            ):
                try:
                    extras = ast.literal_eval(node.value)
                except (ValueError, SyntaxError):
                    pass
                break

    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; MarketBacktest/1.0)"})
    with urlopen(req, timeout=40) as response:
        tables = pd.read_html(io.BytesIO(response.read()))
    sp = tables[0]["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
    tickers = list(dict.fromkeys(extras + sp))
    return [x for x in tickers if isinstance(x, str) and x.strip() and x not in ("SPY", "QQQ")]


def adjusted_closes(symbols, start=START):
    result = {}
    for batch_num, offset in enumerate(range(0, len(symbols), BATCH_SIZE), 1):
        batch = symbols[offset:offset+BATCH_SIZE]
        print(f"データ取得 {offset+1}-{min(offset+BATCH_SIZE, len(symbols))}/{len(symbols)}", flush=True)
        try:
            raw = yf.download(batch, start=start, auto_adjust=True, progress=False,
                              group_by="column", threads=True, timeout=30)
            if isinstance(raw.columns, pd.MultiIndex):
                closes = raw["Close"]
            else:
                closes = pd.DataFrame({batch[0]: raw["Close"]})
            for t in batch:
                if t in closes:
                    s = pd.to_numeric(closes[t], errors="coerce").dropna()
                    if len(s) >= 253:
                        result[t] = s
        except Exception as exc:
            print(f"バッチ取得失敗: {exc}")
        if batch_num % 3 == 0:
            time.sleep(1)
    if not result:
        raise RuntimeError("株価データを取得できませんでした")
    return pd.DataFrame(result).sort_index()


def compute(dates: pd.DatetimeIndex, prices: pd.DataFrame, spy: pd.Series, qqq: pd.Series):
    # 比較対象をSPYの米国取引日に統一。欠損値を過去価格で補完しない。
    p = prices.reindex(dates)
    ma50 = p.rolling(50, min_periods=50).mean()
    ma150 = p.rolling(150, min_periods=150).mean()
    ma200 = p.rolling(200, min_periods=200).mean()
    high52 = p.rolling(252, min_periods=252).max()  # 現行モデル同様、終値ベース
    low52 = p.rolling(252, min_periods=252).min()
    price = p

    # 現行screener.py のRS_rawと一致する区間（63/126/189/252日前）
    p3, p6, p9, p12 = (p.shift(n) for n in (63, 126, 189, 252))
    raw_rs = (0.4*(p/p3-1) + 0.2*(p3/p6-1)
              + 0.2*(p6/p9-1) + 0.2*(p9/p12-1))
    rs = (raw_rs.rank(axis=1, pct=True)*99).round().clip(1, 99)

    trend = ((price > ma50) & (price > ma150) & (price > ma200)
             & (ma50 > ma150) & (ma150 > ma200)
             & (ma200 > ma200.shift(21))
             & (price >= high52*0.75) & (price >= low52*1.30))

    # 未上場/履歴不足を分母に含めない。全項目が算出できる銘柄だけを母集団とする。
    valid = price.notna() & ma50.notna() & ma150.notna() & ma200.notna() \
        & ma200.shift(21).notna() & high52.notna() & low52.notna() & raw_rs.notna()
    universe_count = valid.sum(axis=1)
    breadth = ((price > ma50) & valid).sum(axis=1).div(universe_count.where(universe_count > 0))*100
    leader_count = (trend & (rs >= 70) & valid).sum(axis=1)
    leader_pct = leader_count.div(universe_count.where(universe_count > 0))*100

    spy = spy.reindex(dates)
    qqq = qqq.reindex(dates)
    spy50 = spy.rolling(50, min_periods=50).mean()
    spy200 = spy.rolling(200, min_periods=200).mean()
    qqq50 = qqq.rolling(50, min_periods=50).mean()
    qqq200 = qqq.rolling(200, min_periods=200).mean()
    spy_up = (spy > spy50) & (spy50 > spy200)
    qqq_up = (qqq > qqq50) & (qqq50 > qqq200)
    score = spy_up.astype(int) + qqq_up.astype(int) + (breadth >= 55).astype(int) + (leader_pct >= 10).astype(int)
    regime = pd.Series(np.select([score >= 3, score == 2], ["強気", "中立"], default="警戒"), index=dates)
    # 現行モデルの計算条件が不足した日は状態を記録しない
    coverage_ok = (spy200.notna() & qqq200.notna() & (universe_count >= 100)
                   & (dates >= pd.Timestamp(EVAL_START)))
    out = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"), "spy_close": spy.values, "qqq_close": qqq.values,
        "spy_ma50": spy50.values, "spy_ma200": spy200.values,
        "qqq_ma50": qqq50.values, "qqq_ma200": qqq200.values,
        "spy_trend_point": spy_up.astype(int).values,
        "qqq_trend_point": qqq_up.astype(int).values,
        "breadth_above_ma50_pct": breadth.values,
        "leader_count": leader_count.values,
        "leader_pct": leader_pct.values,
        "universe_count": universe_count.values,
        "score": score.values, "regime": regime.values,
    }, index=dates)
    out = out.loc[coverage_ok].copy()
    if out.empty:
        raise RuntimeError("判定可能な日足がありません（データ取得状況を確認してください）")
    out["spy_return_20d_pct"] = (out["spy_close"].shift(-20)/out["spy_close"]-1)*100
    out["spy_return_60d_pct"] = (out["spy_close"].shift(-60)/out["spy_close"]-1)*100
    return out


def create_report(daily, tickers, valid_count):
    change = daily["regime"].ne(daily["regime"].shift(1))
    transitions = daily.loc[change].copy()
    transitions["prev_regime"] = daily["regime"].shift(1).loc[transitions.index]
    # 初日の状態は「切替」ではない
    transitions = transitions.iloc[1:].copy()
    transitions["transition"] = transitions["prev_regime"] + " → " + transitions["regime"]
    transitions.to_csv(OUT / "market_backtest_transitions.csv", index=False, encoding="utf-8-sig")
    caution = transitions.loc[transitions["regime"] == "警戒"]
    valid20 = caution["spy_return_20d_pct"].dropna()
    valid60 = caution["spy_return_60d_pct"].dropna()
    first_date = daily["date"].iloc[0]
    last_date = daily["date"].iloc[-1]
    lines = [
        "# 米国市場地合い判定バックテスト", "",
        f"集計期間: {first_date} ～ {last_date}",
        f"現在の銘柄リスト: {len(tickers)}件 / 日足取得成功: {valid_count}件", "",
        "## 判定ルール（実運用と同じ4項目）", "",
        "- SPY: 終値 > SMA50 > SMA200 なら1点",
        "- QQQ: 終値 > SMA50 > SMA200 なら1点",
        "- Breadth: 対象銘柄の50日線上の割合 ≥55%なら1点",
        "- 先導株: Trend Template合格かつRS70以上の割合 ≥10%なら1点",
        "- 合計0–1点=警戒、2点=中立、3–4点=強気", "",
        "## 集計", "",
        f"- 警戒へ移行: {len(caution)}回",
        f"- 警戒点灯後20営業日のSPY平均騰落率: {valid20.mean():+.2f}% ({len(valid20)}件)" if len(valid20) else "- 20営業日の評価可能データなし",
        f"- 警戒点灯後60営業日のSPY平均騰落率: {valid60.mean():+.2f}% ({len(valid60)}件)" if len(valid60) else "- 60営業日の評価可能データなし",
        "", "## 警戒へ切替えた日", "",
        "| 日付 | 前状態 | スコア | SPY | 20営業日後 | 60営業日後 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for _, r in caution.iterrows():
        def f(v):
            return f"{v:+.2f}%" if pd.notna(v) else "未確定"
        lines.append(f"| {r['date']} | {r['prev_regime']} | {int(r['score'])} | "
                     f"${r['spy_close']:.2f} | {f(r['spy_return_20d_pct'])} | {f(r['spy_return_60d_pct'])} |")
    lines += ["", "## 重要な制約", "",
              "- **生存者バイアス**: 現在のS&P500構成銘柄＋現行監視リストで過去を計算。除外済み銘柄は含まれない。",
              "- **母集団の変動**: 当時取得可能な252営業日以上の履歴がある銘柄だけで、当日のRSとBreadthを計算。実運用の母集団と必ずしも一致しない。",
              "- **将来情報なし**: シグナルは当日までの確定終値のみ。20/60営業日後騰落率は評価用で判定には使用しない。",
              "- **簡易検証**: 配当調整後データ・取得可能な銘柄に依存。売買コストやポジション管理は含まない。",
              "- **独自モデル**: Minervini氏の非公開マーケットリスクモデルの再現ではない。",
              "- **S&P500の下落開始からのリード日数**は定義が主観的なので自動的には算出していない。",
              ""]
    (OUT / "market_backtest_report.md").write_text("\n".join(lines), encoding="utf-8")
    return transitions


def main():
    tickers = ticker_universe()
    print(f"対象リスト {len(tickers)}銘柄", flush=True)
    prices = adjusted_closes(tickers)
    indexes = adjusted_closes(["SPY", "QQQ"])
    if "SPY" not in indexes or "QQQ" not in indexes:
        raise RuntimeError("SPY/QQQを取得できませんでした")
    trading_days = indexes["SPY"].dropna().index
    daily = compute(trading_days, prices, indexes["SPY"], indexes["QQQ"])
    daily.to_csv(OUT / "market_backtest_daily.csv", index=False, encoding="utf-8-sig")
    transitions = create_report(daily, tickers, prices.shape[1])
    print(f"完了: {len(daily)}営業日、状態変更 {len(transitions)}回、警戒切替 {(transitions['regime']=='警戒').sum()}回", flush=True)
    print(f"出力先: {OUT}")


if __name__ == "__main__":
    main()
