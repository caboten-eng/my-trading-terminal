
import yfinance as yf
import pandas as pd
from pathlib import Path

# 標準モード：既存のTT/VCP/PASSは変更せず、SETUP_PASSを追加
SCREEN_MODE = "standard"
RS_MIN = 70
PIVOT_LOOKBACK = 15
PIVOT_MAX_DISTANCE_PCT = 5.0
VOLUME_DRY_MAX_RATIO = 0.70

def one_column(df, name):
    series = df[name]
    return series.iloc[:, 0] if isinstance(series, pd.DataFrame) else series

def vcp_stage(score):
    """既存VCP_SCOREを分類。形状を保証するものではない。"""
    return {0: "対象外", 1: "形成初期", 2: "形成中", 3: "完成候補"}.get(int(score), "対象外")

def setup_metrics(df):
    # Pivot: 当日を含めず、直前15営業日の高値を採用
    high = one_column(df, "High").dropna()
    close = one_column(df, "Close").dropna()
    volume = one_column(df, "Volume").dropna()
    if len(high) < PIVOT_LOOKBACK + 1 or len(close) < 1 or len(volume) < 50:
        return None, None, False, False
    pivot = float(high.iloc[-(PIVOT_LOOKBACK + 1):-1].max())
    price = float(close.iloc[-1])
    avg5 = float(volume.iloc[-5:].mean())
    avg50 = float(volume.iloc[-50:].mean())
    distance_pct = (pivot / price - 1) * 100 if price > 0 else None
    dry_ratio = avg5 / avg50 if avg50 > 0 else None
    # 価格がピボット以下かつ距離が5%以内の未突破候補
    near = distance_pct is not None and 0 <= distance_pct <= PIVOT_MAX_DISTANCE_PCT
    dry = dry_ratio is not None and dry_ratio <= VOLUME_DRY_MAX_RATIO
    return distance_pct, dry_ratio, near, dry

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


# VCP（値動きの収縮）候補を判定
def detect_vcp(df):
    high = df["High"]
    low = df["Low"]
    volume = df["Volume"]

    # yfinanceの複数階層カラムに対応
    if isinstance(high, pd.DataFrame):
        high = high.iloc[:, 0]
    if isinstance(low, pd.DataFrame):
        low = low.iloc[:, 0]
    if isinstance(volume, pd.DataFrame):
        volume = volume.iloc[:, 0]

    if len(df) < 252:
        return 0, False

    # 直近60営業日を20日ずつ3区間に分割
    ranges = []

    for start, end in [(-60, -40), (-40, -20), (-20, None)]:
        h = float(high.iloc[start:end].max())
        l = float(low.iloc[start:end].min())

        if l <= 0:
            return 0, False

        ranges.append((h - l) / h)

    # 値幅が段階的に縮小
    contraction = ranges[0] > ranges[1] > ranges[2]

    # 直近10日とその前20日の出来高を比較
    recent_volume = float(volume.iloc[-10:].mean())
    previous_volume = float(volume.iloc[-30:-10].mean())

    volume_dryup = (
        previous_volume > 0
        and recent_volume < previous_volume * 0.8
    )

    # 52週高値から15%以内
    high52 = float(high.iloc[-252:].max())
    current_price = float(df["Close"].iloc[-1].iloc[0]) \
        if isinstance(df["Close"], pd.DataFrame) \
        else float(df["Close"].iloc[-1])

    near_high = current_price >= high52 * 0.85

    # 3条件を点数化
    score = (
        int(contraction)
        + int(volume_dryup)
        + int(near_high)
    )

    # 3点ならVCP候補
    candidate = score == 3

    return score, candidate

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
        vcp_score, vcp_candidate = detect_vcp(df)
        pivot_distance, volume_ratio, pivot_near, volume_dry = setup_metrics(df)
        results.append({
            "Ticker": ticker,
            "Price": round(price, 2),
            "MA50": round(float(ma50.iloc[-1]), 2),
            "MA150": round(float(ma150.iloc[-1]), 2),
            "MA200": round(float(ma200.iloc[-1]), 2),
            "Trend_PASS": bool(all(checks.values())),
            "RS_raw": rs_raw,
            "VCP_SCORE": vcp_score,
            "VCP_STAGE": vcp_stage(vcp_score),
            "VCP_CANDIDATE": bool(vcp_candidate),
            "Pivot_Distance_Pct": round(pivot_distance, 2) if pivot_distance is not None else None,
            "Volume5_50_Ratio": round(volume_ratio, 3) if volume_ratio is not None else None,
            "Pivot_NEAR": bool(pivot_near),
            "Volume_DRY": bool(volume_dry)
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
output["RS_PASS"] = output["RS"] >= RS_MIN

# トレンド条件とRS条件を統合
output["PASS"] = (
    output["Trend_PASS"] & output["RS_PASS"]
)

# 標準モードの追加判定。旧PASSは互換性のためそのまま保持。
output["SETUP_PASS"] = (
    output["Trend_PASS"]
    & output["RS_PASS"]
    & output["VCP_CANDIDATE"]
    & output["Pivot_NEAR"]
    & output["Volume_DRY"]
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
    "setup_passed": int(output["SETUP_PASS"].sum()),
    "vcp_stage_counts": {label: int((output["VCP_STAGE"] == label).sum()) for label in ["対象外", "形成初期", "形成中", "完成候補"]},
    "screen_mode": SCREEN_MODE,
    "settings": {"rs_min": RS_MIN, "pivot_max_distance_pct": PIVOT_MAX_DISTANCE_PCT, "volume_dry_max_ratio": VOLUME_DRY_MAX_RATIO},
    "stocks": output.to_dict(orient="records")
}

with open("results/screening.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print("\nスクリーニング完了")
print(f"対象銘柄: {data['total']}")
print(f"合格銘柄: {data['passed']}")
print(f"VCP候補: {int(output['VCP_CANDIDATE'].sum())}銘柄")
print(f"セットアップ合格: {int(output['SETUP_PASS'].sum())}銘柄")
print(output)

print("VCP段階別:", data["vcp_stage_counts"])
