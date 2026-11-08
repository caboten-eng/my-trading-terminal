
import yfinance as yf
import pandas as pd
from pathlib import Path

# 標準モード：既存のTT/VCP/PASSは変更せず、SETUP_PASSを追加
SCREEN_MODE = "simple_vcp_jp"
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

def vcp_pullback_depths(df):
    """直近60日を古い順に20日×3区間とし、各区間の最大ピーク→その後の安値の下落率を測定。

    ※固定区間ベースの簡易近似。スイング高安の自動認識ではない。
    高値は当日までの累積最大、安値はその日の日中安値を使い、未来の高値を参照しない。
    """
    high = one_column(df, "High")
    low = one_column(df, "Low")
    if len(high) < 60 or len(low) < 60:
        return None, None, None, False, None
    depths = []
    for start, end in [(-60, -40), (-40, -20), (-20, None)]:
        h = high.iloc[start:end].astype(float)
        l = low.iloc[start:end].astype(float)
        if len(h) != 20 or len(l) != 20 or h.isna().any() or l.isna().any() or (h <= 0).any() or (l <= 0).any():
            return None, None, None, False, None
        peaks = h.cummax()
        drop = ((peaks - l) / peaks * 100).clip(lower=0)
        depths.append(float(drop.max()))
    # ゼロ近辺はノイズに敏感なので各区間1%以上の下落を要求
    shrinking = all(d >= 1.0 for d in depths) and depths[0] > depths[1] > depths[2]
    ratio = depths[2] / depths[0] if depths[0] > 0 else None
    return *depths, bool(shrinking), ratio
def detect_swing_vcp(df):
    """
    スイング高値・安値からVCPの収縮を検出する。

    ・左右3営業日で確定した高値・安値を使用
    ・直近120営業日の値動きを分析
    ・直近3回の押し目の深さを比較
    ・最後の安値が直近25営業日以内であること
    ・最後の押し目で出来高が減少していること

    既存のVCP_SCOREやSETUP_PASSは変更しない。
    """
    import numpy as np

    high = one_column(df, "High")
    low = one_column(df, "Low")
    volume = one_column(df, "Volume")

    if len(df) < 120:
        return None

    high = high.iloc[-120:].to_numpy(dtype=float)
    low = low.iloc[-120:].to_numpy(dtype=float)
    volume = volume.iloc[-120:].to_numpy(dtype=float)

    if (
        not np.isfinite(high).all()
        or not np.isfinite(low).all()
        or not np.isfinite(volume).all()
        or np.any(high <= 0)
        or np.any(low <= 0)
    ):
        return None

    window = 3
    swings = []

    # スイング高値・安値を確定
    for i in range(window, len(high) - window):

        h = high[i]
        l = low[i]

        is_peak = (
            h > max(high[i-window:i])
            and h >= max(high[i+1:i+window+1])
        )

        is_trough = (
            l < min(low[i-window:i])
            and l <= min(low[i+1:i+window+1])
        )

        # 同じ足が高値・安値の両方なら除外
        if is_peak == is_trough:
            continue

        kind = "H" if is_peak else "L"
        price = h if is_peak else l

        # 同種類の山・谷は極値だけ残す
        if swings and swings[-1][0] == kind:

            old_kind, old_i, old_price = swings[-1]

            better = (
                price > old_price
                if kind == "H"
                else price < old_price
            )

            if better:
                swings[-1] = (kind, i, price)

        else:
            swings.append((kind, i, price))

    contractions = []

    # 高値 → 安値の組を抽出
    for a, b in zip(swings, swings[1:]):

        if a[0] != "H" or b[0] != "L":
            continue

        peak = a[2]
        trough = b[2]

        if peak <= 0 or trough >= peak:
            continue

        depth = (peak - trough) / peak * 100

        # 小さすぎるノイズや極端な下落は除外
        if not 2 <= depth <= 45:
            continue

        contractions.append({
            "peak_index": a[1],
            "low_index": b[1],
            "depth": depth
        })

    if len(contractions) < 3:
        return {
            "score": 0,
            "candidate": False,
            "depths": [],
            "reason": "押し目が3回未満"
        }

    recent = contractions[-3:]
    depths = [x["depth"] for x in recent]

    # 3回の押し目が順次浅くなる
    shrinking = (
        depths[0] > depths[1] > depths[2]
    )

    # 最後の谷が最近形成されている
    days_since_low = (
        len(high) - 1 - recent[-1]["low_index"]
    )

    recent_low = days_since_low <= 25

    # 最後の押し目の出来高と前の押し目を比較
    def avg_volume(c):
        start = c["peak_index"]
        end = c["low_index"] + 1
        return float(np.mean(volume[start:end]))

    previous_vol = avg_volume(recent[-2])
    latest_vol = avg_volume(recent[-1])

    volume_shrinking = (
        previous_vol > 0
        and latest_vol < previous_vol * 0.85
    )

    # 最後の押し目が最初の60%以下
    tight = depths[2] <= depths[0] * 0.6

    score = sum([
        shrinking,
        recent_low,
        volume_shrinking,
        tight
    ])

    return {
        "score": int(score),
        "candidate": bool(score == 4),
        "depths": [round(x, 2) for x in depths],
        "days_since_low": int(days_since_low),
        "volume_shrinking": bool(volume_shrinking),
        "shrinking": bool(shrinking),
        "tight": bool(tight)
    }
def detect_base_breakout(df):
    """過去の固定期間の高値を基準にした暫定ベース判定。"""
    import numpy as np

    if len(df) < 120:
        return None

    high = one_column(df, "High").astype(float)
    low = one_column(df, "Low").astype(float)
    close = one_column(df, "Close").astype(float)
    volume = one_column(df, "Volume").astype(float)

    h = high.to_numpy()
    l = low.to_numpy()
    c = close.to_numpy()
    v = volume.to_numpy()

    if not all(np.isfinite(a[-120:]).all() for a in (h, l, c, v)):
        return None
    if np.any(h[-120:] <= 0) or np.any(l[-120:] <= 0):
        return None

    # 最近10日を除く60営業日で基準高値を固定
    reference = h[-70:-10]
    pivot_idx_local = int(np.argmax(reference))
    pivot_idx = len(h) - 70 + pivot_idx_local
    pivot = float(reference[pivot_idx_local])

    current = float(c[-1])
    previous = float(c[-2])
    if pivot <= 0 or current <= 0:
        return None

    distance = (pivot - current) / pivot * 100
    base_low = float(np.min(l[pivot_idx + 1:]))
    depth = (pivot - base_low) / pivot * 100

    avg_volume = float(np.mean(v[-51:-1]))
    volume_ratio = (
        float(v[-1] / avg_volume)
        if avg_volume > 0 else 0.0
    )

    days_since_peak = len(h) - 1 - pivot_idx

    # 調整幅3〜35%、高値形成から10営業日以上
    valid_base = (
        3 <= depth <= 35
        and days_since_peak >= 10
    )

    stage = "FORMING"

    if valid_base:
        if previous <= pivot < current:
            stage = (
                "BREAKOUT"
                if volume_ratio >= 1.5
                else "BREAKOUT_LOW_VOLUME"
            )
        elif distance < -5:
            stage = "EXTENDED"
        elif distance < 0:
            stage = "ABOVE_PIVOT"
        elif distance <= 5:
            stage = "READY"

    return {
        "pivot": round(pivot, 2),
        "distance_pct": round(distance, 2),
        "base_depth_pct": round(depth, 2),
        "volume_ratio": round(volume_ratio, 2),
        "stage": stage,
        "valid_base": bool(valid_base),
        "days_since_peak": int(days_since_peak),
    }
    
def detect_resistance_pivot(df):
    """
    複数のスイング高値から抵抗帯Pivotを検出する。

    ・直近90営業日を分析
    ・左右3営業日で確定した高値を使用
    ・高値同士の価格差は3%以内
    ・高値の間隔は5営業日以上
    ・現在株価より上の抵抗帯を優先
    ・すべて突破済みなら最も近い下の抵抗帯を採用

    既存のBASE_STAGEやSETUP_PASSは変更しない。
    """
    import numpy as np

    if len(df) < 100:
        return None

    high = one_column(df, "High").iloc[-100:]
    close = one_column(df, "Close").iloc[-100:]

    h = high.to_numpy(dtype=float)
    c = close.to_numpy(dtype=float)

    if (
        not np.isfinite(h).all()
        or not np.isfinite(c).all()
        or np.any(h <= 0)
        or c[-1] <= 0
    ):
        return None

    current = float(c[-1])
    n = len(h)

    peaks = []

    # 確定済みスイング高値を検出
    for i in range(10, n - 3):
        if (
            h[i] > max(h[i-3:i])
            and h[i] >= max(h[i+1:i+4])
        ):
            peaks.append((i, float(h[i])))

    candidates = []

    # すべての高値ペアを調べる
    for j in range(1, len(peaks)):
        recent_i, recent_price = peaks[j]

        for k in range(j):
            old_i, old_price = peaks[k]
            # 同じベース内の高値候補に限定（試験版）
            peak_gap = recent_i - old_i
            if not 5 <= peak_gap <= 40:
                continue

            # 直近20営業日以内に試された抵抗帯だけ採用
            days_since_recent_peak = n - 1 - recent_i
            if days_since_recent_peak > 20:
                continue

            difference = (
                abs(recent_price - old_price)
                / max(recent_price, old_price)
                * 100
            )

            if difference > 3:
                continue

            pivot = max(recent_price, old_price)
            # 2つの高値の間の調整幅を計算
            low_between = float(
                one_column(df, "Low")
                .iloc[-100:]
                .iloc[old_i:recent_i + 1]
                .min()
            )

            base_depth = (
                (pivot - low_between) / pivot * 100
            )

            # 調整幅3〜25%のベースに限定（試験版）
            if not 3 <= base_depth <= 25:
                continue

            candidates.append({
                "pivot": float(pivot),
                "first_i": old_i,
                "last_i": recent_i,
                "difference": float(difference)
            })

    if not candidates:
        return None

    # 現在株価以上の抵抗帯を優先
    above = [
        x for x in candidates
        if x["pivot"] >= current
    ]

    if above:
        # 現在株価に最も近い上値抵抗帯
        best = min(
            above,
            key=lambda x: (
                x["pivot"] - current,
                -x["last_i"]
            )
        )    
    else:
        # 上値抵抗帯がない場合はREADY候補なし
        return None

    pivot = best["pivot"]

    return {
        "pivot": round(pivot, 2),
        "first_peak_days_ago": int(
            n - 1 - best["first_i"]
        ),
        "last_peak_days_ago": int(
            n - 1 - best["last_i"]
        ),
        "peak_difference_pct": round(
            best["difference"], 2
        ),
        "touches": 2,
        "position": (
            "BELOW_PIVOT"
            if current <= pivot
            else "ABOVE_PIVOT"
        )
    }
def detect_single_peak_pivot(df):
    import numpy as np

    if len(df) < 100:
        return None

    h = one_column(df, "High").iloc[-100:].to_numpy(dtype=float)
    c = one_column(df, "Close").iloc[-100:].to_numpy(dtype=float)

    if not np.isfinite(h).all() or not np.isfinite(c).all():
        return None

    current = float(c[-1])
    if current <= 0:
        return None

    n = len(h)
    candidates = []

    for i in range(n - 21, n - 3):
        if h[i] > max(h[i-3:i]) and h[i] >= max(h[i+1:i+4]):
            pivot = float(h[i])
            distance = (pivot - current) / pivot * 100

            if 0 <= distance <= 5:
                candidates.append((pivot, n - 1 - i))

    if not candidates:
        return None

    pivot, days = min(candidates, key=lambda x: x[0])

    return {
        "pivot": round(pivot, 2),
        "distance_pct": round((pivot - current) / pivot * 100, 2),
        "days_since_peak": int(days),
        "type": "SINGLE_PEAK"
    }
def detect_minervini_swings(df):
    import numpy as np

    window = 120

    if len(df) < window:
        return None

    data = df.iloc[-window:]

    high = one_column(data, "High").to_numpy(dtype=float)
    low = one_column(data, "Low").to_numpy(dtype=float)

    if not np.isfinite(high).all():
        return None
    if not np.isfinite(low).all():
        return None

    peaks = []
    troughs = []

    for i in range(3, len(data) - 3):

        if (
            high[i] > max(high[i-3:i])
            and high[i] >= max(high[i+1:i+4])
        ):
            peaks.append({
                "index": i,
                "price": round(float(high[i]), 2),
                "days_ago": len(data) - 1 - i
            })

        if (
            low[i] < min(low[i-3:i])
            and low[i] <= min(low[i+1:i+4])
        ):
            troughs.append({
                "index": i,
                "price": round(float(low[i]), 2),
                "days_ago": len(data) - 1 - i
            })

    return {
        "peaks": peaks,
        "troughs": troughs
    }
def calculate_minervini_contractions(df):
    swings = detect_minervini_swings(df)

    if swings is None:
        return None

    peaks = swings["peaks"]
    troughs = swings["troughs"]

    if len(peaks) < 2:
        return None

    contractions = []

    for i in range(len(peaks) - 1):
        first = peaks[i]
        second = peaks[i + 1]

        between = [
            t for t in troughs
            if first["index"] < t["index"] < second["index"]
        ]

        if not between:
            continue

        bottom = min(between, key=lambda x: x["price"])
        high = first["price"]
        low = bottom["price"]

        if high <= 0 or low >= high:
            continue

        depth = (high - low) / high * 100

        contractions.append({
            "high": high,
            "low": low,
            "depth_pct": round(depth, 2),
            "peak_days_ago": first["days_ago"],
            "trough_days_ago": bottom["days_ago"]
        })

    if len(contractions) < 2:
        return None

    recent = contractions[-4:]
    depths = [x["depth_pct"] for x in recent]

    shrinking = all(
        depths[i] > depths[i + 1]
        for i in range(len(depths) - 1)
    )

    return {
        "contraction_count": len(recent),
        "contractions": recent,
        "depths_pct": depths,
        "shrinking": bool(shrinking)
    }


def detect_provisional_pivot(df):
    import numpy as np

    if len(df) < 21:
        return None

    high = one_column(df, "High").iloc[-21:-1].to_numpy(dtype=float)
    close = float(one_column(df, "Close").iloc[-1])

    if not np.isfinite(high).all() or not np.isfinite(close):
        return None

    pivot = float(np.max(high))

    if pivot <= 0:
        return None

    distance = (pivot - close) / pivot * 100

    if distance < 0:
        stage = "ABOVE_PIVOT"
    elif distance <= 5:
        stage = "NEAR_PIVOT"
    else:
        stage = "FORMING"

    return {
        "pivot": round(pivot, 2),
        "distance_pct": round(distance, 2),
        "stage": stage
    }
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

# JPX公式の東証上場銘柄一覧。ETF・REIT・外国株式は含めない。
def get_prime_tickers():
    from io import BytesIO
    from urllib.request import Request, urlopen
    import re

    url = "https://www.jpx.co.jp/markets/statistics-equities/misc/tvdivq0000001vg2-att/data_j.xlsx"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=60) as response:
        content = response.read()
    listing = pd.read_excel(BytesIO(content), dtype=str, engine="openpyxl")
    required = {"コード", "市場・商品区分"}
    if not required.issubset(set(listing.columns)):
        raise RuntimeError(f"JPXの列形式が変更されました: {list(listing.columns)}")
    prime = listing[listing["市場・商品区分"].astype(str).str.strip() == "プライム（内国株式）"]
    codes = prime["コード"].dropna().astype(str).str.strip().tolist()
    codes = list(dict.fromkeys(c for c in codes if re.fullmatch(r"[0-9]{4}|[0-9]{3}[A-Z]", c)))
    if len(codes) < 1000:
        raise RuntimeError(f"東証プライム銘柄が少なすぎます（{len(codes)}）。JPXの形式を確認してください")
    return codes

# JPXの銘柄リストが取得できない場合は古い結果を上書きせず停止
CODES = get_prime_tickers()
TICKERS = [code + ".T" for code in CODES]
print(f"東証プライム: {len(TICKERS)}銘柄（JPX月次リスト）")

import json
import math
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
        # 日本市場の未確定日足を除外（15:30引け、16:00まで保守的に除外）
        from datetime import datetime, time
        from zoneinfo import ZoneInfo
        jst_now = datetime.now(ZoneInfo("Asia/Tokyo"))
        last_bar_date = df.index[-1].date()
        if last_bar_date == jst_now.date() and jst_now.time() < time(16, 0):
            df = df.iloc[:-1].copy()
            print(f"{ticker}: 未確定日足 {last_bar_date} を除外")

        if df.empty:
            print(f"{ticker}: 確定日足なし")
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
        pb1, pb2, pb3, pb_shrinking, pb_ratio = vcp_pullback_depths(df)
        swing_vcp = detect_swing_vcp(df)
        
        # Swing VCPのブレイクアウト準備判定
        # 既存のVCPスコア・SETUP判定は変更しない

        swing_depths = (
            swing_vcp.get("depths", [])
            if swing_vcp else []
        )

        # 最後の押し目が10%以下
        swing_final_tight = (
            len(swing_depths) == 3
            and 2 <= swing_depths[-1] <= 10
        )

        # 直近10営業日の高値・安値による値幅
        recent_high = float(
            one_column(df, "High").iloc[-10:].max()
        )
        recent_low = float(
            one_column(df, "Low").iloc[-10:].min()
        )

        swing_range_10d = (
            (recent_high - recent_low) / recent_high * 100
            if recent_high > 0 else None
        )

        # 直近10営業日の値幅10%以下
        swing_range_tight = (
            swing_range_10d is not None
            and swing_range_10d <= 10
        )

        # 直近15日高値をピボットとして5%以内
        swing_pivot_ready = bool(pivot_near)

        # 4点満点 + 追加3条件を満たす
        swing_vcp_ready = bool(
            swing_vcp
            and swing_vcp.get("candidate", False)
            and swing_final_tight
            and swing_range_tight
            and swing_pivot_ready
        )
        # ベース上限を使ったピボット判定
        base_breakout = detect_base_breakout(df)

        # 複数スイング高値による抵抗帯判定
        resistance = detect_resistance_pivot(df)
        single_peak = detect_single_peak_pivot(df)
        provisional = detect_provisional_pivot(df)
        minervini_vcp = calculate_minervini_contractions(df)
        unified = resistance if resistance is not None else single_peak
        unified_pivot = unified["pivot"] if unified else None
        unified_pivot_type = (
            "MULTI_PEAK" if resistance is not None
            else "SINGLE_PEAK" if single_peak is not None
            else None
        )

        resistance_pivot = (
            resistance.get("pivot") if resistance else None
        )

        resistance_distance = (
            (resistance_pivot - price) / resistance_pivot * 100
            if resistance_pivot is not None and resistance_pivot > 0
            else None
        )

        resistance_ready = bool(
            resistance_distance is not None
            and 0 <= resistance_distance <= 5
            and swing_vcp_ready
            and all(checks.values())
        )

        # 監視用の簡易VCP：ミネルヴィニの裁量VCPとは別の近似指標。
        # 「過去の大きな上昇」ではなく、直近の収縮と高値への接近を見る。
        close10 = one_column(df, "Close").astype(float)
        high10 = one_column(df, "High").astype(float)
        low10 = one_column(df, "Low").astype(float)
        vol10 = one_column(df, "Volume").astype(float)

        # 同じ日数で比較することで、5日と10日の値幅を直接比較しない。
        recent5_range = (
            float(high10.iloc[-5:].max() - low10.iloc[-5:].min())
            / float(high10.iloc[-5:].max()) * 100
        )
        previous5_range = (
            float(high10.iloc[-10:-5].max() - low10.iloc[-10:-5].min())
            / float(high10.iloc[-10:-5].max()) * 100
        )
        recent5_vol = float(vol10.iloc[-5:].mean())
        previous5_vol = float(vol10.iloc[-10:-5].mean())
        simple_range_shrinking = bool(recent5_range < previous5_range)
        simple_volume_shrinking = bool(
            previous5_vol > 0 and recent5_vol < previous5_vol
        )
        # バーが近接しているだけで「正式なVCP」とは扱わない。
        simple_vcp_shape = bool(simple_range_shrinking or simple_volume_shrinking)
        simple_pivot_distance = (
            provisional["distance_pct"] if provisional else None
        )
        simple_pivot_near = bool(
            simple_pivot_distance is not None
            and 0 <= simple_pivot_distance <= 5
        )

        # 過去10営業日で、各日までの直前20日高値を終値で更新したか。
        # VCPの正式なブレイクアウト判定ではなく、参考分類。
        recent_high_break_days = []
        high_series = one_column(df, "High").astype(float)
        close_series = one_column(df, "Close").astype(float)
        for days_ago in range(9, -1, -1):
            day = len(df) - 1 - days_ago
            prior_20_high = float(high_series.iloc[day - 20:day].max())
            if float(close_series.iloc[day]) > prior_20_high:
                recent_high_break_days.append(days_ago)
        recent_20d_high_break = bool(recent_high_break_days)
        last_break_days_ago = (
            recent_high_break_days[-1] if recent_high_break_days else None
        )

        results.append({
            "RECENT_20D_HIGH_BREAK": recent_20d_high_break,
            "RECENT_BREAK_DAYS_AGO": last_break_days_ago,
            "SIMPLE_RANGE_SHRINK": simple_range_shrinking,
            "SIMPLE_VOLUME_SHRINK": simple_volume_shrinking,
            "SIMPLE_VCP_SHAPE": simple_vcp_shape,
            "SIMPLE_RANGE_5D_PCT": round(recent5_range, 2),
            "SIMPLE_PREV_RANGE_5D_PCT": round(previous5_range, 2),
            "SIMPLE_VOLUME_5D_RATIO": (
                round(recent5_vol / previous5_vol, 3)
                if previous5_vol > 0 else None
            ),
            "SIMPLE_PIVOT_NEAR": simple_pivot_near,
            "BAR_DATE": df.index[-1].strftime("%Y-%m-%d"),
            "Ticker": ticker.removesuffix(".T"),
            "Market": "JP",
            "YahooSymbol": ticker,
            "Price": round(price, 2),
            "MA50": round(float(ma50.iloc[-1]), 2),
            "MA150": round(float(ma150.iloc[-1]), 2),
            "MA200": round(float(ma200.iloc[-1]), 2),
            "Trend_PASS": bool(all(checks.values())),
            "RS_raw": rs_raw,
            "VCP_SCORE": vcp_score,
            "VCP_STAGE": vcp_stage(vcp_score),
            "VCP_PB1_PCT": round(pb1, 2) if pb1 is not None else None,
            "VCP_PB2_PCT": round(pb2, 2) if pb2 is not None else None,
            "VCP_PB3_PCT": round(pb3, 2) if pb3 is not None else None,
            "VCP_PB_SHRINK": bool(pb_shrinking),
            "VCP_PB_RATIO": round(pb_ratio, 3) if pb_ratio is not None else None,
            "VCP_CANDIDATE": bool(vcp_candidate),

            # スイングVCPスコア
            "SWING_VCP_SCORE": (
                swing_vcp.get("score", 0) if swing_vcp else 0
            ),

            # ブレイクアウト準備判定
            "SWING_VCP_READY": bool(swing_vcp_ready),
            "SWING_VCP_FINAL_TIGHT": bool(swing_final_tight),
            "SWING_VCP_RANGE_10D_PCT": (
                round(swing_range_10d, 2)
                if swing_range_10d is not None else None
            ),
            "SWING_VCP_RANGE_TIGHT": bool(swing_range_tight),
            "SWING_VCP_PIVOT_READY": bool(swing_pivot_ready),
            "SWING_VCP_CANDIDATE": (
                swing_vcp.get("candidate", False) if swing_vcp else False
            ),
            "SWING_VCP_DEPTHS": (
                swing_vcp.get("depths", []) if swing_vcp else []
            ),
            "SWING_VCP_SHRINK": (
                swing_vcp.get("shrinking", False) if swing_vcp else False
            ),
            "SWING_VCP_VOLUME_SHRINK": (
                swing_vcp.get("volume_shrinking", False) if swing_vcp else False
            ),
            "SWING_VCP_DAYS_SINCE_LOW": (
                swing_vcp.get("days_since_low") if swing_vcp else None
            ),
            # 抵抗帯Pivotを基準にしたREADY判定
            "RESISTANCE_READY": bool(resistance_ready),

            "RESISTANCE_DISTANCE_PCT": (
                round(resistance_distance, 2)
                if resistance_distance is not None
                else None
            ),            
            "UNIFIED_PIVOT": unified_pivot,
            "MINERVINI_T_COUNT": minervini_vcp["contraction_count"] if minervini_vcp else 0,
            "MINERVINI_T_DEPTHS": minervini_vcp["depths_pct"] if minervini_vcp else [],
            "MINERVINI_LAST_T_DAYS": minervini_vcp["contractions"][-1]["trough_days_ago"] if minervini_vcp and minervini_vcp["contractions"] else None,
            "MINERVINI_SHRINKING": minervini_vcp["shrinking"] if minervini_vcp else False,
            "UNIFIED_PIVOT_TYPE": unified_pivot_type,
            "PROVISIONAL_PIVOT": provisional["pivot"] if provisional else None,
            "PROVISIONAL_DISTANCE_PCT": provisional["distance_pct"] if provisional else None,
            "PROVISIONAL_STAGE": provisional["stage"] if provisional else None,
            "SINGLE_PEAK_PIVOT": single_peak.get("pivot") if single_peak else None,
            "SINGLE_PEAK_DISTANCE_PCT": single_peak.get("distance_pct") if single_peak else None,
            "SINGLE_PEAK_DAYS": single_peak.get("days_since_peak") if single_peak else None,
            "RESISTANCE_PIVOT": (
                resistance.get("pivot") if resistance else None
            ),
            "RESISTANCE_TOUCHES": (
                resistance.get("touches", 0) if resistance else 0
            ),
            "RESISTANCE_PEAK_DIFF_PCT": (
                resistance.get("peak_difference_pct")
                if resistance else None
            ),
            "RESISTANCE_FIRST_PEAK_DAYS": (
                resistance.get("first_peak_days_ago")
                if resistance else None
            ),
            "RESISTANCE_LAST_PEAK_DAYS": (
                resistance.get("last_peak_days_ago")
                if resistance else None
            ),
            "BASE_PIVOT": (
                base_breakout.get("pivot")
                if base_breakout else None
            ),
            "BASE_STAGE": (
                base_breakout.get("stage", "NO_DATA")
                if base_breakout else "NO_DATA"
            ),
            "BASE_DISTANCE_PCT": (
                base_breakout.get("distance_pct")
                if base_breakout else None
            ),
            "BASE_DEPTH_PCT": (
                base_breakout.get("base_depth_pct")
                if base_breakout else None
            ),
            "BASE_VOLUME_RATIO": (
                base_breakout.get("volume_ratio")
                if base_breakout else None
            ),
            "BASE_VALID": (
                base_breakout.get("valid_base", False)
                if base_breakout else False
            ),
            "BASE_DAYS_SINCE_PEAK": (
                base_breakout.get("days_since_peak")
                if base_breakout else None
            ),
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

# 新しいシンプルVCP監視候補：Trend Template + RS + 収縮兆候 + 暫定高値接近。
# 収縮兆候は（5日値幅縮小 または 5日出来高減少）。あくまで手動確認候補。
output["SIMPLE_VCP_CANDIDATE"] = (
    output["Trend_PASS"]
    & output["RS_PASS"]
    & output["SIMPLE_VCP_SHAPE"]
    & output["SIMPLE_PIVOT_NEAR"]
)

unified_pivot_series = pd.to_numeric(output["UNIFIED_PIVOT"], errors="coerce")
unified_distance = (
    (unified_pivot_series - output["Price"])
    / unified_pivot_series * 100
)

output["UNIFIED_READY"] = (
    output["Trend_PASS"]
    & output["RS_PASS"]
    & (output["SWING_VCP_SCORE"] == 4)
    & output["SWING_VCP_READY"]
    & unified_distance.between(0, 5)
)

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

output.to_csv("results/screening_jp.csv", index=False)

data = {
    "updated_at": datetime.now(timezone.utc).isoformat(),
    "total": len(output),
    "passed": int(output["PASS"].sum()),
    "setup_passed": int(output["SETUP_PASS"].sum()),
    "vcp_pullback_shrinking": int(output["VCP_PB_SHRINK"].sum()),
    "vcp_stage_counts": {label: int((output["VCP_STAGE"] == label).sum()) for label in ["対象外", "形成初期", "形成中", "完成候補"]},
    "screen_mode": SCREEN_MODE,
    "market": "JP",
    "settings": {"rs_min": RS_MIN, "pivot_max_distance_pct": PIVOT_MAX_DISTANCE_PCT, "volume_dry_max_ratio": VOLUME_DRY_MAX_RATIO},
    "stocks": output.astype(object).where(pd.notna(output), None).to_dict(orient="records")
}

with open("results/screening_jp.json", "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)

print("\nスクリーニング完了")
print(f"対象銘柄: {data['total']}")
print(f"合格銘柄: {data['passed']}")
print(f"VCP候補: {int(output['VCP_CANDIDATE'].sum())}銘柄")
print(f"セットアップ合格: {int(output['SETUP_PASS'].sum())}銘柄")
print(output[["Ticker", "RS", "SIMPLE_VCP_CANDIDATE"]].head(20))

print("VCP段階別:", data["vcp_stage_counts"])

