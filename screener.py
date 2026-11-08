1
2
3
4
5
6
7
8
9
10
11
12
13
14
15
16
17
18
19
20
21
22
23
24
25
26
27
28
29
30
31
32
33
34
35
36
37
38
39
40
41
42
43
44
45
46
47
48
49
50
51
52
53
54
55
56
57
58
59
60
61
62
63
64
65
66
67
68
69
70
71
72
73
74
75
76
77
78
79
80
81
82
83
84
85
86
87
88
89
90
91
92
93
94
95
96

import yfinance as yf
import pandas as pd
from pathlib import Path

# 標準モード：既存のTT/VCP/PASSは変更せず、SETUP_PASSを追加
SCREEN_MODE = "simple_vcp_plus_legacy"
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
