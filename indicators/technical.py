"""专业技术指标库。

函数式模块 + :class:`TechnicalIndicators` 门面类。
所有指标纯 pandas/numpy 向量化实现（SAR 等必须迭代的除外），
前 period-1 个周期保持 NaN，不做零填充。

分类：
  - 趋势类：atr / adx / dmi / ichimoku / sar
  - 震荡类：kdj / cci / wr / roc / mom
  - 成交量类：obv / vwap / mfi / cmf
  - 波动率类：bollinger_bandwidth / atr_ratio / historical_volatility
  - 形态类：detect_ma_alignment / detect_cross
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# 内部工具
# ---------------------------------------------------------------------------


def _true_range(high: pd.Series, low: pd.Series, close: pd.Series) -> pd.Series:
    """真实波幅 TR。

    TR = max(high-low, |high-prev_close|, |low-prev_close|)
    首行 prev_close 为 NaN，pandas 逐行 max 自动跳过 NaN，退化为 high-low。
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    tr.name = "tr"
    return tr


def _wilder(s: pd.Series, period: int) -> pd.Series:
    """Wilder 平滑（教科书口径）。

    首个值 = 前 period 个原始值的简单均值（SMA 种子），
    其后递归：out_t = (out_{t-1}*(period-1) + x_t) / period。
    向量化实现：以 ewm(alpha=1/period, adjust=False) 为基，
    再用 (1-1/period)^k 的衰减修正种子偏差；自动跳过序列头部 NaN。
    """
    s = s.astype(float)
    valid = s.dropna()
    out = pd.Series(np.nan, index=s.index, dtype=float)
    if len(valid) < period:
        return out
    pos = period - 1
    z = valid.ewm(alpha=1.0 / period, adjust=False).mean()
    seed = valid.rolling(period, min_periods=period).mean()
    if pd.isna(seed.iloc[pos]):
        return out
    d0 = seed.iloc[pos] - z.iloc[pos]
    powers = (1.0 - 1.0 / period) ** np.arange(len(valid) - pos)
    tail = z.iloc[pos:].to_numpy() + d0 * powers
    out.loc[valid.index[pos]:valid.index[-1]] = tail
    return out


# ---------------------------------------------------------------------------
# 趋势类
# ---------------------------------------------------------------------------


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """平均真实波幅 ATR。

    公式：
        TR = max(high-low, |high-prev_close|, |low-prev_close|)
        ATR = Wilder 平滑(TR, period)

    Args:
        high: 最高价序列。
        low: 最低价序列。
        close: 收盘价序列。
        period: 平滑周期，默认 14。

    Returns:
        ATR 序列，前 period 个值为 NaN。
    """
    tr = _true_range(high, low, close)
    result = _wilder(tr, period)
    result.name = "atr"
    return result


def _dmi_core(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int
) -> Dict[str, pd.Series]:
    """DMI/ADX 共用核心计算。

    公式：
        +DM = up if up>down and up>0 else 0（up=high.diff(), down=-low.diff()）
        -DM = down if down>up and down>0 else 0
        +DI = 100 * Wilder(+DM) / Wilder(TR)
        -DI = 100 * Wilder(-DM) / Wilder(TR)
        DX  = 100 * |+DI - -DI| / (+DI + -DI)
        ADX = Wilder(DX, period)
    """
    up = high.diff()
    down = -low.diff()

    plus_dm = pd.Series(
        np.where((up > down) & (up > 0), up, 0.0), index=high.index
    )
    minus_dm = pd.Series(
        np.where((down > up) & (down > 0), down, 0.0), index=high.index
    )
    # diff() 首行为 NaN，where 已把 NaN 比较结果置 0，这里显式兜底
    plus_dm = plus_dm.where(high.notna(), np.nan)
    minus_dm = minus_dm.where(low.notna(), np.nan)

    tr = _true_range(high, low, close)
    atr_w = _wilder(tr, period)
    plus_di = 100 * _wilder(plus_dm, period) / atr_w
    minus_di = 100 * _wilder(minus_dm, period) / atr_w

    di_sum = (plus_di + minus_di).replace(0.0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / di_sum
    adx = _wilder(dx, period)
    return {"adx": adx, "plus_di": plus_di, "minus_di": minus_di}


def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """平均趋向指数 ADX。

    公式见 :func:`_dmi_core`。ADX>25 表示趋势较强，ADX<20 表示无趋势/震荡。

    Returns:
        DataFrame，列 ``adx`` / ``plus_di``(+DI) / ``minus_di``(-DI)。
    """
    core = _dmi_core(high, low, close, period)
    df = pd.DataFrame(
        {"adx": core["adx"], "plus_di": core["plus_di"], "minus_di": core["minus_di"]}
    )
    return df


def dmi(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.DataFrame:
    """趋向指标 DMI（与 :func:`adx` 共享计算，列顺序按 DMI 惯例排列）。

    Returns:
        DataFrame，列 ``plus_di``(+DI) / ``minus_di``(-DI) / ``adx``。
    """
    core = _dmi_core(high, low, close, period)
    return pd.DataFrame(
        {"plus_di": core["plus_di"], "minus_di": core["minus_di"], "adx": core["adx"]}
    )


def ichimoku(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    tenkan: int = 9,
    kijun: int = 26,
    senkou_b: int = 52,
) -> pd.DataFrame:
    """一目均衡表（Ichimoku Kinko Hyo）。

    公式：
        转换线 tenkan_sen = (HHV(high,tenkan) + LLV(low,tenkan)) / 2
        基准线 kijun_sen  = (HHV(high,kijun)  + LLV(low,kijun))  / 2
        先行带A senkou_a  = (转换线 + 基准线) / 2，向前投影 kijun 期
        先行带B senkou_b  = (HHV(high,sb) + LLV(low,sb)) / 2，向前投影 kijun 期
        迟行带 chikou     = close 向后平移 kijun 期

    Returns:
        DataFrame，列 tenkan_sen/kijun_sen/senkou_a/senkou_b/chikou。
        senkou_a/senkou_b 末尾 kijun 期为 NaN（投影区域），
        chikou 头部 kijun 期为 NaN。
    """
    hh_t = high.rolling(tenkan, min_periods=tenkan).max()
    ll_t = low.rolling(tenkan, min_periods=tenkan).min()
    tenkan_sen = (hh_t + ll_t) / 2.0

    hh_k = high.rolling(kijun, min_periods=kijun).max()
    ll_k = low.rolling(kijun, min_periods=kijun).min()
    kijun_sen = (hh_k + ll_k) / 2.0

    senkou_a = ((tenkan_sen + kijun_sen) / 2.0).shift(kijun)

    hh_sb = high.rolling(senkou_b, min_periods=senkou_b).max()
    ll_sb = low.rolling(senkou_b, min_periods=senkou_b).min()
    senkou_b_line = ((hh_sb + ll_sb) / 2.0).shift(kijun)

    chikou = close.shift(-kijun)

    return pd.DataFrame(
        {
            "tenkan_sen": tenkan_sen,
            "kijun_sen": kijun_sen,
            "senkou_a": senkou_a,
            "senkou_b": senkou_b_line,
            "chikou": chikou,
        }
    )


def sar(
    high: pd.Series,
    low: pd.Series,
    af_start: float = 0.02,
    af_max: float = 0.2,
) -> pd.Series:
    """抛物线转向指标 SAR（Parabolic SAR）。

    公式（迭代计算）：
        多头：SAR_t = SAR_{t-1} + AF * (EP - SAR_{t-1})，
              EP 为期间最高价，创新高时 AF = min(AF + af_start, af_max)；
              SAR 不得高于前两根最低价，被价格跌破则翻空。
        空头：对称，EP 为期间最低价，SAR 不得低于前两根最高价，
              被价格涨破则翻多。

    Args:
        high: 最高价序列。
        low: 最低价序列。
        af_start: 初始加速因子，默认 0.02。
        af_max: 加速因子上限，默认 0.2。

    Returns:
        SAR 值序列。
    """
    h = high.to_numpy(dtype=float)
    l = low.to_numpy(dtype=float)
    n = len(h)
    out = np.full(n, np.nan)
    if n < 3:
        return pd.Series(out, index=high.index, name="sar")

    # 初始状态：默认从多头开始，EP=首日高点，SAR=首日低点，AF=af_start
    is_up = True
    ep = float(h[0])
    sar_val = float(l[0])
    af = af_start

    for i in range(1, n):
        out[i - 1] = sar_val
        prev1 = float(l[i - 1])
        prev2 = float(l[i - 2]) if i >= 2 else prev1
        prev1h = float(h[i - 1])
        prev2h = float(h[i - 2]) if i >= 2 else prev1h

        if is_up:
            sar_val = sar_val + af * (ep - sar_val)
            # SAR 不能高于前两根 K 线最低价
            sar_val = min(sar_val, prev1, prev2)
            if l[i] < sar_val:
                # 多头被跌破 -> 翻空
                is_up = False
                sar_val = ep
                ep = float(l[i])
                af = af_start
            else:
                if h[i] > ep:
                    ep = float(h[i])
                    af = min(af + af_start, af_max)
        else:
            sar_val = sar_val - af * (sar_val - ep)
            sar_val = max(sar_val, prev1h, prev2h)
            if h[i] > sar_val:
                is_up = True
                sar_val = ep
                ep = float(h[i])
                af = af_start
            else:
                if l[i] < ep:
                    ep = float(l[i])
                    af = min(af + af_start, af_max)
    out[n - 1] = sar_val
    return pd.Series(out, index=high.index, name="sar")


# ---------------------------------------------------------------------------
# 震荡类
# ---------------------------------------------------------------------------


def kdj(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    k_period: int = 9,
    d_period: int = 3,
    j_period: int = 3,
) -> pd.DataFrame:
    """随机指标 KDJ（中式通用口径）。

    公式：
        RSV = (close - LLV(low,k)) / (HHV(high,k) - LLV(low,k)) * 100
        K_t = (d-1)/d * K_{t-1} + 1/d * RSV_t  （Wilder 式平滑，alpha=1/d）
        D_t = (d-1)/d * D_{t-1} + 1/d * K_t
        J   = 3*K - 2*D

    Returns:
        DataFrame，列 ``k`` / ``d`` / ``j``。
    """
    ll = low.rolling(k_period, min_periods=k_period).min()
    hh = high.rolling(k_period, min_periods=k_period).max()
    rsv = (close - ll) / (hh - ll).replace(0.0, np.nan) * 100.0

    k = rsv.ewm(alpha=1.0 / d_period, adjust=False).mean()
    d = k.ewm(alpha=1.0 / j_period, adjust=False).mean()
    j = 3.0 * k - 2.0 * d
    return pd.DataFrame({"k": k, "d": d, "j": j})


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """商品通道指数 CCI。

    公式：
        TP  = (high + low + close) / 3
        CCI = (TP - SMA(TP, period)) / (0.015 * MAD(TP, period))
        MAD = TP 与其周期均值的平均绝对偏差。

    Returns:
        CCI 序列，典型区间 ±100，前 period-1 个值为 NaN。
    """
    tp = (high + low + close) / 3.0
    ma = tp.rolling(period, min_periods=period).mean()
    mad = tp.rolling(period, min_periods=period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    result = (tp - ma) / (0.015 * mad).replace(0.0, np.nan)
    result.name = "cci"
    return result


def wr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """威廉指标 WR（Williams %R）。

    公式：
        WR = (HHV(close,N) - close) / (HHV(close,N) - LLV(close,N)) * (-100)

    Returns:
        WR 序列，范围 -100~0，低于 -80 为超卖，高于 -20 为超买。
    """
    hhv = close.rolling(period, min_periods=period).max()
    llv = close.rolling(period, min_periods=period).min()
    result = (hhv - close) / (hhv - llv).replace(0.0, np.nan) * -100.0
    result.name = "wr"
    return result


def roc(close: pd.Series, period: int = 12) -> pd.Series:
    """变动率 ROC。

    公式：ROC = (close - close[t-N]) / close[t-N] * 100
    """
    prev = close.shift(period)
    result = (close - prev) / prev.replace(0.0, np.nan) * 100.0
    result.name = "roc"
    return result


def mom(close: pd.Series, period: int = 10) -> pd.Series:
    """动量 MOM。

    公式：MOM = close - close[t-N]
    """
    result = close - close.shift(period)
    result.name = "mom"
    return result


# ---------------------------------------------------------------------------
# 成交量类
# ---------------------------------------------------------------------------


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """能量潮 OBV。

    公式：close 上涨日 += volume，下跌日 -= volume，平盘 += 0；累计求和。
    """
    direction = np.sign(close.diff()).fillna(0.0)
    result = (direction * volume).cumsum()
    result.name = "obv"
    return result


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series
) -> pd.Series:
    """成交量加权平均价 VWAP（自起始日累计口径）。

    公式：VWAP = sum(TP * volume) / sum(volume)，TP = (high+low+close)/3。
    """
    tp = (high + low + close) / 3.0
    cum_pv = (tp * volume).cumsum()
    cum_v = volume.cumsum().replace(0.0, np.nan)
    result = cum_pv / cum_v
    result.name = "vwap"
    return result


def mfi(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 14
) -> pd.Series:
    """资金流量指标 MFI（带成交量权重的 RSI）。

    公式：
        TP  = (high + low + close) / 3，资金流 MF = TP * volume
        资金流为正：TP 较前日上涨日的 MF 之和；为负：下跌日之和
        MFR = sum(+MF) / sum(-MF)
        MFI = 100 - 100 / (1 + MFR)
    """
    tp = (high + low + close) / 3.0
    mf = tp * volume
    diff = tp.diff()
    pos_mf = mf.where(diff > 0, 0.0)
    neg_mf = mf.where(diff < 0, 0.0)
    pos_sum = pos_mf.rolling(period, min_periods=period).sum()
    neg_sum = neg_mf.rolling(period, min_periods=period).sum()
    mfr = pos_sum / neg_sum.replace(0.0, np.nan)
    result = 100.0 - 100.0 / (1.0 + mfr)
    result.name = "mfi"
    return result


def cmf(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """蔡金资金流 CMF。

    公式：
        MFM = ((close - low) - (high - close)) / (high - low)   # 收盘位置
        CMF = sum(MFM * volume, N) / sum(volume, N)
    """
    rng = (high - low).replace(0.0, np.nan)
    mfm = ((close - low) - (high - close)) / rng
    mfv = mfm * volume
    result = mfv.rolling(period, min_periods=period).sum() / volume.rolling(
        period, min_periods=period
    ).sum()
    result.name = "cmf"
    return result


# ---------------------------------------------------------------------------
# 波动率类
# ---------------------------------------------------------------------------


def bollinger_bandwidth(
    close: pd.Series, period: int = 20, num_std: float = 2.0
) -> pd.Series:
    """布林带带宽 Bandwidth。

    公式：BW = (上轨 - 下轨) / 中轨；带宽扩张表示波动率放大。
    """
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    result = (upper - lower) / mid.replace(0.0, np.nan)
    result.name = "bb_bandwidth"
    return result


def atr_ratio(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """ATR 比率（ATR / close），波动率相对价格的无量纲口径。"""
    result = atr(high, low, close, period) / close.replace(0.0, np.nan)
    result.name = "atr_ratio"
    return result


def historical_volatility(
    close: pd.Series, period: int = 20, trading_days: int = 252
) -> pd.Series:
    """历史波动率（年化）。

    公式：HV = std(日收益率, period) * sqrt(trading_days)
    """
    ret = close.pct_change()
    result = ret.rolling(period, min_periods=period).std() * math.sqrt(trading_days)
    result.name = "hist_vol"
    return result


# ---------------------------------------------------------------------------
# 形态类
# ---------------------------------------------------------------------------


def detect_ma_alignment(
    close: pd.Series, periods: Optional[List[int]] = None
) -> pd.Series:
    """均线多头/空头排列检测。

    Args:
        close: 收盘价序列。
        periods: 均线周期列表，默认 [5, 10, 20, 60]。

    Returns:
        Series：1 = 多头排列（短周期均线依次高于长周期），
        -1 = 空头排列，0 = 无排列。前 max(periods)-1 个值为 NaN。
    """
    if periods is None:
        periods = [5, 10, 20, 60]
    mas = [close.rolling(p, min_periods=p).mean() for p in periods]
    aligned = pd.Series(0, index=close.index, name="ma_alignment")

    bull = pd.Series(True, index=close.index)
    bear = pd.Series(True, index=close.index)
    any_valid = pd.Series(False, index=close.index)
    for i in range(len(mas) - 1):
        bull &= mas[i] > mas[i + 1]
        bear &= mas[i] < mas[i + 1]
        any_valid |= mas[i].notna()

    aligned[bull & any_valid] = 1
    aligned[bear & any_valid] = -1
    aligned[~any_valid] = np.nan
    return aligned


def detect_cross(fast: pd.Series, slow: pd.Series) -> pd.Series:
    """金叉/死叉检测。

    金叉：fast[t-1] <= slow[t-1] 且 fast[t] > slow[t]；
    死叉：fast[t-1] >= slow[t-1] 且 fast[t] < slow[t]。

    Returns:
        Series：1 = 金叉，-1 = 死叉，0 = 无交叉；输入为 NaN 的位置为 NaN。
    """
    prev_fast = fast.shift(1)
    prev_slow = slow.shift(1)
    golden = (prev_fast <= prev_slow) & (fast > slow)
    death = (prev_fast >= prev_slow) & (fast < slow)

    result = pd.Series(0, index=fast.index, name="cross")
    result[golden] = 1
    result[death] = -1
    # 输入本身为 NaN 的位置标记为 NaN（前一周期 NaN 视为无交叉，保持 0）
    result[fast.isna() | slow.isna()] = np.nan
    return result


# ---------------------------------------------------------------------------
# 指标注册表 & 门面类
# ---------------------------------------------------------------------------

#: 指标注册表：name -> 元信息（分类 / 函数 / 参数说明 / 输出列）
INDICATOR_REGISTRY: List[Dict[str, Any]] = [
    {
        "name": "atr",
        "category": "趋势",
        "description": "平均真实波幅，衡量价格波动幅度",
        "params": {"period": 14},
        "output": ["atr"],
    },
    {
        "name": "adx",
        "category": "趋势",
        "description": "平均趋向指数，ADX>25 趋势强，<20 无趋势",
        "params": {"period": 14},
        "output": ["adx", "plus_di", "minus_di"],
    },
    {
        "name": "dmi",
        "category": "趋势",
        "description": "趋向指标，+DI/-DI 指示多空力量方向",
        "params": {"period": 14},
        "output": ["plus_di", "minus_di", "adx"],
    },
    {
        "name": "ichimoku",
        "category": "趋势",
        "description": "一目均衡表：转换线/基准线/先行带A/B/迟行带",
        "params": {"tenkan": 9, "kijun": 26, "senkou_b": 52},
        "output": ["tenkan_sen", "kijun_sen", "senkou_a", "senkou_b", "chikou"],
    },
    {
        "name": "sar",
        "category": "趋势",
        "description": "抛物线转向，止损/反转点位",
        "params": {"af_start": 0.02, "af_max": 0.2},
        "output": ["sar"],
    },
    {
        "name": "kdj",
        "category": "震荡",
        "description": "随机指标 K/D/J，J>100 超买，J<0 超卖",
        "params": {"k_period": 9, "d_period": 3, "j_period": 3},
        "output": ["k", "d", "j"],
    },
    {
        "name": "cci",
        "category": "震荡",
        "description": "商品通道指数，超出 ±100 视为极端区域",
        "params": {"period": 20},
        "output": ["cci"],
    },
    {
        "name": "wr",
        "category": "震荡",
        "description": "威廉指标，<-80 超卖，>-20 超买",
        "params": {"period": 14},
        "output": ["wr"],
    },
    {
        "name": "roc",
        "category": "震荡",
        "description": "变动率，价格相对 N 周期前的涨跌幅百分比",
        "params": {"period": 12},
        "output": ["roc"],
    },
    {
        "name": "mom",
        "category": "震荡",
        "description": "动量，价格相对 N 周期前的绝对差值",
        "params": {"period": 10},
        "output": ["mom"],
    },
    {
        "name": "obv",
        "category": "成交量",
        "description": "能量潮，量价趋势验证",
        "params": {},
        "output": ["obv"],
    },
    {
        "name": "vwap",
        "category": "成交量",
        "description": "成交量加权平均价（累计）",
        "params": {},
        "output": ["vwap"],
    },
    {
        "name": "mfi",
        "category": "成交量",
        "description": "资金流量指标，>80 超买，<20 超卖",
        "params": {"period": 14},
        "output": ["mfi"],
    },
    {
        "name": "cmf",
        "category": "成交量",
        "description": "蔡金资金流，正值表示资金净流入",
        "params": {"period": 20},
        "output": ["cmf"],
    },
    {
        "name": "bollinger_bandwidth",
        "category": "波动率",
        "description": "布林带带宽，(上轨-下轨)/中轨",
        "params": {"period": 20, "num_std": 2.0},
        "output": ["bb_bandwidth"],
    },
    {
        "name": "atr_ratio",
        "category": "波动率",
        "description": "ATR 比率（ATR/close）",
        "params": {"period": 14},
        "output": ["atr_ratio"],
    },
    {
        "name": "historical_volatility",
        "category": "波动率",
        "description": "历史波动率（年化）",
        "params": {"period": 20, "trading_days": 252},
        "output": ["hist_vol"],
    },
    {
        "name": "detect_ma_alignment",
        "category": "形态",
        "description": "均线排列检测：1 多头 / -1 空头 / 0 无",
        "params": {"periods": [5, 10, 20, 60]},
        "output": ["ma_alignment"],
    },
    {
        "name": "detect_cross",
        "category": "形态",
        "description": "金叉/死叉检测：1 金叉 / -1 死叉 / 0 无",
        "params": {},
        "output": ["cross"],
    },
]


class TechnicalIndicators:
    """技术指标门面类，提供统一计算入口。

    Examples:
        >>> ti = TechnicalIndicators()
        >>> ti.list_indicators()  # 查看可用指标
        >>> result = ti.calculate(df, "adx")  # 计算 ADX
        >>> all_df = ti.calculate_all(df)     # 计算全部指标并合并
    """

    def __init__(self) -> None:
        self._registry = INDICATOR_REGISTRY
        self._funcs: Dict[str, Callable[..., Any]] = {
            "atr": atr,
            "adx": adx,
            "dmi": dmi,
            "ichimoku": ichimoku,
            "sar": sar,
            "kdj": kdj,
            "cci": cci,
            "wr": wr,
            "roc": roc,
            "mom": mom,
            "obv": obv,
            "vwap": vwap,
            "mfi": mfi,
            "cmf": cmf,
            "bollinger_bandwidth": bollinger_bandwidth,
            "atr_ratio": atr_ratio,
            "historical_volatility": historical_volatility,
            "detect_ma_alignment": detect_ma_alignment,
            "detect_cross": detect_cross,
        }

    # ------------------------------------------------------------------

    def list_indicators(self) -> List[Dict[str, Any]]:
        """返回可用指标列表（名称/分类/参数说明/默认值/输出列）。"""
        return [
            {
                "name": item["name"],
                "category": item["category"],
                "description": item["description"],
                "params": dict(item["params"]),
                "output": list(item["output"]),
            }
            for item in self._registry
        ]

    # ------------------------------------------------------------------

    def calculate(self, df: pd.DataFrame, indicator_name: str, **kwargs: Any) -> Any:
        """按名称计算指定指标。

        Args:
            df: 行情数据，需包含 open/high/low/close/volume 列。
            indicator_name: 指标名称，见 :meth:`list_indicators`。
            **kwargs: 覆盖指标默认参数（如 period=20）。

        Returns:
            pd.Series 或 pd.DataFrame（多列指标）。

        Raises:
            KeyError: 指标名称不存在。
        """
        if indicator_name not in self._funcs:
            raise KeyError(
                f"未知指标: {indicator_name}，"
                f"可用: {[i['name'] for i in self._registry]}"
            )
        func = self._funcs[indicator_name]
        h, l, c, v = df["high"], df["low"], df["close"], df["volume"]

        if indicator_name in ("atr", "atr_ratio"):
            return func(h, l, c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name in ("adx", "dmi"):
            return func(h, l, c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "ichimoku":
            return func(h, l, c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "sar":
            return func(h, l, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "kdj":
            return func(h, l, c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "cci":
            return func(h, l, c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "wr":
            return func(h, l, c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "roc":
            return func(c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "mom":
            return func(c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "obv":
            return func(c, v)
        if indicator_name == "vwap":
            return func(h, l, c, v)
        if indicator_name in ("mfi", "cmf"):
            return func(h, l, c, v, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "bollinger_bandwidth":
            return func(c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "historical_volatility":
            return func(c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "detect_ma_alignment":
            return func(c, **_params_with_defaults(self._registry, indicator_name, kwargs))
        if indicator_name == "detect_cross":
            fast = kwargs.pop("fast", None)
            slow = kwargs.pop("slow", None)
            if fast is None or slow is None:
                fast = df["close"].rolling(5, min_periods=5).mean()
                slow = df["close"].rolling(20, min_periods=20).mean()
            return func(fast, slow)
        return func(**kwargs)  # pragma: no cover

    # ------------------------------------------------------------------

    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有指标并合并到 DataFrame 副本中。

        Returns:
            原 df + 各指标列（多列指标按其输出列名追加）。
        """
        out = df.copy()
        for item in self._registry:
            name = item["name"]
            result = self.calculate(out, name)
            if isinstance(result, pd.DataFrame):
                for col in result.columns:
                    out[f"{name}_{col}"] = result[col]
            else:
                out[result.name or name] = result
        return out


def _params_with_defaults(
    registry: List[Dict[str, Any]], name: str, overrides: Dict[str, Any]
) -> Dict[str, Any]:
    """合并注册表默认参数与调用方覆盖参数。"""
    meta = next((i for i in registry if i["name"] == name), None)
    params: Dict[str, Any] = dict(meta["params"]) if meta else {}
    params.update({k: v for k, v in overrides.items() if v is not None})
    return params
