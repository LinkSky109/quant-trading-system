"""专业多因子分析引擎。

实现 7 大类共 20 个因子的计算、IC 分析、分层回测与因子暴露度分析。

- 价量/技术类因子：真实从 K 线 DataFrame 计算（依赖 :mod:`utils.indicators`）。
- 基本面类因子（价值/成长/质量）：当前使用基于 symbol 哈希生成的确定性 mock
  数据，并预留 :meth:`FactorEngine.load_fundamental_data` 接口，未来接入真实
  财务数据源后只需覆盖该方法即可。

典型用法::

    engine = FactorEngine()
    factor_df = engine.calculate_factors(klines_df, symbol="600519.SH")
    panel = engine.build_factor_panel({"600519.SH": df1, "000858.SZ": df2},
                                      factor_name="momentum_20")
    ic = engine.factor_ic_analysis(panel_dict, "momentum_20", forward_days=5)
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from utils import indicators as ta

logger = logging.getLogger(__name__)

# 横截面分析时每期最少需要的样本数，低于此数的期不参与统计
_MIN_CROSS_SECTION = 3

# 单只标的因子面板：{symbol: 带因子列与 close 列的 DataFrame}
FactorPanelInput = Union[Dict[str, pd.DataFrame], pd.DataFrame]


class FactorEngine:
    """多因子计算与分析引擎。

    所有价量因子直接基于传入的 K 线 DataFrame（index 为日期，列含
    open/high/low/close/volume/amount）计算；基本面因子使用确定性 mock
    数据并预留真实数据接入点。
    """

    def __init__(self) -> None:
        """初始化引擎，加载基本面 mock 数据缓存。"""
        self._fundamental_cache: Dict[str, Dict[str, float]] = {}
        # 预留：真实财务数据源（如 Wind/同花顺 API）接入后在此填充
        self._fundamental_source: Optional[Any] = None
        # 因子元数据注册表：name -> (category, description, direction, func_name)
        self._factor_registry: List[Dict[str, Any]] = self._build_registry()

    # ------------------------------------------------------------------ #
    # 因子元数据
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_registry() -> List[Dict[str, Any]]:
        """构建因子元数据注册表。

        direction: +1 表示因子值越大预期收益越高（做多高分位）；
                   -1 表示因子值越小预期收益越高（做多低分位）。
        """
        return [
            # 价值因子（低估值为好）
            {"name": "pe_inverse", "category": "价值",
             "description": "PE 倒数（E/P），估值越低得分越高", "direction": +1},
            {"name": "pb_inverse", "category": "价值",
             "description": "PB 倒数（B/P），账面估值越低得分越高", "direction": +1},
            {"name": "ps_inverse", "category": "价值",
             "description": "PS 倒数（S/P），市销率越低得分越高", "direction": +1},
            {"name": "dividend_yield", "category": "价值",
             "description": "股息率，分红回报越高得分越高", "direction": +1},
            # 成长因子
            {"name": "revenue_growth", "category": "成长",
             "description": "营业收入同比增速（mock）", "direction": +1},
            {"name": "profit_growth", "category": "成长",
             "description": "净利润同比增速（mock）", "direction": +1},
            {"name": "roe_change", "category": "成长",
             "description": "ROE 环比变化（mock）", "direction": +1},
            # 动量因子
            {"name": "momentum_20", "category": "动量",
             "description": "20 日收益率", "direction": +1},
            {"name": "momentum_60", "category": "动量",
             "description": "60 日收益率", "direction": +1},
            {"name": "momentum_120_excl5", "category": "动量",
             "description": "120 日收益率并剔除最近 5 日（规避短期反转）",
             "direction": +1},
            # 质量因子
            {"name": "roe", "category": "质量",
             "description": "净资产收益率 ROE（mock）", "direction": +1},
            {"name": "gross_margin", "category": "质量",
             "description": "毛利率（mock）", "direction": +1},
            {"name": "debt_ratio_inverse", "category": "质量",
             "description": "资产负债率倒数，杠杆越低得分越高", "direction": +1},
            # 波动率因子（低波为好）
            {"name": "volatility_20_inverse", "category": "波动率",
             "description": "20 日收益率标准差的倒数，低波得分高", "direction": +1},
            {"name": "volatility_60_inverse", "category": "波动率",
             "description": "60 日收益率标准差的倒数，低波得分高", "direction": +1},
            # 技术因子
            {"name": "rsi_14", "category": "技术",
             "description": "14 日相对强弱指标 RSI", "direction": -1},
            {"name": "macd_hist", "category": "技术",
             "description": "MACD 柱状图（DIF-DEA）*2", "direction": +1},
            {"name": "bollinger_position", "category": "技术",
             "description": "收盘价在布林带中的位置 (close-lower)/(upper-lower)",
             "direction": +1},
            # 流动性因子
            {"name": "turnover_inverse", "category": "流动性",
             "description": "换手率倒数（成交量/流通股），低换手得分高",
             "direction": +1},
            {"name": "amount_log", "category": "流动性",
             "description": "成交额对数 log(amount)，流动性越好得分越高",
             "direction": +1},
            # 高频因子
            {"name": "overnight_jump", "category": "高频",
             "description": "隔夜跳空收益率 (open-pre_close)/pre_close",
             "direction": -1},
            {"name": "intraday_range", "category": "高频",
             "description": "日内振幅比 (high-low)/close", "direction": -1},
            {"name": "volume_imbalance", "category": "高频",
             "description": "成交量量比", "direction": +1},
            {"name": "price_acceleration", "category": "高频",
             "description": "价格加速度（收益率二阶差分）", "direction": -1},
            {"name": "vwap_deviation", "category": "高频",
             "description": "VWAP 偏离度", "direction": -1},
            {"name": "consecutive_up", "category": "高频",
             "description": "连续上涨天数", "direction": -1},
            {"name": "consecutive_down", "category": "高频",
             "description": "连续下跌天数", "direction": +1},
            {"name": "volatility_clustering", "category": "高频",
             "description": "波动率聚集度", "direction": +1},
        ]

    def get_factor_list(self) -> List[Dict[str, Any]]:
        """返回可用因子列表。

        Returns:
            每个元素含 name/category/description/direction 四个字段。
        """
        return [
            {"name": f["name"], "category": f["category"],
             "description": f["description"], "direction": f["direction"]}
            for f in self._factor_registry
        ]

    @property
    def factor_names(self) -> List[str]:
        """所有因子名列表。"""
        return [f["name"] for f in self._factor_registry]

    # ------------------------------------------------------------------ #
    # 基本面 mock 数据
    # ------------------------------------------------------------------ #
    def load_fundamental_data(self) -> Optional[Dict[str, Any]]:
        """加载真实基本面财务数据（预留接口）。

        当前未接入真实数据源时返回 None，因子计算将使用
        :meth:`_generate_mock_fundamentals` 生成的确定性 mock 数据。
        未来接入 Wind / Tushare / 财报库后，在此填充
        ``{symbol: {field: value}}`` 结构即可。

        Returns:
            基本面数据字典或 None。
        """
        return self._fundamental_source

    def _generate_mock_fundamentals(self, symbol: str) -> Dict[str, float]:
        """基于 symbol 哈希生成确定性 mock 基本面数据。

        同一 symbol 每次调用返回完全相同的值（结果带缓存），保证
        因子分析可复现。同时尝试从 ``config/stock_pool.yaml`` 读取真实
        pe_ttm 覆盖 mock PE。

        Args:
            symbol: 标准化后的标的代码。

        Returns:
            含 pe/pb/ps/股息率/成长率/ROE/毛利率/负债率/流通股本等字段的字典。
        """
        if symbol in self._fundamental_cache:
            return self._fundamental_cache[symbol]

        # 用 symbol 的 MD5 前 8 位构造可复现随机种子
        seed_hex = hashlib.md5(symbol.encode("utf-8")).hexdigest()[:8]
        seed = int(seed_hex, 16)
        rng = np.random.RandomState(seed)

        fundamentals: Dict[str, float] = {
            "pe_ttm": float(np.clip(rng.uniform(6, 55), 2.0, None)),
            "pb": float(rng.uniform(0.8, 8.0)),
            "ps": float(rng.uniform(0.5, 10.0)),
            "dividend_yield": float(rng.uniform(0.0, 0.05)),
            "revenue_growth": float(rng.uniform(-0.10, 0.40)),
            "profit_growth": float(rng.uniform(-0.20, 0.50)),
            "roe": float(rng.uniform(0.03, 0.35)),
            "roe_change": float(rng.uniform(-0.05, 0.10)),
            "gross_margin": float(rng.uniform(0.10, 0.70)),
            "debt_ratio": float(rng.uniform(0.10, 0.80)),
            "shares_out_yi": float(rng.uniform(5.0, 300.0)),  # 流通股本（亿股）
        }

        # 尝试用 stock_pool.yaml 中的真实 pe_ttm 覆盖
        try:
            from config import load_stock_pool
            pool = load_stock_pool()
            for s in pool.get("stocks", []):
                if s.get("symbol") == symbol and s.get("pe_ttm"):
                    fundamentals["pe_ttm"] = float(s["pe_ttm"])
                    break
        except Exception as exc:  # pragma: no cover - 配置缺失时静默降级
            logger.debug("未加载 stock_pool.yaml，PE 使用 mock 值: %s", exc)

        # 若外部已注入真实财务数据，优先使用
        source = self.load_fundamental_data()
        if source and symbol in source:
            fundamentals.update(source[symbol])

        self._fundamental_cache[symbol] = fundamentals
        return fundamentals

    # ------------------------------------------------------------------ #
    # 因子计算
    # ------------------------------------------------------------------ #
    def calculate_factors(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> pd.DataFrame:
        """计算单只股票的全部因子值。

        Args:
            df: K 线 DataFrame，index 为日期，需含 close/volume/amount 列。
            symbol: 标的代码，用于生成确定性基本面 mock 数据。

        Returns:
            附加了所有因子列的 DataFrame；输入为空时原样返回。
        """
        out = df.copy()
        if out.empty or "close" not in out.columns:
            return out

        close = out["close"]
        ret = close.pct_change()

        # ---------------- 动量因子（价量真实计算） ----------------
        out["momentum_20"] = close.pct_change(20)
        out["momentum_60"] = close.pct_change(60)
        # 120 日收益率，剔除最近 5 日：t 日取值为 close[t-5]/close[t-125]-1
        out["momentum_120_excl5"] = close.shift(5) / close.shift(125) - 1.0

        # ---------------- 波动率因子（价量真实计算） ----------------
        vol20 = ret.rolling(window=20, min_periods=20).std()
        vol60 = ret.rolling(window=60, min_periods=60).std()
        out["volatility_20_inverse"] = 1.0 / vol20.replace(0, np.nan)
        out["volatility_60_inverse"] = 1.0 / vol60.replace(0, np.nan)

        # ---------------- 技术因子（调用 utils.indicators） ----------------
        out["rsi_14"] = ta.rsi(close, 14)
        _, _, macd_hist = ta.macd(close, 12, 26, 9)
        out["macd_hist"] = macd_hist
        bb_mid, bb_upper, bb_lower = ta.bollinger_bands(close, 20, 2.0)
        bb_width = (bb_upper - bb_lower).replace(0, np.nan)
        out["bollinger_position"] = (close - bb_lower) / bb_width

        # ---------------- 流动性因子 ----------------
        out["amount_log"] = np.log(out["amount"].replace(0, np.nan))
        fund = self._generate_mock_fundamentals(symbol)
        shares = fund["shares_out_yi"] * 1e8  # 亿股 -> 股
        turnover = out["volume"] / shares
        out["turnover_inverse"] = -turnover

        # ---------------- 基本面因子（mock，按日广播为常数序列） ----------------
        out["pe_inverse"] = 1.0 / fund["pe_ttm"]
        out["pb_inverse"] = 1.0 / fund["pb"]
        out["ps_inverse"] = 1.0 / fund["ps"]
        out["dividend_yield"] = fund["dividend_yield"]
        out["revenue_growth"] = fund["revenue_growth"]
        out["profit_growth"] = fund["profit_growth"]
        out["roe_change"] = fund["roe_change"]
        out["roe"] = fund["roe"]
        out["gross_margin"] = fund["gross_margin"]
        out["debt_ratio_inverse"] = 1.0 / fund["debt_ratio"]

        # ---------------- 高频因子（需 open/high/low/volume/amount） ----------------
        if all(c in out.columns for c in ("open", "high", "low", "volume", "amount")):
            from factors.high_frequency import HighFrequencyFactorEngine as _HF
            hf = _HF()
            out = hf.calculate_all(out)

        return out

    # ------------------------------------------------------------------ #
    # 横截面面板构建
    # ------------------------------------------------------------------ #
    def build_factor_panel(
        self, symbol_data: Dict[str, pd.DataFrame], factor_name: str
    ) -> pd.DataFrame:
        """构建横截面因子面板。

        Args:
            symbol_data: ``{symbol: K线DataFrame}``，内部会先对每只股票计算因子。
            factor_name: 要抽取的因子名。

        Returns:
            DataFrame，index 为日期，columns 为标的代码，值为因子值。
        """
        if factor_name not in self.factor_names:
            raise ValueError(f"未知因子: {factor_name}，可选: {self.factor_names}")

        series: Dict[str, pd.Series] = {}
        for symbol, df in symbol_data.items():
            if df is None or df.empty:
                continue
            computed = self.calculate_factors(df, symbol=symbol)
            if factor_name in computed.columns:
                s = computed[factor_name].copy()
                s.name = symbol
                series[symbol] = s

        if not series:
            return pd.DataFrame()
        panel = pd.concat(series.values(), axis=1)
        panel.columns = list(series.keys())
        return panel.sort_index()

    # ------------------------------------------------------------------ #
    # 内部工具：统一面板输入
    # ------------------------------------------------------------------ #
    @staticmethod
    def _to_factor_matrix(
        factor_panel: FactorPanelInput, factor_name: str = ""
    ) -> pd.DataFrame:
        """把 {symbol: DataFrame} 或 DataFrame 统一为 日期×标的 因子矩阵。"""
        if isinstance(factor_panel, pd.DataFrame):
            return factor_panel
        # dict: 每个 value 是带因子列（可能还有 close 列）的 DataFrame
        cols: Dict[str, pd.Series] = {}
        for sym, df in factor_panel.items():
            if df is None or df.empty:
                continue
            col = df[factor_name] if factor_name in df.columns else df.iloc[:, 0]
            cols[sym] = col
        if not cols:
            return pd.DataFrame()
        mat = pd.concat(cols.values(), axis=1)
        mat.columns = list(cols.keys())
        return mat.sort_index()

    def _build_forward_returns(
        self,
        factor_panel: FactorPanelInput,
        factor_name: str,
        forward_days: int,
        price_panel: Optional[pd.DataFrame] = None,
    ) -> pd.DataFrame:
        """构造未来 N 日收益率矩阵（日期×标的）。"""
        if price_panel is not None:
            close = price_panel.sort_index()
        elif isinstance(factor_panel, dict):
            close_dict: Dict[str, pd.Series] = {}
            for sym, df in factor_panel.items():
                if df is None or df.empty or "close" not in df.columns:
                    continue
                s = df["close"].copy()
                s.name = sym
                close_dict[sym] = s
            if not close_dict:
                return pd.DataFrame()
            close = pd.concat(close_dict.values(), axis=1)
            close.columns = list(close_dict.keys())
            close = close.sort_index()
        else:
            raise ValueError(
                "factor_panel 为 DataFrame 时必须提供 price_panel（未来收益所需的收盘价）"
            )
        # t 日买入，t+forward_days 日卖出的收益
        fwd = close.shift(-forward_days) / close - 1.0
        return fwd

    # ------------------------------------------------------------------ #
    # IC 分析
    # ------------------------------------------------------------------ #
    def factor_ic_analysis(
        self,
        factor_panel: FactorPanelInput,
        factor_name: str,
        forward_days: int = 5,
        price_panel: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """计算因子的信息系数（IC）序列与统计量。

        对每个交易日，计算当期因子值横截面与未来 N 日收益率横截面的
        Spearman 秩相关系数。

        Args:
            factor_panel: ``{symbol: 含因子列与 close 列的 DataFrame}``，
                或已经构建好的 日期×标的 因子值 DataFrame。
            factor_name: 因子名（dict 输入时用于取列）。
            forward_days: 未来收益率天数。
            price_panel: 当 factor_panel 已是 DataFrame 时，需另外传入收盘价矩阵。

        Returns:
            含 ic_mean/ic_std/ic_ir/ic_win_rate/ic_series 的字典；
            数据不足时各统计量为 NaN。
        """
        F = self._to_factor_matrix(factor_panel, factor_name)
        if F.empty:
            return self._empty_ic_result()

        R = self._build_forward_returns(factor_panel, factor_name,
                                        forward_days, price_panel)
        if R.empty:
            return self._empty_ic_result()

        common_idx = F.index.intersection(R.index)
        common_cols = F.columns.intersection(R.columns)
        F = F.loc[common_idx, common_cols]
        R = R.loc[common_idx, common_cols]

        ic_values: List[float] = []
        ic_dates: List[Any] = []
        for dt in F.index:
            f_row = F.loc[dt]
            r_row = R.loc[dt]
            mask = f_row.notna() & r_row.notna()
            if int(mask.sum()) < _MIN_CROSS_SECTION:
                continue
            fv = f_row[mask].values
            rv = r_row[mask].values
            if np.std(fv) == 0 or np.std(rv) == 0:
                continue
            ic, _ = stats.spearmanr(fv, rv)
            if np.isnan(ic):
                continue
            ic_values.append(float(ic))
            ic_dates.append(dt)

        ic_series = pd.Series(ic_values, index=pd.Index(ic_dates, name="date"),
                              name="ic")
        return self._summarize_ic(ic_series)

    @staticmethod
    def _empty_ic_result() -> Dict[str, Any]:
        return {
            "ic_mean": float("nan"), "ic_std": float("nan"),
            "ic_ir": float("nan"), "ic_win_rate": float("nan"),
            "ic_series": pd.Series(dtype=float, name="ic"),
            "n_periods": 0,
        }

    @staticmethod
    def _summarize_ic(ic_series: pd.Series) -> Dict[str, Any]:
        if ic_series.empty:
            return FactorEngine._empty_ic_result()
        ic_mean = float(ic_series.mean())
        ic_std = float(ic_series.std(ddof=1)) if len(ic_series) > 1 else float("nan")
        ic_ir = ic_mean / ic_std if ic_std and not np.isnan(ic_std) else float("nan")
        win_rate = float((ic_series > 0).mean())
        return {
            "ic_mean": ic_mean,
            "ic_std": ic_std,
            "ic_ir": ic_ir,
            "ic_win_rate": win_rate,
            "ic_series": ic_series,
            "n_periods": int(len(ic_series)),
        }

    # ------------------------------------------------------------------ #
    # 分层回测
    # ------------------------------------------------------------------ #
    def layered_backtest(
        self,
        factor_panel: FactorPanelInput,
        factor_name: str,
        n_layers: int = 5,
        forward_days: int = 5,
        price_panel: Optional[pd.DataFrame] = None,
    ) -> Dict[str, Any]:
        """因子分层回测。

        每个横截面期按因子值将标的等分为 ``n_layers`` 组（Q1 最低 ~ Qn 最高），
        组内等权持有 ``forward_days`` 天，统计各层平均收益与多空收益。

        Args:
            factor_panel: 同 :meth:`factor_ic_analysis`。
            factor_name: 因子名。
            n_layers: 分层数，默认 5。
            forward_days: 持有期天数。
            price_panel: 收盘价矩阵（factor_panel 为 DataFrame 时必传）。

        Returns:
            含 layer_returns / long_short_return / monotonicity / n_periods 的字典。
        """
        F = self._to_factor_matrix(factor_panel, factor_name)
        if F.empty:
            return self._empty_layered_result(n_layers)
        R = self._build_forward_returns(factor_panel, factor_name,
                                        forward_days, price_panel)
        if R.empty:
            return self._empty_layered_result(n_layers)

        common_idx = F.index.intersection(R.index)
        common_cols = F.columns.intersection(R.columns)
        F = F.loc[common_idx, common_cols]
        R = R.loc[common_idx, common_cols]

        # 记录每期每层收益的列表
        layer_period_returns: List[List[float]] = [[] for _ in range(n_layers)]
        long_short: List[float] = []

        for dt in F.index:
            f_row = F.loc[dt]
            r_row = R.loc[dt]
            mask = f_row.notna() & r_row.notna()
            n_valid = int(mask.sum())
            if n_valid < n_layers:
                continue
            valid = pd.DataFrame({"f": f_row[mask], "r": r_row[mask]})
            # qcut 等分为 n_layers 组；duplicates='drop' 容错
            try:
                valid["layer"] = pd.qcut(valid["f"], q=n_layers, labels=False,
                                         duplicates="drop")
            except ValueError:
                continue
            realized_layers = sorted(valid["layer"].unique())
            if len(realized_layers) < 2:
                continue
            layer_ret: Dict[int, float] = {}
            for lyr, grp in valid.groupby("layer"):
                layer_ret[int(lyr)] = float(grp["r"].mean())
            # 映射到 0..n_layers-1
            ret_by_order: List[float] = []
            for order, lyr_val in enumerate(realized_layers):
                layer_period_returns[order].append(layer_ret[lyr_val])
                ret_by_order.append(layer_ret[lyr_val])
            if len(ret_by_order) == n_layers:
                long_short.append(ret_by_order[-1] - ret_by_order[0])

        # 汇总各层平均收益（对齐到实际分层数）
        layer_means: List[float] = []
        for i in range(n_layers):
            samples = layer_period_returns[i]
            layer_means.append(float(np.mean(samples)) if samples else float("nan"))

        valid_layers = [v for v in layer_means if not np.isnan(v)]
        long_short_mean = (float(np.mean(long_short))
                           if long_short else float("nan"))

        # 单调性检验：层序与层收益的 Spearman 相关
        mono_corr = float("nan")
        if len(valid_layers) >= 3:
            ranks = np.arange(1, len(valid_layers) + 1)
            mono_corr, _ = stats.spearmanr(ranks, valid_layers)

        return {
            "layer_returns": {f"Q{i+1}": layer_means[i] for i in range(n_layers)},
            "long_short_return": long_short_mean,
            "long_short_series": pd.Series(long_short),
            "monotonicity": float(mono_corr) if not np.isnan(mono_corr) else float("nan"),
            "n_layers": n_layers,
            "n_periods": int(len(long_short)),
        }

    @staticmethod
    def _empty_layered_result(n_layers: int) -> Dict[str, Any]:
        return {
            "layer_returns": {f"Q{i+1}": float("nan") for i in range(n_layers)},
            "long_short_return": float("nan"),
            "long_short_series": pd.Series(dtype=float),
            "monotonicity": float("nan"),
            "n_layers": n_layers,
            "n_periods": 0,
        }

    # ------------------------------------------------------------------ #
    # 因子暴露度
    # ------------------------------------------------------------------ #
    @staticmethod
    def _zscore(series: pd.Series) -> pd.Series:
        """对单因子历史序列做 z-score 标准化（截面之外的时序标准化）。"""
        mu = series.mean()
        sigma = series.std(ddof=0)
        if sigma == 0 or np.isnan(sigma):
            return series * 0.0
        return (series - mu) / sigma

    def factor_exposure(
        self, df: pd.DataFrame, symbol: str = ""
    ) -> Dict[str, float]:
        """计算单只股票在各因子上的 z-score 标准化暴露度（取最新交易日）。

        对每个因子在该股票的历史时序上做 z-score 标准化，返回最新一日
        的暴露度。理论上每个因子全历史标准化后均值≈0、标准差≈1。

        Args:
            df: K 线 DataFrame。
            symbol: 标的代码。

        Returns:
            ``{factor_name: 最新暴露度}`` 的字典；无法计算的因子值为 NaN。
        """
        computed = self.calculate_factors(df, symbol=symbol)
        exposure: Dict[str, float] = {}
        for name in self.factor_names:
            if name not in computed.columns:
                exposure[name] = float("nan")
                continue
            z = self._zscore(computed[name].dropna())
            exposure[name] = float(z.iloc[-1]) if not z.empty else float("nan")
        return exposure
