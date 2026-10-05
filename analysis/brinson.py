"""Brinson 绩效归因模块（机构级多因子拆解）。

本模块实现经典 Brinson 归因框架，将组合相对基准的超额收益拆解为三个可解释分量：

- **配置效应（Allocation Effect, AR）**：组合在各行业/分组上的超配/低配相对基准带来的收益；
- **选择效应（Selection Effect, SR）**：在各分组内部，组合选股相对基准选股带来的收益；
- **交互效应（Interaction Effect, IR）**：超配/低配与选股能力交叉产生的残差项。

支持两种模型：

- ``model="bhb"``：Brinson–Hood–Beebower（1986）经典模型，
  ``AR_i = (w_pi - w_bi) * (r_bi - r_b)``；
- ``model="fachler"``：Brinson–Fachler（1986）变体，
  ``AR_i = (w_pi - w_bi) * (r_bi - r_b_total)``。

.. note::
    按本模块契约实现，``r_b`` 与 ``r_b_total`` 均为整体基准收益
    ``Σ w_bi·r_bi``。在权重闭合（``Σw_pi = Σw_bi = 1``）时，两模型逐组配置效应
    完全相等，配置效应总和也相等（差额 ``= r_b·Σ(w_pi - w_bi) = 0``）。
    文献中另一种常见记法把"是否扣减市场基准收益 r_b"作为 BHB 与 Fachler 的区分点：
    原始 BHB 写作 ``AR_i = (w_pi - w_bi)·r_bi``（不扣市场收益），
    Brinson–Fachler 写作 ``AR_i = (w_pi - w_bi)(r_bi - r_b)``（扣市场收益）。
    本模块统一采用扣减市场基准收益的写法，两模型仅在显式引用上区分。

多期归因支持 Cariño（1999）平滑链接与简单 GRAP 算术链接两种方法。

.. warning::
    **行业分类为简化 / mock 数据**：当调用方仅传入持仓（symbol+weight）而未提供分组时，
    本模块使用基于 symbol 稳定哈希的 5 组 mock 行业划分，**不代表真实行业归属**。
    P0-04 财务数据接入后应切换为真实申万/中信行业分类。
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

__all__ = ["BrinsonAttribution", "BrinsonGroupResult", "BrinsonResult"]

# 闭合校验与权重校验容差
_CLOSURE_TOL = 1e-6   # 总超额收益闭合残差容差（<0.01%）
_WEIGHT_TOL = 1e-4    # 权重求和≈1 的容差
_VALID_MODELS = ("bhb", "fachler")


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------


@dataclass
class BrinsonGroupResult:
    """单个分组（行业/风格/市值）的归因结果。

    Attributes:
        sector: 分组名称。
        w_p: 组合权重。
        w_b: 基准权重。
        r_p: 组合分组收益。
        r_b: 基准分组收益。
        ar: 配置效应。
        sr: 选择效应。
        ir: 交互效应。
        total: 本组三效应合计 = ar + sr + ir。
    """

    sector: str
    w_p: float
    w_b: float
    r_p: float
    r_b: float
    ar: float
    sr: float
    ir: float

    @property
    def total(self) -> float:
        """本组效应合计。"""
        return self.ar + self.sr + self.ir

    def as_dict(self) -> Dict[str, float]:
        """序列化为 dict，便于 JSON 输出。"""
        return {
            "sector": self.sector,
            "w_p": self.w_p,
            "w_b": self.w_b,
            "r_p": self.r_p,
            "r_b": self.r_b,
            "ar": self.ar,
            "sr": self.sr,
            "ir": self.ir,
            "total": self.total,
        }


@dataclass
class BrinsonResult:
    """单期 Brinson 归因结果。

    Attributes:
        groups: 各分组明细。
        allocation_effect: 总配置效应（ΣAR）。
        selection_effect: 总选择效应（ΣSR）。
        interaction_effect: 总交互效应（ΣIR）。
        total_active_return: 总超额收益（三效应之和）。
        portfolio_return: 组合总收益 r_p = Σ w_pi·r_pi。
        benchmark_return: 基准总收益 r_b = Σ w_bi·r_bi。
        closure_residual: 闭合残差 = total_active_return - (r_p - r_b)。
        model: 使用的模型 bhb/fachler。
        dimension: 归因维度名称（行业/风格/市值）。
        sector_source: 行业分类来源说明（真实 / mock）。
    """

    groups: List[BrinsonGroupResult]
    allocation_effect: float
    selection_effect: float
    interaction_effect: float
    total_active_return: float
    portfolio_return: float
    benchmark_return: float
    closure_residual: float
    model: str = "bhb"
    dimension: str = "industry"
    sector_source: str = "provided"
    meta: Dict[str, Any] = field(default_factory=dict)

    # -- 序列化为 dict ------------------------------------------------------
    def as_dict(self) -> Dict[str, Any]:
        """序列化为可 JSON 化的 dict。"""
        return {
            "model": self.model,
            "dimension": self.dimension,
            "sector_source": self.sector_source,
            "portfolio_return": self.portfolio_return,
            "benchmark_return": self.benchmark_return,
            "allocation_effect": self.allocation_effect,
            "selection_effect": self.selection_effect,
            "interaction_effect": self.interaction_effect,
            "total_active_return": self.total_active_return,
            "closure_residual": self.closure_residual,
            "groups": [g.as_dict() for g in self.groups],
            "waterfall": self.waterfall_data(),
            "sector_bar": self.sector_bar_data(),
            **self.meta,
        }

    # -- 可视化数据 ---------------------------------------------------------
    def waterfall_data(self) -> List[Dict[str, Any]]:
        """生成瀑布图序列数据。

        序列为：基准收益 r_b → +配置效应 → +选择效应 → +交互效应 → 组合收益 r_p。

        Returns:
            每个元素 ``{"name", "start", "change", "end"}``：
            首尾两根为总量柱（start=0），中间三根为浮动柱（效应变化）。
        """
        rb = self.benchmark_return
        rp = self.portfolio_return
        ar = self.allocation_effect
        sr = self.selection_effect
        ir = self.interaction_effect

        bars: List[Dict[str, Any]] = [
            {"name": "benchmark_return", "start": 0.0, "change": rb, "end": rb},
        ]
        cur = rb
        for name, eff in (
            ("allocation_effect", ar),
            ("selection_effect", sr),
            ("interaction_effect", ir),
        ):
            bars.append({"name": name, "start": cur, "change": eff, "end": cur + eff})
            cur += eff
        # 末根：组合收益总量柱，应等于 r_b + 三效应 = r_p
        bars.append({"name": "portfolio_return", "start": 0.0, "change": rp, "end": rp})
        return bars

    def sector_bar_data(self) -> List[Dict[str, Any]]:
        """生成分组堆叠条形图数据（各组 AR/SR/IR）。

        Returns:
            每个元素 ``{"sector", "ar", "sr", "ir", "total"}``。
        """
        return [g.as_dict() for g in self.groups]


# ---------------------------------------------------------------------------
# 核心归因器
# ---------------------------------------------------------------------------


class BrinsonAttribution:
    """机构级 Brinson 多因子绩效归因器。

    典型用法::

        res = BrinsonAttribution.attribute(
            sectors=["金融", "消费", "科技"],
            w_p=[0.4, 0.3, 0.3],
            w_b=[0.3, 0.4, 0.3],
            r_p_sector=[0.08, 0.05, 0.12],
            r_b_sector=[0.06, 0.04, 0.10],
            model="bhb",
        )
        print(res.total_active_return, res.closure_residual)
    """

    # -- mock 行业分组 -----------------------------------------------------
    @staticmethod
    def mock_sector(symbol: str, n_buckets: int = 5) -> str:
        """基于 symbol 稳定哈希的 mock 行业分组。

        使用 ``zlib.crc32`` 保证跨进程/跨运行稳定（不使用 Python 随机化 hash）。

        Args:
            symbol: 证券代码。
            n_buckets: 分组数，默认 5。

        Returns:
            形如 ``"MockSector_0"`` 的分组名。
        """
        idx = zlib.crc32(str(symbol).encode("utf-8")) % max(1, n_buckets)
        return f"MockSector_{idx}"

    # -- 输入校验 -----------------------------------------------------------
    @staticmethod
    def _validate(
        sectors: Sequence[str],
        w_p: Sequence[float],
        w_b: Sequence[float],
        r_p: Sequence[float],
        r_b: Sequence[float],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """校验并转换输入数组。

        Raises:
            ValueError: 长度不一致、权重不闭合。
        """
        n = len(sectors)
        arrays = [np.asarray(w_p, dtype=float), np.asarray(w_b, dtype=float),
                  np.asarray(r_p, dtype=float), np.asarray(r_b, dtype=float)]
        if any(a.size != n for a in arrays):
            raise ValueError(
                f"输入长度不一致: sectors={n}, w_p={arrays[0].size}, "
                f"w_b={arrays[1].size}, r_p={arrays[2].size}, r_b={arrays[3].size}"
            )
        w_p_a, w_b_a, r_p_a, r_b_a = arrays
        if abs(w_p_a.sum() - 1.0) > _WEIGHT_TOL:
            raise ValueError(
                f"组合权重之和={w_p_a.sum():.6f} 不等于 1（容差 {_WEIGHT_TOL}）"
            )
        if abs(w_b_a.sum() - 1.0) > _WEIGHT_TOL:
            raise ValueError(
                f"基准权重之和={w_b_a.sum():.6f} 不等于 1（容差 {_WEIGHT_TOL}）"
            )
        return w_p_a, w_b_a, r_p_a, r_b_a, np.asarray(sectors)

    # -- 单期归因 ----------------------------------------------------------
    @classmethod
    def attribute(
        cls,
        sectors: Sequence[str],
        w_p: Sequence[float],
        w_b: Sequence[float],
        r_p_sector: Sequence[float],
        r_b_sector: Sequence[float],
        model: str = "bhb",
        dimension: str = "industry",
        sector_source: str = "provided",
    ) -> BrinsonResult:
        """执行单期 Brinson 归因。

        Args:
            sectors: 分组名列表（行业/风格/市值分组）。
            w_p: 组合在各分组的权重，长度与 sectors 一致，和≈1。
            w_b: 基准在各分组的权重，和≈1。
            r_p_sector: 组合在各分组的收益。
            r_b_sector: 基准在各分组的收益。
            model: ``"bhb"`` 或 ``"fachler"``。
            dimension: 归因维度名，如 industry/style/market_cap。
            sector_source: 行业分类来源说明。

        Returns:
            ``BrinsonResult``，含各组明细、三效应总和与闭合残差。

        Raises:
            ValueError: model 非法、长度不一致、权重不闭合。
        """
        if model not in _VALID_MODELS:
            raise ValueError(f"model 必须为 {_VALID_MODELS} 之一，收到 {model!r}")

        w_p_a, w_b_a, r_p_a, r_b_a, sec_arr = cls._validate(
            sectors, w_p, w_b, r_p_sector, r_b_sector
        )

        # 组合/基准总收益
        rp = float(np.sum(w_p_a * r_p_a))
        rb = float(np.sum(w_b_a * r_b_a))
        # 配置效应中扣减的市场基准收益：两模型均引用整体基准收益
        rb_total = rb

        dw = w_p_a - w_b_a  # 主动权重
        dr = r_p_a - r_b_a  # 分组收益差

        if model == "bhb":
            # AR_i = (w_pi - w_bi) * (r_bi - r_b)
            ar_arr = dw * (r_b_a - rb)
        else:  # fachler
            # AR_i = (w_pi - w_bi) * (r_bi - r_b_total)
            ar_arr = dw * (r_b_a - rb_total)
        sr_arr = w_b_a * dr                 # SR_i = w_bi * (r_pi - r_bi)
        ir_arr = dw * dr                    # IR_i = (w_pi - w_bi) * (r_pi - r_bi)

        groups = [
            BrinsonGroupResult(
                sector=str(sec_arr[i]),
                w_p=float(w_p_a[i]), w_b=float(w_b_a[i]),
                r_p=float(r_p_a[i]), r_b=float(r_b_a[i]),
                ar=float(ar_arr[i]), sr=float(sr_arr[i]), ir=float(ir_arr[i]),
            )
            for i in range(len(sec_arr))
        ]

        ar_sum = float(np.sum(ar_arr))
        sr_sum = float(np.sum(sr_arr))
        ir_sum = float(np.sum(ir_arr))
        total_active = ar_sum + sr_sum + ir_sum
        closure_residual = total_active - (rp - rb)

        return BrinsonResult(
            groups=groups,
            allocation_effect=ar_sum,
            selection_effect=sr_sum,
            interaction_effect=ir_sum,
            total_active_return=total_active,
            portfolio_return=rp,
            benchmark_return=rb,
            closure_residual=closure_residual,
            model=model,
            dimension=dimension,
            sector_source=sector_source,
        )

    # -- 由持仓聚合后归因（可选 mock 分组） --------------------------------
    @classmethod
    def attribute_holdings(
        cls,
        holdings: Sequence[Dict[str, Any]],
        benchmark_holdings: Optional[Sequence[Dict[str, Any]]] = None,
        symbol_returns: Optional[Dict[str, float]] = None,
        benchmark_symbol_returns: Optional[Dict[str, float]] = None,
        sector_map: Optional[Dict[str, str]] = None,
        model: str = "bhb",
        dimension: str = "industry",
    ) -> BrinsonResult:
        """由持仓（symbol/weight）聚合到分组后归因。

        .. warning::
            若未提供 ``sector_map``，将使用基于 symbol 哈希的 **mock 5 组** 划分，
            结果中 ``sector_source`` 会标注为 ``"mock_simplified"``。

        Args:
            holdings: 组合持仓，每项 ``{"symbol": str, "weight": float}``。
            benchmark_holdings: 基准持仓，同构；缺省则视为与组合同权重基准。
            symbol_returns: 组合各 symbol 收益 ``{symbol: ret}``。
            benchmark_symbol_returns: 基准各 symbol 收益。
            sector_map: symbol→sector 映射；缺省用 mock 分组。
            model: bhb/fachler。
            dimension: 维度名。

        Returns:
            ``BrinsonResult``。
        """
        if benchmark_holdings is None:
            benchmark_holdings = list(holdings)

        def _group(hold: Sequence[Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
            agg: Dict[str, Tuple[float, float]] = {}
            for h in hold:
                sym = h["symbol"]
                w = float(h["weight"])
                sec = sector_map.get(sym) if sector_map else cls.mock_sector(sym)
                ret = (symbol_returns or {}).get(sym, 0.0)
                if sec in agg:
                    pw, pr = agg[sec]
                    agg[sec] = (pw + w, pr + ret * w)
                else:
                    agg[sec] = (w, ret * w)
            return agg

        # 简化：组合与基准使用同一分组集合
        all_secs: List[str] = []
        seen: set = set()
        for h in list(holdings) + list(benchmark_holdings):
            sym = h["symbol"]
            sec = sector_map.get(sym) if sector_map else cls.mock_sector(sym)
            if sec not in seen:
                seen.add(sec)
                all_secs.append(sec)

        def _agg_weights(hold: Sequence[Dict[str, Any]]) -> Dict[str, float]:
            out: Dict[str, float] = {}
            for h in hold:
                sym = h["symbol"]
                sec = sector_map.get(sym) if sector_map else cls.mock_sector(sym)
                out[sec] = out.get(sec, 0.0) + float(h["weight"])
            return out

        def _agg_ret(hold: Sequence[Dict[str, Any]],
                     ret_map: Optional[Dict[str, float]]) -> Dict[str, float]:
            # 分组收益 = Σ 个股权重*收益 / Σ 个股权重（加权）
            num: Dict[str, float] = {}
            den: Dict[str, float] = {}
            for h in hold:
                sym = h["symbol"]
                sec = sector_map.get(sym) if sector_map else cls.mock_sector(sym)
                w = float(h["weight"])
                r = (ret_map or {}).get(sym, 0.0)
                num[sec] = num.get(sec, 0.0) + w * r
                den[sec] = den.get(sec, 0.0) + w
            return {s: (num[s] / den[s] if den[s] else 0.0) for s in den}

        wp = _agg_weights(holdings)
        wb = _agg_weights(benchmark_holdings)
        rp = _agg_ret(holdings, symbol_returns)
        rb = _agg_ret(benchmark_holdings, benchmark_symbol_returns)

        sectors = all_secs
        w_p = [wp.get(s, 0.0) for s in sectors]
        w_b = [wb.get(s, 0.0) for s in sectors]
        r_p = [rp.get(s, 0.0) for s in sectors]
        r_b = [rb.get(s, 0.0) for s in sectors]

        sector_source = "provided" if sector_map else "mock_simplified"
        res = cls.attribute(sectors, w_p, w_b, r_p, r_b,
                            model=model, dimension=dimension,
                            sector_source=sector_source)
        if sector_source == "mock_simplified":
            res.meta["sector_classification_warning"] = (
                "行业分类为简化/mock 数据，P0-04 财务数据接入后切换真实分类"
            )
        return res

    # -- 多期归因 ----------------------------------------------------------
    @classmethod
    def multi_period_attribute(
        cls,
        period_inputs: Sequence[Dict[str, Any]],
        model: str = "bhb",
        method: str = "carino",
        dimension: str = "industry",
    ) -> Dict[str, Any]:
        """多期 Brinson 归因并链接。

        逐期调用 :meth:`attribute` 后，用 Cariño 平滑系数或 GRAP 算术链接把
        各期三效应缩放求和，使链接后总效应逼近全期复利超额收益。

        Args:
            period_inputs: 各期输入列表，每项需含
                ``sectors/w_p/w_b/r_p_sector/r_b_sector``。
            model: bhb/fachler。
            method: ``"carino"``（Cariño 平滑链接）或 ``"grap"``（算术链接）。
            dimension: 维度名。

        Returns:
            ``{"periods": [...], "linked": {三效应+总超额}, "compounded_active_return",
            "link_residual", "method"}``。

        Raises:
            ValueError: method 非法或期数为空。
        """
        if method not in ("carino", "grap"):
            raise ValueError(f"method 必须为 carino/grap，收到 {method!r}")
        if not period_inputs:
            raise ValueError("period_inputs 不能为空")

        period_results: List[BrinsonResult] = []
        for inp in period_inputs:
            res = cls.attribute(
                sectors=inp["sectors"],
                w_p=inp["w_p"],
                w_b=inp["w_b"],
                r_p_sector=inp["r_p_sector"],
                r_b_sector=inp["r_b_sector"],
                model=model,
                dimension=dimension,
                sector_source=inp.get("sector_source", "provided"),
            )
            period_results.append(res)

        # 逐期三效应与组合/基准收益
        ar_t = np.array([r.allocation_effect for r in period_results])
        sr_t = np.array([r.selection_effect for r in period_results])
        ir_t = np.array([r.interaction_effect for r in period_results])
        rp_t = np.array([r.portfolio_return for r in period_results])
        rb_t = np.array([r.benchmark_return for r in period_results])

        if method == "carino":
            # Cariño(1999) 平滑系数: k_t = (Rp,t - Rb,t) / ln((1+Rp,t)/(1+Rb,t))
            denom = np.log((1.0 + rp_t) / (1.0 + rb_t))
            active_t = rp_t - rb_t
            k_t = np.where(np.abs(denom) > 1e-12,
                           active_t / denom,
                           1.0)  # 退化为 1
            ar_link = float(np.sum(k_t * ar_t))
            sr_link = float(np.sum(k_t * sr_t))
            ir_link = float(np.sum(k_t * ir_t))
        else:  # grap：算术直接求和
            ar_link = float(np.sum(ar_t))
            sr_link = float(np.sum(sr_t))
            ir_link = float(np.sum(ir_t))

        # 全期复利收益与复利超额
        rp_comp = float(np.prod(1.0 + rp_t) - 1.0)
        rb_comp = float(np.prod(1.0 + rb_t) - 1.0)
        compounded_active = rp_comp - rb_comp

        linked_total = ar_link + sr_link + ir_link
        link_residual = linked_total - compounded_active

        return {
            "method": method,
            "model": model,
            "dimension": dimension,
            "periods": [
                {**r.as_dict(), "k_t": (float(k_t[i]) if method == "carino" else 1.0)}
                for i, r in enumerate(period_results)
            ],
            "linked": {
                "allocation_effect": ar_link,
                "selection_effect": sr_link,
                "interaction_effect": ir_link,
                "total_active_return": linked_total,
            },
            "portfolio_compounded_return": rp_comp,
            "benchmark_compounded_return": rb_comp,
            "compounded_active_return": compounded_active,
            "link_residual": link_residual,
        }
