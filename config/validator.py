"""配置启动校验器。

在服务启动时校验 config.yaml 和 stock_pool.yaml 的所有关键参数，
不合规直接报错退出，不带着错误配置运行。

校验级别:
- ERROR: 必须修复，否则服务不启动
- WARNING: 不阻断启动，但记录警告（如Jev服务不可用）
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# A股代码格式: 6位数字 + .SH/.SZ/.BJ
_A_SHARE_PATTERN = re.compile(r"^\d{6}\.(SH|SZ|BJ)$")
# 美股代码格式: 1-5位大写字母 + .US
_US_PATTERN = re.compile(r"^[A-Z]{1,5}\.US$")
# 港股代码格式: 4-5位数字 + .HK
_HK_PATTERN = re.compile(r"^\d{4,5}\.HK$")
# API Key格式: sk_前缀 + 至少32位十六进制字符
_API_KEY_PATTERN = re.compile(r"^sk_[0-9a-f]{32,}$")
# URL格式
_URL_PATTERN = re.compile(r"^https?://[^\s/$.?#].[^\s]*$")


@dataclass
class ValidationIssue:
    """单条校验问题。"""
    level: str  # "error" or "warning"
    field: str  # 配置字段路径，如 "risk.single_stop_loss"
    message: str  # 问题描述
    value: Any = None  # 实际值（可选）

    def __str__(self) -> str:
        val_str = f" (当前值: {self.value!r})" if self.value is not None else ""
        return f"[{self.level.upper()}] {self.field}: {self.message}{val_str}"


@dataclass
class ValidationResult:
    """校验结果汇总。"""
    errors: List[ValidationIssue] = field(default_factory=list)
    warnings: List[ValidationIssue] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return len(self.errors) > 0

    @property
    def ok(self) -> bool:
        return not self.has_errors

    def add_error(self, field: str, message: str, value: Any = None) -> None:
        self.errors.append(ValidationIssue("error", field, message, value))

    def add_warning(self, field: str, message: str, value: Any = None) -> None:
        self.warnings.append(ValidationIssue("warning", field, message, value))

    def summary(self) -> str:
        lines = []
        if self.errors:
            lines.append(f"❌ {len(self.errors)} 个错误（必须修复）:")
            for e in self.errors:
                lines.append(f"  {e}")
        if self.warnings:
            lines.append(f"⚠️  {len(self.warnings)} 个警告:")
            for w in self.warnings:
                lines.append(f"  {w}")
        if not self.errors and not self.warnings:
            lines.append("✅ 配置校验全部通过")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# 校验规则
# ---------------------------------------------------------------------------

def _validate_range(
    value: Any,
    field: str,
    low: float,
    high: float,
    result: ValidationResult,
    low_inclusive: bool = False,
    high_inclusive: bool = True,
    label: str = "",
) -> None:
    """校验数值是否在范围内。"""
    if value is None:
        result.add_error(field, f"{label}缺失")
        return
    try:
        v = float(value)
    except (TypeError, ValueError):
        result.add_error(field, f"{label}必须是数字", value)
        return
    if low_inclusive:
        if v < low:
            result.add_error(field, f"{label}必须≥{low}", v)
    else:
        if v <= low:
            result.add_error(field, f"{label}必须>{low}", v)
    if high_inclusive:
        if v > high:
            result.add_error(field, f"{label}必须≤{high}", v)
    else:
        if v >= high:
            result.add_error(field, f"{label}必须<{high}", v)


def validate_data_section(cfg: Dict[str, Any], result: ValidationResult) -> None:
    """校验数据源配置。"""
    data = cfg.get("data", {})
    if not data:
        result.add_error("data", "数据源配置缺失")
        return

    provider = data.get("provider", "")
    if provider not in ("quantdash", "mock"):
        result.add_error("data.provider", f"不支持的数据源类型: {provider}", provider)

    api_key = data.get("api_key", "")
    if provider == "quantdash":
        if not api_key:
            result.add_error("data.api_key", "QuantDash API Key不能为空")
        elif not _API_KEY_PATTERN.match(api_key):
            result.add_error(
                "data.api_key",
                "API Key格式应为 sk_ + 32位以上十六进制字符",
                api_key[:10] + "..." if len(api_key) > 10 else api_key,
            )

    cache_ttl = data.get("cache_ttl_hours")
    if cache_ttl is not None:
        _validate_range(cache_ttl, "data.cache_ttl_hours", 0, 168, result,
                        low_inclusive=True, label="缓存有效期")


def validate_jev_section(cfg: Dict[str, Any], result: ValidationResult) -> None:
    """校验Jev配置。"""
    jev = cfg.get("jev", {})
    if not jev:
        result.add_warning("jev", "Jev配置缺失，将使用默认值")
        return

    base_url = jev.get("base_url", "")
    if base_url and not _URL_PATTERN.match(base_url):
        result.add_error("jev.base_url", "Jev服务地址格式不合法", base_url)

    timeout = jev.get("timeout")
    if timeout is not None:
        _validate_range(timeout, "jev.timeout", 0, 60, result,
                        low_inclusive=False, high_inclusive=True, label="超时时间")

    threshold = jev.get("confidence_threshold")
    if threshold is not None:
        _validate_range(threshold, "jev.confidence_threshold", 0, 1, result,
                        low_inclusive=True, high_inclusive=True, label="置信度阈值")

    retry = jev.get("retry_count")
    if retry is not None:
        try:
            r = int(retry)
            if r < 0 or r > 10:
                result.add_error("jev.retry_count", "重试次数应在0-10之间", r)
        except (TypeError, ValueError):
            result.add_error("jev.retry_count", "重试次数必须是整数", retry)


def validate_risk_section(cfg: Dict[str, Any], result: ValidationResult) -> None:
    """校验风控参数。"""
    risk = cfg.get("risk", {})
    if not risk:
        result.add_error("risk", "风控配置缺失")
        return

    _validate_range(
        risk.get("single_stop_loss"),
        "risk.single_stop_loss", 0, 0.2, result,
        low_inclusive=False, high_inclusive=True, label="单笔止损率",
    )
    _validate_range(
        risk.get("single_take_profit"),
        "risk.single_take_profit", 0, 1.0, result,
        low_inclusive=False, high_inclusive=True, label="单笔止盈率",
    )
    _validate_range(
        risk.get("max_drawdown_pause"),
        "risk.max_drawdown_pause", 0, 0.5, result,
        low_inclusive=False, high_inclusive=True, label="最大回撤暂停阈值",
    )
    _validate_range(
        risk.get("max_position_per_symbol"),
        "risk.max_position_per_symbol", 0, 1.0, result,
        low_inclusive=False, high_inclusive=True, label="单标的仓位上限",
    )
    _validate_range(
        risk.get("max_total_position"),
        "risk.max_total_position", 0, 1.0, result,
        low_inclusive=False, high_inclusive=True, label="总仓位上限",
    )
    _validate_range(
        risk.get("daily_loss_limit"),
        "risk.daily_loss_limit", 0, 0.5, result,
        low_inclusive=False, high_inclusive=True, label="单日亏损限额",
    )

    # 逻辑校验: 单标的仓位 ≤ 总仓位
    per_sym = risk.get("max_position_per_symbol")
    total_pos = risk.get("max_total_position")
    if per_sym is not None and total_pos is not None:
        try:
            if float(per_sym) > float(total_pos):
                result.add_error(
                    "risk.max_position_per_symbol",
                    "单标的仓位上限不能超过总仓位上限",
                    f"per_symbol={per_sym}, total={total_pos}",
                )
        except (TypeError, ValueError):
            pass


def validate_backtest_section(cfg: Dict[str, Any], result: ValidationResult) -> None:
    """校验回测配置。"""
    bt = cfg.get("backtest", {})
    if not bt:
        result.add_warning("backtest", "回测配置缺失，将使用默认值")
        return

    capital = bt.get("initial_capital")
    if capital is not None:
        try:
            c = float(capital)
            if c <= 0:
                result.add_error("backtest.initial_capital", "初始资金必须>0", c)
        except (TypeError, ValueError):
            result.add_error("backtest.initial_capital", "初始资金必须是数字", capital)

    _validate_range(
        bt.get("commission_rate"),
        "backtest.commission_rate", 0, 0.01, result,
        low_inclusive=True, high_inclusive=True, label="佣金率",
    )
    _validate_range(
        bt.get("stamp_tax_rate"),
        "backtest.stamp_tax_rate", 0, 0.01, result,
        low_inclusive=True, high_inclusive=True, label="印花税率",
    )
    _validate_range(
        bt.get("slippage_rate"),
        "backtest.slippage_rate", 0, 0.05, result,
        low_inclusive=True, high_inclusive=True, label="滑点率",
    )


def validate_accounts_section(cfg: Dict[str, Any], result: ValidationResult) -> None:
    """校验账户配置。"""
    accounts = cfg.get("accounts", [])
    if not accounts:
        result.add_error("accounts", "至少需要配置一个模拟账户")
        return

    seen_ids = set()
    for i, acc in enumerate(accounts):
        prefix = f"accounts[{i}]"
        acc_id = acc.get("account_id", "")
        if not acc_id:
            result.add_error(f"{prefix}.account_id", "账户ID不能为空")
        elif acc_id in seen_ids:
            result.add_error(f"{prefix}.account_id", f"账户ID重复: {acc_id}", acc_id)
        else:
            seen_ids.add(acc_id)

        name = acc.get("name", "")
        if not name:
            result.add_error(f"{prefix}.name", "账户名称不能为空")

        capital = acc.get("initial_capital")
        if capital is None:
            result.add_error(f"{prefix}.initial_capital", "初始资金缺失")
        else:
            try:
                c = float(capital)
                if c <= 0:
                    result.add_error(f"{prefix}.initial_capital", "初始资金必须>0", c)
            except (TypeError, ValueError):
                result.add_error(f"{prefix}.initial_capital", "初始资金必须是数字", capital)

        acc_type = acc.get("account_type", "simulated")
        if acc_type not in ("simulated", "real"):
            result.add_error(f"{prefix}.account_type", "账户类型必须是 simulated 或 real", acc_type)

        strategy = acc.get("strategy", "")
        valid_strategies = cfg.get("strategies", {}).keys()
        if strategy and strategy not in valid_strategies:
            result.add_error(
                f"{prefix}.strategy",
                f"策略 '{strategy}' 未在 strategies 节定义",
                strategy,
            )

        # 校验种子持仓
        seeds = acc.get("seed_positions", [])
        total_ratio = 0.0
        for j, seed in enumerate(seeds):
            if not isinstance(seed, (list, tuple)) or len(seed) != 2:
                result.add_error(
                    f"{prefix}.seed_positions[{j}]",
                    "种子持仓格式应为 [symbol, ratio]",
                    seed,
                )
                continue
            sym, ratio = seed
            if not _is_valid_symbol(sym):
                result.add_error(
                    f"{prefix}.seed_positions[{j}].symbol",
                    "标的代码格式不合法",
                    sym,
                )
            try:
                r = float(ratio)
                if r <= 0 or r > 1:
                    result.add_error(
                        f"{prefix}.seed_positions[{j}].ratio",
                        "仓位比例应在(0, 1]",
                        r,
                    )
                total_ratio += r
            except (TypeError, ValueError):
                result.add_error(
                    f"{prefix}.seed_positions[{j}].ratio",
                    "仓位比例必须是数字",
                    ratio,
                )
        if total_ratio > 0.95:
            result.add_warning(
                f"{prefix}.seed_positions",
                f"种子持仓总比例 {total_ratio:.2%} 过高，建议保留现金",
                total_ratio,
            )


def validate_strategies_section(cfg: Dict[str, Any], result: ValidationResult) -> None:
    """校验策略配置。"""
    strategies = cfg.get("strategies", {})
    if not strategies:
        result.add_error("strategies", "至少需要配置一个策略")
        return

    for name, strat in strategies.items():
        prefix = f"strategies.{name}"
        if not isinstance(strat, dict):
            result.add_error(prefix, "策略配置必须是字典", strat)
            continue
        if not strat.get("label"):
            result.add_error(f"{prefix}.label", "策略显示名称不能为空")

        # 策略特定参数校验
        if name == "ma_cross":
            fast = strat.get("fast_period")
            slow = strat.get("slow_period")
            if fast is not None and slow is not None:
                try:
                    if int(fast) >= int(slow):
                        result.add_error(
                            f"{prefix}",
                            "快线周期必须小于慢线周期",
                            f"fast={fast}, slow={slow}",
                        )
                except (TypeError, ValueError):
                    pass
        elif name == "bollinger":
            _validate_range(
                strat.get("num_std"),
                f"{prefix}.num_std", 0.5, 5, result,
                low_inclusive=True, high_inclusive=True, label="布林带标准差倍数",
            )
        elif name == "momentum_breakout":
            bp = strat.get("breakout_period")
            bdp = strat.get("breakdown_period")
            if bp is not None and bdp is not None:
                try:
                    if int(bp) <= 0 or int(bdp) <= 0:
                        result.add_error(f"{prefix}", "突破周期必须>0")
                except (TypeError, ValueError):
                    pass


def _is_valid_symbol(symbol: str) -> bool:
    """校验标的代码格式是否合法。"""
    if not isinstance(symbol, str):
        return False
    return bool(
        _A_SHARE_PATTERN.match(symbol)
        or _US_PATTERN.match(symbol)
        or _HK_PATTERN.match(symbol)
    )


def validate_stock_pool(pool: Dict[str, Any], result: ValidationResult) -> None:
    """校验股票池配置。"""
    stocks = pool.get("stocks", [])
    if not stocks:
        result.add_error("stock_pool.stocks", "股票池为空，至少需要1只标的")
        return

    seen_symbols = set()
    for i, stock in enumerate(stocks):
        prefix = f"stock_pool.stocks[{i}]"
        symbol = stock.get("symbol", "")
        if not symbol:
            result.add_error(f"{prefix}.symbol", "标的代码不能为空")
        elif not _is_valid_symbol(symbol):
            result.add_error(f"{prefix}.symbol", "标的代码格式不合法（应为 600519.SH / AAPL.US / 00700.HK）", symbol)
        elif symbol in seen_symbols:
            result.add_error(f"{prefix}.symbol", f"标的代码重复: {symbol}", symbol)
        else:
            seen_symbols.add(symbol)

        if not stock.get("name"):
            result.add_error(f"{prefix}.name", "标的名称不能为空")

        # 市值和成交额应为正数
        for field_name in ("market_cap_yi", "daily_amount_yi"):
            val = stock.get(field_name)
            if val is not None:
                try:
                    if float(val) <= 0:
                        result.add_error(f"{prefix}.{field_name}", "必须>0", val)
                except (TypeError, ValueError):
                    result.add_error(f"{prefix}.{field_name}", "必须是数字", val)


# ---------------------------------------------------------------------------
# 主入口
# ---------------------------------------------------------------------------

def validate_config(
    main_cfg: Dict[str, Any],
    stock_pool: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """校验全部配置。

    Args:
        main_cfg: config.yaml 解析后的字典。
        stock_pool: stock_pool.yaml 解析后的字典（可选）。

    Returns:
        ValidationResult，包含 errors 和 warnings。
    """
    result = ValidationResult()

    validate_data_section(main_cfg, result)
    validate_jev_section(main_cfg, result)
    validate_risk_section(main_cfg, result)
    validate_backtest_section(main_cfg, result)
    validate_strategies_section(main_cfg, result)
    validate_accounts_section(main_cfg, result)

    if stock_pool is not None:
        validate_stock_pool(stock_pool, result)

    return result


def validate_and_exit_on_error(
    main_cfg: Dict[str, Any],
    stock_pool: Optional[Dict[str, Any]] = None,
) -> ValidationResult:
    """校验配置，有错误则打印并退出进程。

    Args:
        main_cfg: config.yaml 解析后的字典。
        stock_pool: stock_pool.yaml 解析后的字典（可选）。

    Returns:
        ValidationResult（仅在无错误时返回）。

    Raises:
        SystemExit: 校验不通过时退出，退出码 1。
    """
    result = validate_config(main_cfg, stock_pool)

    # 打印结果
    print("\n" + "=" * 60)
    print("配置启动校验")
    print("=" * 60)
    print(result.summary())
    print("=" * 60 + "\n")

    # 记录日志
    for e in result.errors:
        logger.error("配置校验错误: %s", e)
    for w in result.warnings:
        logger.warning("配置校验警告: %s", w)

    if result.has_errors:
        logger.critical("配置校验不通过，服务终止启动")
        raise SystemExit(1)

    logger.info("配置校验通过（%d 个警告）", len(result.warnings))
    return result
