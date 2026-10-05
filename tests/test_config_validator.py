"""配置校验器单元测试。

覆盖: 数据源/Jev/风控/回测/策略/账户/股票池各节的正常与异常校验。
"""
from __future__ import annotations

import copy

import pytest

from config.validator import (
    ValidationResult,
    _is_valid_symbol,
    validate_accounts_section,
    validate_backtest_section,
    validate_config,
    validate_data_section,
    validate_jev_section,
    validate_risk_section,
    validate_stock_pool,
    validate_strategies_section,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def valid_config() -> dict:
    """一份完全合法的配置。"""
    return {
        "data": {
            "provider": "quantdash",
            "api_key": "sk_292e39cbd4884df0bc9bac0d05d3686d",
            "cache_ttl_hours": 4,
        },
        "jev": {
            "base_url": "http://localhost:8765",
            "timeout": 5.0,
            "confidence_threshold": 0.6,
            "retry_count": 2,
            "mock_mode": True,
        },
        "risk": {
            "single_stop_loss": 0.03,
            "single_take_profit": 0.08,
            "max_drawdown_pause": 0.10,
            "max_position_per_symbol": 0.20,
            "max_total_position": 0.80,
            "daily_loss_limit": 0.02,
            "jev_confidence_threshold": 0.6,
        },
        "backtest": {
            "initial_capital": 1000000.0,
            "commission_rate": 0.00025,
            "stamp_tax_rate": 0.0005,
            "slippage_rate": 0.001,
        },
        "strategies": {
            "ma_cross": {
                "label": "双均线", "fast_period": 5, "slow_period": 20, "enabled": True,
            },
            "bollinger": {
                "label": "布林带", "period": 20, "num_std": 2.0, "enabled": True,
            },
            "momentum_breakout": {
                "label": "动量突破", "breakout_period": 20, "breakdown_period": 10, "enabled": True,
            },
        },
        "accounts": [
            {
                "account_id": "acc_1", "name": "稳健账户",
                "initial_capital": 1000000.0, "strategy": "ma_cross",
                "account_type": "simulated",
                "seed_positions": [["600519.SH", 0.45]],
            },
        ],
    }


@pytest.fixture
def valid_stock_pool() -> dict:
    return {
        "pool_meta": {"name": "测试池", "count": 2},
        "stocks": [
            {"symbol": "600519.SH", "name": "贵州茅台", "market_cap_yi": 15500, "daily_amount_yi": 30.8},
            {"symbol": "300750.SZ", "name": "宁德时代", "market_cap_yi": 13800, "daily_amount_yi": 90.6},
        ],
    }


# ---------------------------------------------------------------------------
# 标的代码格式
# ---------------------------------------------------------------------------

class TestSymbolValidation:
    @pytest.mark.parametrize("symbol", [
        "600519.SH", "000001.SZ", "300750.SZ", "688981.SH", "430047.BJ",
        "AAPL.US", "TSLA.US", "GOOGL.US",
        "00700.HK", "09988.HK",
    ])
    def test_valid_symbols(self, symbol):
        assert _is_valid_symbol(symbol)

    @pytest.mark.parametrize("symbol", [
        "", "600519", "600519.sh", "600519.SHH", "abc.SH",
        "AAPL.us", "aapl.US", "AAPL.US.X",
        "700.HK", "000700.HK",
        None, 123,
    ])
    def test_invalid_symbols(self, symbol):
        assert not _is_valid_symbol(symbol)


# ---------------------------------------------------------------------------
# 数据源校验
# ---------------------------------------------------------------------------

class TestDataSection:
    def test_valid(self, valid_config):
        r = ValidationResult()
        validate_data_section(valid_config, r)
        assert not r.has_errors

    def test_missing_section(self):
        r = ValidationResult()
        validate_data_section({}, r)
        assert r.has_errors
        assert any("数据源配置缺失" in e.message for e in r.errors)

    def test_invalid_provider(self):
        r = ValidationResult()
        validate_data_section({"data": {"provider": "invalid"}}, r)
        assert any("不支持的数据源类型" in e.message for e in r.errors)

    def test_empty_api_key(self):
        r = ValidationResult()
        validate_data_section({"data": {"provider": "quantdash", "api_key": ""}}, r)
        assert any("API Key不能为空" in e.message for e in r.errors)

    def test_bad_api_key_format(self):
        r = ValidationResult()
        validate_data_section({"data": {"provider": "quantdash", "api_key": "invalid_key"}}, r)
        assert any("API Key格式" in e.message for e in r.errors)

    def test_mock_provider_no_key_needed(self):
        r = ValidationResult()
        validate_data_section({"data": {"provider": "mock"}}, r)
        assert not r.has_errors

    def test_negative_cache_ttl(self):
        cfg = {"data": {"provider": "mock", "cache_ttl_hours": -1}}
        r = ValidationResult()
        validate_data_section(cfg, r)
        assert r.has_errors


# ---------------------------------------------------------------------------
# Jev校验
# ---------------------------------------------------------------------------

class TestJevSection:
    def test_valid(self, valid_config):
        r = ValidationResult()
        validate_jev_section(valid_config, r)
        assert not r.has_errors

    def test_missing_section(self):
        r = ValidationResult()
        validate_jev_section({}, r)
        assert len(r.warnings) == 1  # 缺失是warning不是error

    def test_invalid_url(self):
        r = ValidationResult()
        validate_jev_section({"jev": {"base_url": "not-a-url"}}, r)
        assert any("地址格式不合法" in e.message for e in r.errors)

    def test_negative_timeout(self):
        r = ValidationResult()
        validate_jev_section({"jev": {"timeout": -1}}, r)
        assert r.has_errors

    def test_threshold_out_of_range(self):
        r = ValidationResult()
        validate_jev_section({"jev": {"confidence_threshold": 1.5}}, r)
        assert r.has_errors

    def test_invalid_retry_count(self):
        r = ValidationResult()
        validate_jev_section({"jev": {"retry_count": "abc"}}, r)
        assert r.has_errors


# ---------------------------------------------------------------------------
# 风控校验
# ---------------------------------------------------------------------------

class TestRiskSection:
    def test_valid(self, valid_config):
        r = ValidationResult()
        validate_risk_section(valid_config, r)
        assert not r.has_errors

    def test_missing_section(self):
        r = ValidationResult()
        validate_risk_section({}, r)
        assert r.has_errors

    def test_stop_loss_too_high(self):
        cfg = {"risk": {"single_stop_loss": 0.5}}  # >0.2
        r = ValidationResult()
        validate_risk_section(cfg, r)
        assert any("单笔止损率" in e.message for e in r.errors)

    def test_stop_loss_zero(self):
        cfg = {"risk": {"single_stop_loss": 0}}  # 必须>0
        r = ValidationResult()
        validate_risk_section(cfg, r)
        assert any("单笔止损率" in e.message for e in r.errors)

    def test_daily_loss_too_high(self):
        cfg = {"risk": {"daily_loss_limit": 0.6}}  # >0.5
        r = ValidationResult()
        validate_risk_section(cfg, r)
        assert any("单日亏损限额" in e.message for e in r.errors)

    def test_per_symbol_exceeds_total(self):
        cfg = {"risk": {"max_position_per_symbol": 0.9, "max_total_position": 0.5}}
        r = ValidationResult()
        validate_risk_section(cfg, r)
        assert any("单标的仓位上限不能超过总仓位上限" in e.message for e in r.errors)

    def test_non_numeric_value(self):
        cfg = {"risk": {"single_stop_loss": "abc"}}
        r = ValidationResult()
        validate_risk_section(cfg, r)
        assert any("必须是数字" in e.message for e in r.errors)


# ---------------------------------------------------------------------------
# 回测校验
# ---------------------------------------------------------------------------

class TestBacktestSection:
    def test_valid(self, valid_config):
        r = ValidationResult()
        validate_backtest_section(valid_config, r)
        assert not r.has_errors

    def test_missing_section(self):
        r = ValidationResult()
        validate_backtest_section({}, r)
        assert len(r.warnings) == 1

    def test_zero_capital(self):
        r = ValidationResult()
        validate_backtest_section({"backtest": {"initial_capital": 0}}, r)
        assert any("初始资金必须>0" in e.message for e in r.errors)

    def test_negative_capital(self):
        r = ValidationResult()
        validate_backtest_section({"backtest": {"initial_capital": -100}}, r)
        assert r.has_errors

    def test_excessive_commission(self):
        r = ValidationResult()
        validate_backtest_section({"backtest": {"commission_rate": 0.1}}, r)
        assert any("佣金率" in e.message for e in r.errors)


# ---------------------------------------------------------------------------
# 策略校验
# ---------------------------------------------------------------------------

class TestStrategiesSection:
    def test_valid(self, valid_config):
        r = ValidationResult()
        validate_strategies_section(valid_config, r)
        assert not r.has_errors

    def test_empty_strategies(self):
        r = ValidationResult()
        validate_strategies_section({"strategies": {}}, r)
        assert r.has_errors

    def test_ma_fast_ge_slow(self):
        cfg = {"strategies": {"ma_cross": {"label": "x", "fast_period": 20, "slow_period": 5}}}
        r = ValidationResult()
        validate_strategies_section(cfg, r)
        assert any("快线周期必须小于慢线周期" in e.message for e in r.errors)

    def test_bollinger_std_out_of_range(self):
        cfg = {"strategies": {"bollinger": {"label": "x", "num_std": 10}}}
        r = ValidationResult()
        validate_strategies_section(cfg, r)
        assert any("布林带标准差倍数" in e.message for e in r.errors)

    def test_missing_label(self):
        cfg = {"strategies": {"ma_cross": {"fast_period": 5, "slow_period": 20}}}
        r = ValidationResult()
        validate_strategies_section(cfg, r)
        assert any("策略显示名称不能为空" in e.message for e in r.errors)


# ---------------------------------------------------------------------------
# 账户校验
# ---------------------------------------------------------------------------

class TestAccountsSection:
    def test_valid(self, valid_config):
        r = ValidationResult()
        validate_accounts_section(valid_config, r)
        assert not r.has_errors

    def test_empty_accounts(self):
        r = ValidationResult()
        validate_accounts_section({"accounts": [], "strategies": {}}, r)
        assert r.has_errors

    def test_duplicate_account_id(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"].append(copy.deepcopy(cfg["accounts"][0]))
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("账户ID重复" in e.message for e in r.errors)

    def test_zero_capital(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"][0]["initial_capital"] = 0
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("初始资金必须>0" in e.message for e in r.errors)

    def test_invalid_account_type(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"][0]["account_type"] = "paper"
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("账户类型必须是" in e.message for e in r.errors)

    def test_undefined_strategy(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"][0]["strategy"] = "nonexistent"
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("未在 strategies 节定义" in e.message for e in r.errors)

    def test_invalid_seed_symbol(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"][0]["seed_positions"] = [["BAD", 0.1]]
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("标的代码格式不合法" in e.message for e in r.errors)

    def test_seed_ratio_out_of_range(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"][0]["seed_positions"] = [["600519.SH", 1.5]]
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("仓位比例" in e.message for e in r.errors)

    def test_high_seed_ratio_warning(self, valid_config):
        cfg = copy.deepcopy(valid_config)
        cfg["accounts"][0]["seed_positions"] = [["600519.SH", 0.5], ["300750.SZ", 0.5]]
        r = ValidationResult()
        validate_accounts_section(cfg, r)
        assert any("种子持仓总比例" in w.message for w in r.warnings)


# ---------------------------------------------------------------------------
# 股票池校验
# ---------------------------------------------------------------------------

class TestStockPool:
    def test_valid(self, valid_stock_pool):
        r = ValidationResult()
        validate_stock_pool(valid_stock_pool, r)
        assert not r.has_errors

    def test_empty_pool(self):
        r = ValidationResult()
        validate_stock_pool({"stocks": []}, r)
        assert r.has_errors

    def test_invalid_symbol(self):
        pool = {"stocks": [{"symbol": "BAD", "name": "测试"}]}
        r = ValidationResult()
        validate_stock_pool(pool, r)
        assert any("标的代码格式不合法" in e.message for e in r.errors)

    def test_duplicate_symbol(self):
        pool = {"stocks": [
            {"symbol": "600519.SH", "name": "A"},
            {"symbol": "600519.SH", "name": "B"},
        ]}
        r = ValidationResult()
        validate_stock_pool(pool, r)
        assert any("标的代码重复" in e.message for e in r.errors)

    def test_missing_name(self):
        pool = {"stocks": [{"symbol": "600519.SH"}]}
        r = ValidationResult()
        validate_stock_pool(pool, r)
        assert any("标的名称不能为空" in e.message for e in r.errors)

    def test_negative_market_cap(self):
        pool = {"stocks": [{"symbol": "600519.SH", "name": "A", "market_cap_yi": -100}]}
        r = ValidationResult()
        validate_stock_pool(pool, r)
        assert any("必须>0" in e.message for e in r.errors)


# ---------------------------------------------------------------------------
# 全量校验
# ---------------------------------------------------------------------------

class TestValidateConfig:
    def test_full_valid(self, valid_config, valid_stock_pool):
        result = validate_config(valid_config, valid_stock_pool)
        assert result.ok
        assert not result.has_errors

    def test_full_invalid(self):
        result = validate_config({}, {})
        assert result.has_errors
        assert len(result.errors) >= 3  # 多个节缺失

    def test_result_summary(self, valid_config, valid_stock_pool):
        result = validate_config(valid_config, valid_stock_pool)
        summary = result.summary()
        assert "全部通过" in summary

    def test_result_summary_with_errors(self):
        result = validate_config({}, {})
        summary = result.summary()
        assert "错误" in summary
